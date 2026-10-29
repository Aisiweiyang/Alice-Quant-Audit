"""夜间优化轮测试：CSV健壮性、原子报告、数据直通、因子抽样、批量对比。"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", types.ModuleType("baostock"))

from commercial.compare import load_compare_plan, validate_overrides
from commercial.config import CommercialRequest, ParameterError
from commercial.data_input import load_csv_source, read_csv_any
from commercial.pipeline import run_full_analysis
from commercial.report import write_report_bundle
from engine.backtest import run as engine_run


def make_data(stocks=8, days=150, start="2025-01-01"):
    rng = np.random.default_rng(21)
    dates = pd.bdate_range(start, periods=days)
    data = {}
    for index in range(stocks):
        code = f"sh.60{index:04d}" if index % 2 == 0 else f"sz.00{index:04d}"
        steps = rng.normal(0.0003, 0.012, size=days)
        closes = 15 * np.exp(np.cumsum(steps))
        preclose = np.concatenate([[closes[0]], closes[:-1]])
        volume = rng.lognormal(13, 0.3, size=days).round(0)
        data[code] = pd.DataFrame(
            {
                "date": dates,
                "open": (closes * 1.001).round(3),
                "high": (closes * 1.012).round(3),
                "low": (closes * 0.99).round(3),
                "close": closes.round(3),
                "volume": volume,
                "amount": (volume * closes).round(0),
                "preclose": preclose.round(3),
                "pctChg": ((closes / preclose - 1) * 100).round(4),
                "tradestatus": 1,
                "isST": 0,
                "pe": 20.0,
                "pb": 2.0,
            }
        )
    data["sh.000300"] = pd.DataFrame(
        {
            "date": dates,
            "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=days)))).round(2),
            "open": 1000.0,
        }
    )
    return data


def write_cache(data, directory, name="market.pkl"):
    from data.cache_store import build_metadata, save_market_cache

    metadata = build_metadata(
        data,
        benchmark="sh.000300",
        adjust="2",
        universe_mode="file",
        source="unittest",
    )
    path = os.path.join(directory, name)
    save_market_cache(path, data, metadata)
    return path


def make_request(cache, output_root, **overrides):
    defaults = dict(
        cache=cache,
        start="2025-05-01",
        end="2025-07-15",
        output_root=output_root,
        quiet=True,
    )
    defaults.update(overrides)
    request = CommercialRequest(**defaults)
    request.validate()
    return request


class CsvRobustnessTests(unittest.TestCase):
    def _write_gbk_alias_csv(self, directory):
        dates = pd.bdate_range("2025-01-01", periods=40)
        frame = pd.DataFrame(
            {
                "trade_date": dates.strftime("%Y%m%d").astype(int),
                "open_price": 10.0,
                "high_price": 10.2,
                "low_price": 9.8,
                "close_price": 10.1,
                "vol": 1000,
                "成交额": 10000.0,
            }
        )
        path = os.path.join(directory, "sh.600000.csv")
        frame.to_csv(path, index=False, encoding="gbk")
        return path

    def test_gbk_encoding_alias_columns_and_compact_dates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_gbk_alias_csv(temp_dir)
            data, warnings, provenance = load_csv_source(temp_dir)
        self.assertIn("sh.600000", data)
        self.assertEqual(len(data["sh.600000"]), 40)
        self.assertTrue(data["sh.600000"]["date"].notna().all())
        self.assertIn("tradestatus", provenance.get("sh.600000", []))

    def test_mixed_date_formats_all_parse(self):
        series = pd.Series(["20250101", "2025/01/02", "2025-01-03", "2025/1/4"])
        from commercial.data_input import _parse_date_column

        parsed = _parse_date_column(series)
        self.assertTrue(parsed.notna().all())

    def test_read_csv_any_reports_unreadable(self):
        from commercial.data_input import DataInputError

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "x.csv")
            with open(path, "wb") as handle:
                handle.write(b"\xff\xfe\x00bad")
            with self.assertRaises(DataInputError):
                read_csv_any(path)


class AtomicBundleTests(unittest.TestCase):
    def test_failed_render_leaves_error_marker_not_partial_html(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir)
            analysis = run_full_analysis(request, report_id="atomic_test")
            root = analysis["report_dir"]
            # 先移除成功产物，验证失败重写不会留下新的半成品
            os.remove(os.path.join(root, "index.html"))
            analysis["manifest"] = None
            with patch("commercial.report.render_html", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    write_report_bundle(analysis["report_dir"], analysis)
            self.assertTrue(os.path.isfile(os.path.join(root, "ERROR.txt")))
            self.assertFalse(os.path.isfile(os.path.join(root, "index.html")))

    def test_success_bundle_has_no_staging_leftover(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, run_stress=False)
            analysis = run_full_analysis(request, report_id="atomic_ok")
            self.assertFalse(os.path.isdir(os.path.join(analysis["report_dir"], ".staging")))
            self.assertTrue(os.path.isfile(os.path.join(analysis["report_dir"], "index.html")))


class EngineDataPassthroughTests(unittest.TestCase):
    def test_engine_runs_directly_from_memory(self):
        data = make_data(stocks=4)
        portfolio, metrics = engine_run(
            "2025-05-01",
            "2025-07-15",
            verbose=False,
            data=data,
        )
        self.assertIsNotNone(portfolio)
        self.assertIsNotNone(metrics)

    def test_engine_requires_benchmark_in_memory_data(self):
        data = make_data(stocks=2)
        data.pop("sh.000300")
        with self.assertRaises(ValueError):
            engine_run("2025-05-01", "2025-07-15", verbose=False, data=data)


class FactorSamplingTests(unittest.TestCase):
    def test_sampling_disclosed_when_universe_exceeds_cap(self):
        from commercial.factor_diagnostics import compute_factor_diagnostics

        data = make_data(stocks=12)
        with patch("commercial.factor_diagnostics.MAX_UNIVERSE", 6):
            summary, corr, meta = compute_factor_diagnostics(data, "2025-05-01", "2025-07-15")
        self.assertTrue(meta["sampling"]["applied"])
        self.assertEqual(meta["sampling"]["used"], 6)
        self.assertIn("抽样", meta["sampling"]["note"])


class CompareTests(unittest.TestCase):
    def _plan(self, path, plans):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"plans": plans}, handle, ensure_ascii=False)

    def test_plan_rejects_non_whitelisted_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = os.path.join(temp_dir, "plan.json")
            self._plan(
                plan_path,
                [
                    {"label": "a", "overrides": {}},
                    {"label": "b", "overrides": {"custom_code": "import os"}},
                ],
            )
            with self.assertRaises(ParameterError):
                load_compare_plan(plan_path)

    def test_plan_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = os.path.join(temp_dir, "plan.json")
            self._plan(plan_path, [{"label": "only-one", "overrides": {}}])
            with self.assertRaises(ParameterError):
                load_compare_plan(plan_path)

    def test_overrides_range_validation(self):
        with self.assertRaises(ParameterError):
            validate_overrides({"slippage": 0.5})
        with self.assertRaises(ParameterError):
            validate_overrides({"factor_weights": {"unknown_factor": 1.0}})
        checked = validate_overrides({"max_positions": 5, "use_market_timing": False})
        self.assertEqual(checked["max_positions"], 5)
        self.assertFalse(checked["use_market_timing"])

    def test_compare_end_to_end(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            plan_path = os.path.join(temp_dir, "plan.json")
            self._plan(
                plan_path,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "重仓", "overrides": {"max_positions": 3}},
                    {"label": "高滑点", "overrides": {"slippage": 0.004}},
                ],
            )
            request = make_request(cache, temp_dir)
            request.output_root = temp_dir
            from commercial.compare import run_compare

            analysis = run_compare(request, plan_path, report_id="cmp_e2e")
            self.assertFalse(analysis["blocked"])
            ok = [r for r in analysis["plan_results"] if r["status"] == "ok"]
            self.assertEqual(len(ok), 3)
            root = analysis["report_dir"]
            for name in [
                "index.html",
                "executive_summary.md",
                "metrics.json",
                "strategy_audit.json",
                "scenario_comparison.csv",
                "data_quality.json",
                "manifest.json",
                "effective_config.json",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(root, name)), name)
            with open(os.path.join(root, "metrics.json"), encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload.get("mode"), "compare")

    def test_compare_blocks_not_ready_data(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            plan_path = os.path.join(temp_dir, "plan.json")
            self._plan(
                plan_path,
                [
                    {"label": "a", "overrides": {}},
                    {"label": "b", "overrides": {}},
                ],
            )
            request = make_request(cache, temp_dir, start="2026-05-01", end="2026-07-15")
            from commercial.compare import run_compare

            analysis = run_compare(request, plan_path, report_id="cmp_blocked")
            self.assertTrue(analysis["blocked"])
            self.assertEqual(analysis["plan_results"], [])


class ExportAndBrandTests(unittest.TestCase):
    def test_export_zip_contains_report_files(self):
        from commercial.report import export_report_zip

        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, run_stress=False)
            analysis = run_full_analysis(request, report_id="export_me")
            report_root = os.path.dirname(analysis["report_dir"])
            zip_path = export_report_zip(report_root, "export_me")
            self.assertTrue(os.path.isfile(zip_path))
            import zipfile

            with zipfile.ZipFile(zip_path) as bundle:
                names = bundle.namelist()
            self.assertTrue(any(n.endswith("index.html") for n in names))
            self.assertTrue(any(n.endswith("manifest.json") for n in names))

    def test_export_missing_report_raises(self):
        from commercial.report import export_report_zip

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                export_report_zip(temp_dir, "no_such_report")

    def test_branding_appears_in_report_and_manifest(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(
                cache, temp_dir, run_stress=False, brand_name="夜航研究", brand_footer="内部资料"
            )
            analysis = run_full_analysis(request, report_id="brand_check")
            with open(
                os.path.join(analysis["report_dir"], "index.html"), encoding="utf-8"
            ) as handle:
                page = handle.read()
            self.assertIn("夜航研究", page)
            self.assertIn("内部资料", page)
            self.assertEqual(analysis["manifest"]["parameters"].get("brand_name"), "夜航研究")


class ManifestTimingTests(unittest.TestCase):
    def test_manifest_contains_stage_timings(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, run_stress=False)
            analysis = run_full_analysis(request, report_id="timing_check")
            manifest_path = os.path.join(analysis["report_dir"], "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertIn("timings_seconds", manifest)
            self.assertIn("base_backtest_seconds", manifest["timings_seconds"])


if __name__ == "__main__":
    unittest.main()
