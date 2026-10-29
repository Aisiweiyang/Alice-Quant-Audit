"""Alice Quant Audit 商业化MVP测试：数据审计、评分、报告、管线与阻断行为。"""

import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", types.ModuleType("baostock"))

from commercial.config import (
    CommercialRequest,
    ParameterError,
    build_engine_config,
)
from commercial.data_audit import audit_data_credibility
from commercial.data_input import (
    load_csv_source,
    normalize_code,
    restrict_to_candidates,
    synthesize_benchmark,
)
from commercial.pipeline import run_data_audit_only, run_full_analysis
from commercial.report import (
    DISCLAIMER,
    create_report_directory,
)
from commercial.reliability_score import compute_reliability_score
from commercial.stress import default_scenarios, sensitivity_summary
from data.cache_store import build_metadata, save_market_cache
from strategy.config import StrategyConfig


def make_data(stocks=6, days=140, start="2025-01-01"):
    rng = np.random.default_rng(7)
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
    bench = pd.DataFrame(
        {
            "date": dates,
            "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=days)))).round(2),
            "open": 1000.0,
        }
    )
    data["sh.000300"] = bench
    return data


def write_cache(data, directory, name="market.pkl"):
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


def make_request(cache, output_root, validate=True, **overrides):
    defaults = dict(
        cache=cache,
        start="2025-05-01",
        end="2025-07-15",
        output_root=output_root,
        max_scenarios=2,
        quiet=True,
    )
    defaults.update(overrides)
    request = CommercialRequest(**defaults)
    if validate:
        request.validate()
    return request


class DataInputTests(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(normalize_code("600000"), "sh.600000")
        self.assertEqual(normalize_code("000001"), "sz.000001")
        self.assertEqual(normalize_code("sz.000001"), "sz.000001")
        self.assertIsNone(normalize_code("abc"))

    def test_csv_directory_import_records_field_provenance(self):
        data = make_data(stocks=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = os.path.join(temp_dir, "csv")
            os.makedirs(csv_dir)
            for code, frame in data.items():
                if code == "sh.000300":
                    continue
                frame[["date", "open", "high", "low", "close", "volume", "amount"]].to_csv(
                    os.path.join(csv_dir, f"{code}.csv"), index=False
                )
            loaded, warnings, provenance = load_csv_source(csv_dir)
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all("tradestatus" in fields for fields in provenance.values()))

    def test_candidates_filter_and_missing_report(self):
        data = make_data(stocks=4)
        filtered, hits, missing = restrict_to_candidates(data, ["sh.600000", "sh.600008", "600002"])
        self.assertEqual(hits, 2)
        self.assertIn("sh.000300", filtered)
        self.assertEqual(missing, ["sh.600008"])

    def test_synthesize_benchmark_creates_equal_weight_series(self):
        data = {k: v for k, v in make_data(stocks=3).items() if k != "sh.000300"}
        synthetic, code = synthesize_benchmark(data)
        self.assertEqual(code, "synthetic:equal_weight")
        self.assertIn(code, synthetic)
        self.assertGreater(len(synthetic[code]), 10)


def load_csv_dir(path):
    from commercial.data_input import load_csv_source

    return load_csv_source(path)


class DataAuditTests(unittest.TestCase):
    def test_ready_rating_for_complete_data(self):
        audit = audit_data_credibility(
            make_data(),
            start="2025-05-01",
            end="2025-07-15",
        )
        self.assertEqual(audit["rating"], "READY")
        self.assertFalse(audit["blockers"])

    def test_not_ready_blocks_when_range_outside_data(self):
        audit = audit_data_credibility(
            make_data(),
            start="2026-01-01",
            end="2026-06-30",
        )
        self.assertEqual(audit["rating"], "NOT_READY")
        self.assertTrue(audit["blockers"])

    def test_csv_import_with_placeholder_fields_is_limited(self):
        data = make_data(stocks=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            for code, frame in data.items():
                if code == "sh.000300":
                    continue
                frame[["date", "open", "high", "low", "close", "volume", "amount"]].to_csv(
                    os.path.join(temp_dir, f"{code}.csv"), index=False
                )
            loaded, warnings, provenance = load_csv_source(temp_dir)
            audit = audit_data_credibility(
                loaded,
                start="2025-05-01",
                end="2025-07-15",
                source_info={"field_provenance": provenance},
            )
        self.assertEqual(audit["rating"], "LIMITED")
        self.assertFalse(audit["fields"]["historical_trade_status"])
        self.assertTrue(any("占位" in w for w in audit["warnings"]))


class ConfigTests(unittest.TestCase):
    def test_whitelist_rejects_out_of_range_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = make_request("x.pkl", temp_dir, validate=False, commission_rate=0.5)
            with self.assertRaises(ParameterError):
                request.validate()

    def test_engine_config_does_not_touch_global(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = make_request("x.pkl", temp_dir, max_positions=5, slippage=0.002)
            view = build_engine_config(request)
            self.assertEqual(view.max_positions, 5)
            self.assertEqual(StrategyConfig.max_positions, 8)
            self.assertEqual(StrategyConfig.slippage, 0.001)


class StressTests(unittest.TestCase):
    def test_scenarios_capped_at_eight_with_ablation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = make_request("x.pkl", temp_dir, max_scenarios=8)
            request.extras["base_factor_weights"] = dict(StrategyConfig.factor_weights)
            scenarios = default_scenarios(request)
        self.assertEqual(len(scenarios), 8)
        keys = {s["key"] for s in scenarios}
        self.assertIn("ablation_no_momentum", keys)
        self.assertIn("baseline", keys)

    def test_sensitivity_flags_sign_flip(self):
        base = {
            "scenario": "baseline",
            "label": "基准",
            "status": "ok",
            "metrics": {"total_return_pct": 5.0, "sharpe_ratio": 1.0, "max_drawdown_pct": -8.0},
            "annualized_turnover": 10.0,
            "total_friction": 100.0,
        }
        flip = {
            "scenario": "cost_up_100",
            "label": "成本翻倍",
            "status": "ok",
            "metrics": {"total_return_pct": -6.0, "sharpe_ratio": -0.5, "max_drawdown_pct": -20.0},
            "annualized_turnover": 10.0,
            "total_friction": 300.0,
            "deltas": {
                "total_return_delta_pct": -11.0,
                "sharpe_delta": -1.5,
                "max_drawdown_delta_pct": -12.0,
                "turnover_delta": 0.0,
                "friction_delta": 200.0,
            },
        }
        summary = sensitivity_summary([base, flip])
        self.assertTrue(summary["over_sensitive"])
        self.assertEqual(summary["most_impactful"]["scenario"], "cost_up_100")


class ReliabilityScoreTests(unittest.TestCase):
    def test_full_evidence_scores_high(self):
        audit = audit_data_credibility(
            make_data(),
            start="2025-05-01",
            end="2025-07-15",
        )
        base = {
            "metrics": {
                "total_return_pct": 4.0,
                "sharpe_ratio": 1.2,
                "sharpe_reliable": True,
                "sharpe_observations": 80,
                "max_drawdown_pct": -6.0,
                "total_trades": 12,
                "unsold_final_positions": 0,
                "reliability_warnings": [],
            },
            "statistics": {"annualized_turnover": 12.0, "total_friction": 500.0},
        }
        stress = {
            "status": "ok",
            "over_sensitive": False,
            "ranking": [
                {
                    "scenario": "cost_up_100",
                    "label": "成本翻倍",
                    "return_delta_pct": -3.0,
                    "sharpe_delta": -0.2,
                    "sign_flipped": False,
                },
                {
                    "scenario": "no_market_timing",
                    "label": "无择时",
                    "return_delta_pct": 1.0,
                    "sharpe_delta": 0.1,
                    "sign_flipped": False,
                },
            ],
        }
        consistency = {
            "status": "ok",
            "verdict": "consistent",
            "detail": "一致",
            "note": "",
            "sharpe_gap": 0.1,
        }
        score = compute_reliability_score(
            data_audit=audit,
            base_result=base,
            segment_consistency=consistency,
            stress_summary=stress,
            manifest_complete=True,
            metadata_present=True,
        )
        self.assertGreaterEqual(score["score"], 85)
        self.assertEqual(score["insufficient_dimensions"], [])
        self.assertIn("不评价未来收益", score["declaration"])

    def test_not_ready_caps_score_below_fifty(self):
        audit = {
            "rating": "NOT_READY",
            "blockers": ["区间内没有可审计的股票数据"],
            "warnings": [],
            "coverage": {},
            "fields": {},
        }
        score = compute_reliability_score(
            data_audit=audit,
            base_result=None,
            segment_consistency=None,
            stress_summary=None,
        )
        self.assertIsNotNone(score["score"])
        self.assertLessEqual(score["score"], 39)

    def test_insufficient_dimensions_excluded_from_denominator(self):
        audit = {
            "rating": "READY",
            "blockers": [],
            "warnings": [],
            "coverage": {},
            "fields": {"historical_trade_status": True},
        }
        score = compute_reliability_score(
            data_audit=audit,
            base_result=None,
            segment_consistency={"status": "insufficient_data", "detail": "样本不足"},
            stress_summary={"status": "insufficient_data"},
        )
        self.assertIn("out_of_sample", score["insufficient_dimensions"])
        self.assertIn("cost_sensitivity", score["insufficient_dimensions"])
        self.assertLess(score["applicable_max_points"], 100)


class ReportTests(unittest.TestCase):
    def test_report_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first, report_id = create_report_directory(temp_dir, "demo_report_1")
            self.assertTrue(os.path.isdir(first))
            with self.assertRaises(FileExistsError):
                create_report_directory(temp_dir, "demo_report_1")
            second, _ = create_report_directory(temp_dir, None)
            self.assertNotEqual(second, first)

    def test_bundle_contains_all_deliverables(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, run_stress=True, max_scenarios=2)
            analysis = run_full_analysis(request, report_id="bundle_test_1")
            root = analysis["report_dir"]
            expected = [
                "index.html",
                "executive_summary.md",
                "metrics.json",
                "data_quality.json",
                "strategy_audit.json",
                "scenario_comparison.csv",
                "factor_diagnostics.csv",
                "effective_config.json",
                "manifest.json",
                "charts",
            ]
            for name in expected:
                self.assertTrue(os.path.exists(os.path.join(root, name)), name)
            manifest = analysis["manifest"]
            self.assertIn("index.html", manifest["outputs_sha256"])
            self.assertTrue(manifest["inputs"]["cache_sha256"])
            self.assertIn("params_digest_sha256", manifest["parameters"])
            with open(os.path.join(root, "executive_summary.md"), encoding="utf-8") as fh:
                summary = fh.read()
            self.assertIn(DISCLAIMER, summary)
            with open(os.path.join(root, "index.html"), encoding="utf-8") as fh:
                page = fh.read()
            for section in [
                "一页结论",
                "数据质量",
                "回测范围",
                "收益与回撤",
                "压力测试",
                "因子诊断",
                "风险暴露",
                "主要问题",
                "改进建议",
                "研究限制",
                "可复现信息",
                "免责声明",
            ]:
                self.assertIn(section, page)
            self.assertIn("不评价未来收益", page)


class FactorDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_summary_on_synthetic_data(self):
        from commercial.factor_diagnostics import compute_factor_diagnostics

        # 每个信号日截面至少需要8只股票才有统计意义的RankIC。
        data = make_data(stocks=12)
        summary, corr, meta = compute_factor_diagnostics(data, "2025-05-01", "2025-07-15")
        self.assertFalse(summary.empty)
        momentum = summary[summary["feature"] == "momentum"].iloc[0]
        self.assertEqual(momentum["status"], "ok")
        for column in [
            "coverage_pct",
            "mean_rank_ic",
            "rank_icir",
            "positive_ic_rate_pct",
            "ic_5d",
            "ic_20d",
        ]:
            self.assertIn(column, summary.columns)
        self.assertIn("duplicate_pairs", meta)
        self.assertIn("行业", meta["exposure_note"])

    def test_diagnostics_raise_when_signals_insufficient(self):
        from commercial.factor_diagnostics import FactorDataInsufficient
        from commercial.factor_diagnostics import compute_factor_diagnostics

        with self.assertRaises(FactorDataInsufficient):
            compute_factor_diagnostics(make_data(), "2025-07-10", "2025-07-18")

    def test_pipeline_can_skip_factor_diagnostics(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, run_stress=False, run_factor_diagnostics=False)
            analysis = run_full_analysis(request)
            self.assertIsNone(analysis["factor_diagnostics"])
            self.assertTrue(any("因子诊断" in n for n in analysis["notes"]))


class PipelineTests(unittest.TestCase):
    def test_full_analysis_end_to_end(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, run_stress=True, max_scenarios=2)
            analysis = run_full_analysis(request)
            self.assertFalse(analysis["blocked"])
            self.assertEqual(analysis["data_audit"]["rating"], "READY")
            self.assertIsNotNone(analysis["score"]["score"])
            self.assertIsNotNone(analysis["base"]["metrics"])
            self.assertIsNotNone(analysis["stress_summary"])
            self.assertEqual(len(analysis["stress_frame"]), 2)

    def test_not_ready_data_blocks_backtest(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, start="2026-05-01", end="2026-09-30")
            analysis = run_full_analysis(request)
            self.assertTrue(analysis["blocked"])
            self.assertIsNone(analysis["base"])
            metrics_path = os.path.join(analysis["report_dir"], "metrics.json")
            with open(metrics_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle).get("status"), "not_run")

    def test_candidates_limit_universe(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            candidates_path = os.path.join(temp_dir, "c.json")
            with open(candidates_path, "w", encoding="utf-8") as handle:
                json.dump({"candidates": [{"code": "sh.600000"}, {"code": "sz.000001"}]}, handle)
            request = make_request(cache, temp_dir, candidates=candidates_path, run_stress=False)
            analysis = run_full_analysis(request)
            self.assertEqual(analysis["candidate_info"]["matched"], 2)

    def test_audit_only_run(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir)
            analysis = run_data_audit_only(request)
            self.assertTrue(analysis["audit_only"])
            self.assertIsNone(analysis["base"])


if __name__ == "__main__":
    unittest.main()
