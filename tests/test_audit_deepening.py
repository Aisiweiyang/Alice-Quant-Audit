"""审计深化轮测试：截面压力场景、因子诊断图表、跨报告只读对比。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from commercial.config import CommercialRequest
from commercial.report import write_charts
from commercial.stress import default_scenarios, run_stress_tests, sensitivity_summary

from tests.test_cross_sectional_strategy import make_data, make_request, write_cache


class CrossSectionalStressTests(unittest.TestCase):
    def test_factor_scenario_set_excludes_buy_delay(self):
        request = make_request("dummy.pkl", run_stress=True)
        scenarios = default_scenarios(request, factor_strategy=True)
        keys = [s["key"] for s in scenarios]
        self.assertNotIn("buy_delay_1d", keys)
        self.assertIn("ablation_no_momentum", keys)
        self.assertLessEqual(len(keys), 8)

    def test_factor_stress_runs_and_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data()
            request = make_request(os.path.join(tmp, "x.pkl"), run_stress=True)
            from commercial.config import build_factor_strategy_config

            factor_config = build_factor_strategy_config(request)
            results = run_stress_tests(request, None, None, data=data, factor_config=factor_config)
            ok = [r for r in results if r.get("status") == "ok"]
            self.assertGreaterEqual(len(ok), 5, "多数截面场景应成功运行")
            baseline = next(r for r in results if r["scenario"] == "baseline")
            self.assertTrue(baseline.get("metrics"))
            ranked = [r for r in results if r["scenario"] != "baseline" and r.get("status") == "ok"]
            self.assertTrue(all("deltas" in r for r in ranked), "非基准场景应含与基准差异")
            summary = sensitivity_summary(results)
            self.assertEqual(summary.get("status"), "ok")
            self.assertFalse(summary.get("over_sensitive") is None)


class FactorDiagnosticChartTests(unittest.TestCase):
    def test_ic_decay_and_quantile_charts_written(self):
        factors = pd.DataFrame(
            [
                dict(
                    feature=f"f{i}",
                    status="ok",
                    ic_1d=0.01 * i,
                    ic_5d=0.02,
                    ic_10d=0.01,
                    ic_20d=0.0,
                    **{f"q{j}_mean_fwd_pct": 0.1 * j - 0.2 + i * 0.01 for j in range(1, 6)},
                )
                for i in range(4)
            ]
        )
        analysis = {"factor_diagnostics": factors}
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertIn("charts/ic_decay.png", written)
            self.assertIn("charts/quantile_returns.png", written)
            for name in ("ic_decay.png", "quantile_returns.png"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "charts", name)))
            self.assertIn("charts/ic_decay.png", analysis["charts_written"])

    def test_no_charts_when_diagnostics_missing(self):
        analysis = {"factor_diagnostics": None}
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertNotIn("charts/ic_decay.png", written)


class RollingOosFactorAndCITests(unittest.TestCase):
    """截面滚动样本外与窗口Sharpe置信带。"""

    def test_factor_rolling_oos_runs_with_ci(self):
        from commercial.config import build_factor_strategy_config
        from commercial.rolling_oos import run_rolling_oos

        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(stocks=12, days=220, seed=23)
            request = make_request(os.path.join(tmp, "x.pkl"))
            factor_config = build_factor_strategy_config(request)
            result = run_rolling_oos(
                request,
                None,
                None,
                windows=2,
                data=data,
                factor_config=factor_config,
            )
            ok = [w for w in result.get("windows", []) if w.get("status") == "ok"]
            self.assertTrue(ok, "至少部分窗口应可运行")
            for window in ok:
                self.assertIn("sharpe_ci_low", window)
                self.assertIn("sharpe_ci_includes_zero", window)
            if result.get("status") == "ok":
                self.assertIn("ci_zero_share", result)
                self.assertTrue(0.0 <= result["ci_zero_share"] <= 1.0)
                self.assertIn("Sharpe置信区间", result.get("note", ""))

    def test_classic_rolling_oos_unchanged_contract(self):
        from commercial.rolling_oos import run_rolling_oos
        from commercial.config import build_engine_config

        with tempfile.TemporaryDirectory() as tmp:
            cache = write_cache(make_data(), tmp)
            request = CommercialRequest(
                cache=cache,
                start="2025-02-01",
                end="2025-05-30",
                run_stress=False,
                run_rolling_oos=False,
                quiet=True,
            )
            request.validate()
            result = run_rolling_oos(
                request,
                build_engine_config(request),
                cache,
                windows=2,
                data=make_data(),
            )
            self.assertEqual(result["status"], "ok")
            self.assertIn("ci_zero_share", result)
            for window in result["windows"]:
                self.assertIn("sharpe_ci_low", window)


class ReportCompareTests(unittest.TestCase):
    def _make_report(self, root, report_id, score, ret):
        os.makedirs(os.path.join(root, report_id), exist_ok=True)
        payload = {
            "created_at": "2026-09-10T01:00:00+08:00",
            "blocked": False,
            "parameters": {"strategy": "classic_trend", "start": "2025-01-01", "end": "2025-06-30"},
        }
        with open(os.path.join(root, report_id, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        with open(os.path.join(root, report_id, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "total_return_pct": ret,
                    "max_drawdown_pct": -5.0,
                    "sharpe_ratio": 1.2,
                    "sortino_ratio": 1.5,
                    "calmar_ratio": 2.0,
                    "total_trades": 30,
                },
                handle,
            )
        with open(
            os.path.join(root, report_id, "strategy_audit.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"score": score, "grade_label": "可以继续研究"}, handle)

    def test_collect_and_missing_report(self):
        from commercial_mvp import collect_report_details

        with tempfile.TemporaryDirectory() as tmp:
            self._make_report(tmp, "rep_a", 80, 6.5)
            self._make_report(tmp, "rep_b", 65, 2.0)
            details = collect_report_details(tmp, ["rep_a", "rep_b", "ghost"])
            self.assertEqual(details[0]["report_id"], "rep_a")
            self.assertEqual(details[0]["score"], 80)
            self.assertEqual(details[0]["sortino_ratio"], 1.5)
            self.assertEqual(details[1]["total_return_pct"], 2.0)
            self.assertTrue(details[2].get("missing"))

    def test_order_preserved_not_resorted(self):
        from commercial_mvp import collect_report_details

        with tempfile.TemporaryDirectory() as tmp:
            self._make_report(tmp, "aaa", 50, 1.0)
            self._make_report(tmp, "zzz", 90, 9.0)
            details = collect_report_details(tmp, ["zzz", "aaa"])
            self.assertEqual([d["report_id"] for d in details], ["zzz", "aaa"])


if __name__ == "__main__":
    unittest.main()
