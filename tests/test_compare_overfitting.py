"""compare报告过拟合审计接入测试：PBO/DSR进对比HTML/CSV/metrics.json，
full模式--trials的Deflated Sharpe，CLI参数接入。"""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from commercial.config import CommercialRequest, ParameterError, request_from_args
from commercial.compare import run_compare
from commercial.pipeline import run_full_analysis
from tests.test_night_round import make_data, write_cache


def write_plan(path, plans):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"plans": plans}, handle, ensure_ascii=False)
    return path


class CompareOverfittingTests(unittest.TestCase):
    def _run_compare(self, temp_dir, plan_defs):
        cache = write_cache(make_data(stocks=8), temp_dir)
        # PBO需要T≥16块×5天：2月-7月中 ≈115个交易日。
        plan_path = write_plan(os.path.join(temp_dir, "plan.json"), plan_defs)
        request = CommercialRequest(
            cache=cache,
            start="2025-02-01",
            end="2025-07-15",
            output_root=temp_dir,
            quiet=True,
        )
        request.validate()
        return run_compare(request, plan_path)

    def test_three_plans_produce_pbo_and_per_plan_dsr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._run_compare(
                temp_dir,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "宽止损", "overrides": {"stop_loss_pct": 0.08}},
                    {"label": "高成本", "overrides": {"commission_rate": 0.0008}},
                ],
            )
            overfit = analysis.get("overfitting_audit") or {}
            self.assertEqual(overfit.get("n_trials"), 3)
            pbo = overfit.get("pbo") or {}
            self.assertEqual(pbo.get("status"), "ok")
            self.assertEqual(pbo.get("cscv_combinations"), 12870)
            self.assertTrue(0.0 <= pbo.get("pbo_probability") <= 1.0)
            ok = [r for r in analysis["plan_results"] if r["status"] == "ok"]
            for result in ok:
                dsr = result.get("deflated_sharpe") or {}
                self.assertEqual(dsr.get("status"), "ok")
                self.assertEqual(dsr.get("n_trials"), 3)
                self.assertIsNotNone(dsr.get("deflated_sharpe"))
            # HTML：过拟合小节+必配免责文案+DSR列。
            with open(
                os.path.join(analysis["report_dir"], "index.html"), encoding="utf-8"
            ) as handle:
                html = handle.read()
            self.assertIn("过拟合审计", html)
            self.assertIn("PBO衡量的是这组方案整体过拟合概率，不是对未来收益的预测", html)
            self.assertIn("DSR", html)
            # metrics.json机读 + CSV含DSR列。
            with open(
                os.path.join(analysis["report_dir"], "metrics.json"), encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            self.assertEqual((payload.get("overfitting_audit") or {}).get("n_trials"), 3)
            with open(
                os.path.join(analysis["report_dir"], "scenario_comparison.csv"),
                encoding="utf-8-sig",
            ) as handle:
                header = next(csv.reader(handle))
            self.assertIn("deflated_sharpe", header)

    def test_two_plans_pbo_honest_insufficient(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._run_compare(
                temp_dir,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "宽止损", "overrides": {"stop_loss_pct": 0.08}},
                ],
            )
            pbo = (analysis.get("overfitting_audit") or {}).get("pbo") or {}
            self.assertEqual(pbo.get("status"), "insufficient_data")
            self.assertIsNone(pbo.get("pbo_probability"))
            self.assertIn("≥3", pbo.get("detail") or "")
            # 方案级DSR仍计算（N=2）。
            ok = [r for r in analysis["plan_results"] if r["status"] == "ok"]
            self.assertTrue(all((r.get("deflated_sharpe") or {}).get("status") == "ok" for r in ok))

    def test_compare_dsr_has_no_user_declared_annotation(self):
        """对照用例：compare模式的N=完成回测方案数（系统实测），
        报告不得出现"用户声明值"标注（该标注仅属于full模式的--trials）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._run_compare(
                temp_dir,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "宽止损", "overrides": {"stop_loss_pct": 0.08}},
                    {"label": "高成本", "overrides": {"commission_rate": 0.0008}},
                ],
            )
            with open(
                os.path.join(analysis["report_dir"], "index.html"), encoding="utf-8"
            ) as handle:
                html = handle.read()
            self.assertIn("DSR", html)
            self.assertNotIn("用户声明值", html)

    def test_compare_report_and_summary_warn_against_single_window_pbo(self):
        """PBO随窗口可能大幅波动，compare报告与一页结论都要提示不可外推。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._run_compare(
                temp_dir,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "宽止损", "overrides": {"stop_loss_pct": 0.08}},
                    {"label": "高成本", "overrides": {"commission_rate": 0.0008}},
                ],
            )
            root = analysis["report_dir"]
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("单窗口数字不可外推", html)
            with open(os.path.join(root, "executive_summary.md"), encoding="utf-8") as handle:
                summary = handle.read()
            self.assertIn("单窗口数字不可外推", summary)

    def test_degradation_regression_shown_in_report_and_summary(self):
        """性能退化回归进compare报告HTML与一页结论。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._run_compare(
                temp_dir,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "宽止损", "overrides": {"stop_loss_pct": 0.08}},
                    {"label": "高成本", "overrides": {"commission_rate": 0.0008}},
                ],
            )
            root = analysis["report_dir"]
            pbo = (analysis.get("overfitting_audit") or {}).get("pbo") or {}
            degradation = pbo.get("degradation_regression") or {}
            self.assertIsNotNone(degradation.get("slope"))
            self.assertEqual(
                degradation.get("pairs"), pbo.get("cscv_combinations"), "成对观测数应等于CSCV组合数"
            )
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("性能退化回归", html)
            # 报告按_fmt两位小数显示。
            self.assertIn(str(round(degradation["slope"], 2)), html)
            with open(os.path.join(root, "executive_summary.md"), encoding="utf-8") as handle:
                summary = handle.read()
            self.assertIn("性能退化回归", summary)


class FullModeTrialsTests(unittest.TestCase):
    def test_trials_declared_dsr_in_metrics_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=8), temp_dir)
            request = CommercialRequest(
                cache=cache,
                start="2025-05-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
                run_stress=False,
                trials=5,
            )
            request.validate()
            analysis = run_full_analysis(request)
            root = analysis["report_dir"]
            metrics = (analysis.get("base") or {}).get("metrics") or {}
            self.assertEqual(metrics.get("deflated_sharpe_trials"), 5)
            self.assertIsNotNone(metrics.get("deflated_sharpe"))
            with open(os.path.join(root, "metrics.json"), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload.get("deflated_sharpe_trials"), 5)
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("Deflated Sharpe", html)
            self.assertIn("不是对未来收益的预测", html)
            manifest = analysis.get("manifest") or {}
            self.assertEqual((manifest.get("parameters") or {}).get("trials"), 5)

    def test_default_trials_one_degenerates_to_psr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=8), temp_dir)
            request = CommercialRequest(
                cache=cache,
                start="2025-05-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
                run_stress=False,
            )
            request.validate()
            analysis = run_full_analysis(request)
            metrics = (analysis.get("base") or {}).get("metrics") or {}
            self.assertEqual(metrics.get("deflated_sharpe_trials"), 1)
            self.assertIsNotNone(metrics.get("deflated_sharpe"))

    def test_dsr_trials_annotation_in_report_and_summary(self):
        """full模式DSR的"试验数N为用户声明值"透明标注必须出现在
        报告HTML与一页结论两处（N=1退化PSR时同样标注声明语义）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=8), temp_dir)
            request = CommercialRequest(
                cache=cache,
                start="2025-05-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
                run_stress=False,
                trials=5,
            )
            request.validate()
            analysis = run_full_analysis(request)
            root = analysis["report_dir"]
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("试验数N为用户声明值", html)
            self.assertIn("N>1时请确保如实申报", html)
            with open(os.path.join(root, "executive_summary.md"), encoding="utf-8") as handle:
                summary = handle.read()
            self.assertIn("Deflated Sharpe", summary)
            self.assertIn("为用户声明值", summary)
            self.assertIn("N>1时请确保如实申报", summary)


class DispersionCorrelationTableTests(unittest.TestCase):
    """compare报告相关矩阵小节。"""

    def test_matrix_renders_labels_order_and_missing_dash(self):
        from commercial.report import render_dispersion_correlation_table

        html = render_dispersion_correlation_table(
            {
                "correlations": [
                    {"pair": ["动量", "均衡"], "correlation": -0.11},
                    {"pair": ["价值", "动量"], "correlation": 0.1234},
                    {"pair": ["价值", "均衡"], "correlation": None},
                ],
            }
        )
        self.assertIn("相关矩阵", html)
        # 表头与行首标签按首次出现顺序：动量、均衡、价值。
        self.assertLess(html.index("<th>动量</th>"), html.index("<th>均衡</th>"))
        self.assertLess(html.index("<th>均衡</th>"), html.index("<th>价值</th>"))
        self.assertIn("<td>-0.1100</td>", html)
        self.assertIn("<td>0.1234</td>", html)
        self.assertIn("<td>—</td>", html, "None相关应显示—")
        self.assertIn("1.0000", html, "对角线恒为1")

    def test_empty_or_single_label_returns_empty_string(self):
        from commercial.report import render_dispersion_correlation_table

        self.assertEqual(render_dispersion_correlation_table({}), "")
        self.assertEqual(render_dispersion_correlation_table({"correlations": []}), "")
        self.assertEqual(
            render_dispersion_correlation_table(
                {"correlations": [{"pair": ["A", None], "correlation": 0.5}]}
            ),
            "",
        )

    def _run_compare(self, temp_dir, plan_defs):
        cache = write_cache(make_data(stocks=8), temp_dir)
        plan_path = write_plan(os.path.join(temp_dir, "plan.json"), plan_defs)
        request = CommercialRequest(
            cache=cache,
            start="2025-02-01",
            end="2025-07-15",
            output_root=temp_dir,
            quiet=True,
        )
        request.validate()
        return run_compare(request, plan_path)

    def test_compare_html_contains_correlation_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = self._run_compare(
                temp_dir,
                [
                    {"label": "基线", "overrides": {}},
                    {"label": "宽止损", "overrides": {"stop_loss_pct": 0.08}},
                    {"label": "高成本", "overrides": {"commission_rate": 0.0008}},
                ],
            )
            with open(
                os.path.join(analysis["report_dir"], "index.html"), encoding="utf-8"
            ) as handle:
                html = handle.read()
            self.assertIn("相关矩阵", html)
            self.assertIn("过拟合审计", html)


class CliWiringTests(unittest.TestCase):
    def test_trials_and_dormant_flags_now_reach_request(self):
        """修复记录：--no-factor-diagnostics/--no-rolling-oos/--rolling-windows
        此前未接入request_from_args（静默失效），本测试防止回归。"""
        from commercial_mvp import build_parser

        args = build_parser().parse_args(
            [
                "full",
                "--cache",
                "x.pkl",
                "--start",
                "2025-01-01",
                "--end",
                "2025-02-01",
                "--trials",
                "7",
                "--no-rolling-oos",
                "--rolling-windows",
                "4",
                "--no-factor-diagnostics",
            ]
        )
        request = request_from_args(args)
        self.assertEqual(request.trials, 7)
        self.assertFalse(request.run_rolling_oos)
        self.assertEqual(request.rolling_windows, 4)
        self.assertFalse(request.run_factor_diagnostics)

    def test_trials_out_of_range_rejected(self):
        request = CommercialRequest(
            cache="x.pkl",
            start="2025-01-01",
            end="2025-02-01",
            trials=0,
        )
        with self.assertRaises(ParameterError):
            request.validate()


if __name__ == "__main__":
    unittest.main()
