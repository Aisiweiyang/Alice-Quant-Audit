"""代码审核修复轮回归测试（2026-09-12逐模块审核发现的缺陷）。"""

import sys
import unittest

import pandas as pd

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from commercial.overfitting import deflated_sharpe, equity_daily_returns
from commercial.reliability_score import compute_reliability_score
from engine.backtest import _dated_returns


class WebExportSanitizationTests(unittest.TestCase):
    def test_safe_report_id_rejects_traversal(self):
        """[安全] /export/的report_id必须净化：'..'等构造返回None。"""
        import commercial_web

        self.assertIsNone(commercial_web.safe_report_id("../.."))
        self.assertIsNone(commercial_web.safe_report_id("..%2F..%2FWindows"))
        self.assertIsNone(commercial_web.safe_report_id("a/b/c"))
        self.assertIsNone(commercial_web.safe_report_id(""))
        self.assertIsNone(commercial_web.safe_report_id("_archive"))
        # 合法id（含时间戳/下划线/点/短横线）全部通过。
        self.assertEqual(
            commercial_web.safe_report_id("alice-framework-4_20260912_010101_000"),
            "alice-framework-4_20260912_010101_000",
        )
        self.assertEqual(commercial_web.safe_report_id("smoke_oss2"), "smoke_oss2")


class AnchoredScoreWordingTests(unittest.TestCase):
    def _score(self, rolling):
        return compute_reliability_score(
            data_audit={"rating": "READY"},
            base_result={"metrics": {"total_return_pct": 1.0}},
            segment_consistency=None,
            stress_summary=None,
            manifest_complete=True,
            metadata_present=True,
            rolling_oos=rolling,
        )

    def test_anchored_windows_not_described_as_non_overlapping(self):
        """[一致性] 锚定窗口的评分理由不得再写"不重叠窗口"。"""
        rolling = {
            "status": "ok",
            "verdict": "stable",
            "anchored": True,
            "positive_share": 1.0,
            "sharpe_std": 0.2,
            "windows": [{"status": "ok"}, {"status": "ok"}, {"status": "ok"}],
            "detail": "全部窗口为正收益",
            "note": "n/a",
        }
        score = self._score(rolling)
        reasons = "；".join(
            r for d in score["dimensions"] if d["key"] == "out_of_sample" for r in d["reasons"]
        )
        self.assertIn("锚定递进窗口", reasons)
        self.assertNotIn("不重叠窗口", reasons)

    def test_default_windows_keep_non_overlapping_wording(self):
        rolling = {
            "status": "ok",
            "verdict": "stable",
            "positive_share": 1.0,
            "sharpe_std": 0.2,
            "windows": [{"status": "ok"}],
            "detail": "x",
            "note": "n/a",
        }
        score = self._score(rolling)
        reasons = "；".join(
            r for d in score["dimensions"] if d["key"] == "out_of_sample" for r in d["reasons"]
        )
        self.assertIn("不重叠窗口", reasons)


class StressLabelTests(unittest.TestCase):
    def test_fewer_positions_label_matches_actual_value(self):
        """[文案] 场景标签应反映实际下调后的持仓数（4只时是"降至3只"不是"减半"）。

        该场景排第9、默认被MAX_SCENARIOS=8截断（设计行为），故临时放开上限取用。
        """
        from commercial import stress as stress_mod

        class _Req:
            max_positions = 4
            max_position_pct = 0.12
            max_scenarios = 9
            commission_rate = 0.0003
            stamp_tax_rate = 0.0005
            slippage = 0.001
            extras = {"base_factor_weights": {"momentum": 1.0}}

        original_cap = stress_mod.MAX_SCENARIOS
        stress_mod.MAX_SCENARIOS = 9
        try:
            scenarios = stress_mod.default_scenarios(_Req())
        finally:
            stress_mod.MAX_SCENARIOS = original_cap
        target = next(s for s in scenarios if s["key"] == "fewer_positions")
        self.assertEqual(target["label"], "持仓数量降至3只")
        self.assertEqual(target["overrides"]["max_positions"], 3)


class CleanlinessRegressionTests(unittest.TestCase):
    def test_equity_daily_returns_delegates_to_engine(self):
        """两处日收益提取实现合一：overfitting委托engine._dated_returns。"""
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2025-01-01", periods=5),
                "total": [100.0, 101.0, 99.0, 100.0, 102.0],
            }
        )
        delegated = equity_daily_returns(frame)
        direct = _dated_returns(frame)
        pd.testing.assert_series_equal(delegated, direct)

    def test_deflated_sharpe_rejects_zero_variance_returns(self):
        """常量收益（零方差）如实insufficient，不再依赖隐式分支。"""
        result = deflated_sharpe(pd.Series([0.01] * 50), n_trials=1)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["deflated_sharpe"])


if __name__ == "__main__":
    unittest.main()
