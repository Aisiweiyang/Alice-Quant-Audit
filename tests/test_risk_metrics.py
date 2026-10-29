"""风险与交易级指标测试（GitHub大型项目口径移植轮）。

覆盖：VaR/CVaR（empyrical）、偏度/超额峰度（scipy默认口径）、
Ulcer Index与Martin ratio（Peter G. Martin公开公式）、
交易级统计（backtesting.py _stats.py口径：期望值/最好最差/持仓天数/SQN）、
蒙特卡洛回撤分布（块自助重排）。
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from engine.backtest import (
    _mc_drawdown_stats,
    _ulcer_index,
    calculate_metrics,
)
from tests.test_oss_practices import fake_portfolio

NEW_KEYS = (
    "var_95_pct",
    "cvar_95_pct",
    "skewness",
    "excess_kurtosis",
    "ulcer_index_pct",
    "martin_ratio",
    "expectancy_pct",
    "best_trade_pct",
    "worst_trade_pct",
    "avg_trade_duration_days",
    "max_trade_duration_days",
    "sqn",
    "mc_max_drawdown_pct_p50",
    "mc_max_drawdown_pct_p95",
    "mc_terminal_return_p5_pct",
    "mc_paths",
)


def fake_trades(rows):
    """rows: (date, code, action, price, shares, pnl或None, cost或None)"""
    trades = []
    for date, code, action, price, shares, pnl, cost in rows:
        row = {"date": date, "code": code, "action": action, "price": price, "shares": shares}
        if pnl is not None:
            row["pnl"] = pnl
        if cost is not None:
            row["cost"] = cost
        trades.append(row)
    return trades


class TailDistributionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(101)
        self.returns = rng.normal(0.0006, 0.01, 200)
        self.totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + self.returns)])
        dates = pd.bdate_range("2025-01-01", periods=len(self.totals))
        portfolio = fake_portfolio(self.totals, dates)
        portfolio.trades = []
        self.metrics = calculate_metrics(portfolio)

    def test_var_cvar_match_empyrical_formula(self):
        """与empyrical实现逐值对照：VaR=P5，CVaR=底部cutoff段的均值。"""
        values = self.returns
        cutoff_index = int((len(values) - 1) * 0.05)
        partitioned = np.partition(values, cutoff_index)
        expected_var = partitioned[cutoff_index] * 100
        expected_cvar = partitioned[: cutoff_index + 1].mean() * 100
        self.assertAlmostEqual(self.metrics["var_95_pct"], round(float(expected_var), 2), places=2)
        self.assertAlmostEqual(
            self.metrics["cvar_95_pct"], round(float(expected_cvar), 2), places=2
        )
        self.assertLessEqual(
            self.metrics["cvar_95_pct"], self.metrics["var_95_pct"] + 1e-9, "CVaR应不浅于（≤）VaR"
        )

    def test_skew_zero_for_symmetric_returns(self):
        values = np.tile([0.01, -0.01, 0.02, -0.02], 25)
        totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + values)])
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        portfolio = fake_portfolio(totals, dates)
        portfolio.trades = []
        metrics = calculate_metrics(portfolio)
        self.assertAlmostEqual(metrics["skewness"], 0.0, places=9, msg="对称收益偏度应为0")

    def test_excess_kurtosis_matches_manual_moments(self):
        centered = self.returns - self.returns.mean()
        m2 = float(np.mean(centered**2))
        expected = float(np.mean(centered**4)) / (m2**2) - 3.0
        self.assertAlmostEqual(self.metrics["excess_kurtosis"], round(expected, 3), places=2)


class UlcerIndexTests(unittest.TestCase):
    def test_zero_for_monotonic_rise(self):
        equity = np.array([100.0, 101, 102, 103, 104])
        self.assertEqual(_ulcer_index(equity), 0.0)

    def test_hand_computed_value(self):
        # 路径 [100, 110, 99]：dd深度 = [0, 0, 10]% → UI = sqrt(100/3) ≈ 5.7735
        ui = _ulcer_index(np.array([100.0, 110, 99]))
        self.assertAlmostEqual(ui, float(np.sqrt(100.0 / 3)), places=6)

    def test_martin_ratio_none_without_drawdown(self):
        values = np.tile([0.01], 60)
        totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + values)])
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        portfolio = fake_portfolio(totals, dates)
        portfolio.trades = []
        metrics = calculate_metrics(portfolio)
        self.assertEqual(metrics["ulcer_index_pct"], 0.0)
        self.assertIsNone(metrics["martin_ratio"], "无回撤时Martin不可定义，如实None")


class TradeLevelStatsTests(unittest.TestCase):
    def _metrics_with_trades(self, rows, totals=None, n_days=80):
        if totals is None:
            rng = np.random.default_rng(7)
            returns = rng.normal(0.0006, 0.008, n_days)
            totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + returns)])
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        portfolio = fake_portfolio(totals, dates)
        portfolio.trades = fake_trades(rows)
        return calculate_metrics(portfolio)

    def test_round_trip_pairing_and_stats(self):
        rows = [
            ("2025-01-02", "sh.600000", "BUY", 10.0, 1000, None, 10000.0),
            ("2025-01-10", "sh.600000", "SELL", 11.0, 1000, 800.0, None),
            ("2025-02-03", "sz.000001", "BUY", 20.0, 500, None, 10000.0),
            ("2025-02-05", "sz.000001", "SELL", 19.0, 500, -600.0, None),
        ]
        metrics = self._metrics_with_trades(rows)
        self.assertAlmostEqual(
            metrics["expectancy_pct"], 1.0, places=6, msg="期望值=(+8% + -6%)/2 = 1%"
        )
        self.assertAlmostEqual(metrics["best_trade_pct"], 8.0, places=6)
        self.assertAlmostEqual(metrics["worst_trade_pct"], -6.0, places=6)
        self.assertAlmostEqual(metrics["avg_trade_duration_days"], 5.0, places=6)
        self.assertEqual(metrics["max_trade_duration_days"], 8)

    def test_sqn_requires_thirty_trades(self):
        rows = []
        dates = pd.bdate_range("2025-01-01", periods=80).strftime("%Y-%m-%d")
        for index in range(40):
            day_in, day_out = dates[2 * index], dates[2 * index + 1]
            pnl = 100.0 if index % 2 == 0 else -50.0
            rows.append((day_in, f"sh.60{index:04d}", "BUY", 10.0, 100, None, 1000.0))
            rows.append((day_out, f"sh.60{index:04d}", "SELL", 10.2, 100, pnl, None))
        metrics = self._metrics_with_trades(rows)
        self.assertIsNotNone(metrics["sqn"])
        pnl_std = float(np.std([100.0 if i % 2 == 0 else -50.0 for i in range(40)], ddof=1))
        expected = np.sqrt(40) * np.mean([100.0, -50.0] * 20) / pnl_std
        self.assertAlmostEqual(metrics["sqn"], round(float(expected), 2), places=2)

        short = rows[:20]  # 只有10个回合
        metrics_short = self._metrics_with_trades(short)
        self.assertIsNone(metrics_short["sqn"], "不足30笔时SQN应如实None")


class MonteCarloDrawdownTests(unittest.TestCase):
    def test_deterministic_and_ordered(self):
        rng = np.random.default_rng(55)
        returns = pd.Series(rng.normal(0.0006, 0.01, 200))
        first = _mc_drawdown_stats(returns)
        second = _mc_drawdown_stats(returns)
        self.assertEqual(first, second, "固定种子必须可复现")
        self.assertLessEqual(
            first["mc_max_drawdown_pct_p50"],
            first["mc_max_drawdown_pct_p95"],
            "回撤深度p50应浅于（≥）p95（负号口径）",
        )
        self.assertTrue(np.isfinite(first["mc_terminal_return_p5_pct"]))
        self.assertEqual(first["mc_paths"], 1000)

    def test_insufficient_sample_none(self):
        result = _mc_drawdown_stats(pd.Series([0.01, -0.01] * 20))
        self.assertIsNone(result)

    def test_shuffling_finds_deeper_drawdowns_than_alternating_order(self):
        """交替正负序列的已实现回撤≈单日深度；重排会让负收益聚集，
        其p95分位应严格深于已实现回撤（负号口径：更小的数）。"""
        returns = pd.Series(np.tile([0.01, -0.01], 100))
        equity_realized = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity_realized)
        realized_dd = float(((equity_realized - running_max) / running_max).min() * 100)
        stats = _mc_drawdown_stats(returns)
        self.assertIsNotNone(stats)
        self.assertLess(
            stats["mc_max_drawdown_pct_p95"],
            realized_dd,
            "重排分布p95应比交替序列的已实现回撤更深",
        )


class ReportIntegrationTests(unittest.TestCase):
    def test_full_report_contains_new_sections(self):
        from commercial.config import CommercialRequest
        from commercial.pipeline import run_full_analysis
        from tests.test_night_round import make_data, write_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=8), temp_dir)
            # MC门槛要求≥60个收益观测，故用约74个交易日的窗口。
            request = CommercialRequest(
                cache=cache,
                start="2025-04-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
                run_stress=False,
            )
            request.validate()
            analysis = run_full_analysis(request)
            root = analysis["report_dir"]
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            for keyword in (
                "蒙特卡洛重构",
                "Ulcer Index",
                "VaR95",
                "SQN",
                "最长持仓",
                "统计重构",
                "仅供研究",
            ):
                self.assertIn(keyword, html, keyword)
            with open(os.path.join(root, "metrics.json"), encoding="utf-8") as handle:
                import json

                payload = json.load(handle)
            for key in NEW_KEYS:
                self.assertIn(key, payload, key)


if __name__ == "__main__":
    unittest.main()
