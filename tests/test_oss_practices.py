"""开源最佳实践移植测试（alphalens / empyrical / 块自助Sharpe置信区间）。"""

import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine.backtest import (
    _bootstrap_sharpe_ci,
    _max_consecutive,
    calculate_metrics,
)
from commercial.factor_diagnostics import (
    _monotonic_label,
    _quintile_means,
)


def fake_portfolio(totals, dates):
    return SimpleNamespace(
        equity_curve=[
            {"date": d.strftime("%Y-%m-%d"), "total": float(t)} for d, t in zip(dates, totals)
        ],
        initial_cash=float(totals[0]),
        summary=lambda: {
            "gross_profit": 100.0,
            "gross_loss": -50.0,
            "total_trades": 10,
            "win_rate": 50.0,
            "avg_win": 10.0,
            "avg_loss": -5.0,
        },
        final_unsold={},
    )


class EmpyricalStyleMetricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.dates = pd.bdate_range("2025-01-01", periods=200)
        self.totals = 100000 * np.cumprod(1 + rng.normal(0.0008, 0.01, 200))
        self.metrics = calculate_metrics(fake_portfolio(self.totals, self.dates))

    def test_new_metric_keys_present(self):
        for key in (
            "sortino_ratio",
            "calmar_ratio",
            "max_consecutive_losses",
            "max_consecutive_wins",
            "sharpe_ci_low",
            "sharpe_ci_high",
            "sharpe_ci_includes_zero",
        ):
            self.assertIn(key, self.metrics)

    def test_sortino_sharpe_relationship(self):
        """下行波动≤总波动 → Sortino ≥ Sharpe（同均值口径）。"""
        self.assertGreaterEqual(self.metrics["sortino_ratio"], self.metrics["sharpe_ratio"])

    def test_calmar_consistent_with_components(self):
        expected = self.metrics["cagr_pct"] / 100 / abs(self.metrics["max_drawdown_pct"] / 100)
        self.assertAlmostEqual(self.metrics["calmar_ratio"], round(expected, 2), places=1)

    def test_max_consecutive_counts(self):
        returns = pd.Series([0.01, -0.01, -0.01, -0.02, 0.01, -0.01, 0.02])
        self.assertEqual(_max_consecutive(returns), (3, 1))

    def test_sharpe_ci_deterministic_and_ordered(self):
        rng = np.random.default_rng(9)
        returns = pd.Series(rng.normal(0.001, 0.01, 200))
        first = _bootstrap_sharpe_ci(returns)
        second = _bootstrap_sharpe_ci(returns)
        self.assertIsNotNone(first)
        self.assertEqual(first, second, "固定种子下自助区间必须可复现")
        self.assertLessEqual(first[0], first[1])

    def test_sharpe_ci_short_sample_none(self):
        rng = np.random.default_rng(9)
        self.assertIsNone(_bootstrap_sharpe_ci(pd.Series(rng.normal(0, 0.01, 15))))


class AlphalensStyleFactorTests(unittest.TestCase):
    def test_quintile_means_ascending_factor(self):
        rng = np.random.default_rng(7)
        values = np.tile(np.arange(50, dtype=float), 4)  # 4个日期 × 50只
        returns = values / 1000 + rng.normal(0, 0.001, len(values))
        means = _quintile_means(values, returns)
        self.assertEqual(len(means), 5)
        self.assertEqual(_monotonic_label(means), "递增")

    def test_monotonic_descending_and_noisy(self):
        self.assertEqual(_monotonic_label([0.05, 0.04, 0.03, 0.02, 0.01]), "递减")
        self.assertEqual(_monotonic_label([0.02, 0.05, 0.01, 0.04, 0.03]), "非单调")
        self.assertEqual(_monotonic_label(None), "数据不足")
        self.assertEqual(_monotonic_label([0.01, np.nan, 0.02, 0.03, 0.04]), "数据不足")

    def test_quintile_means_insufficient_sample(self):
        self.assertIsNone(_quintile_means(np.array([1.0]), np.array([0.01])))

    def test_diagnostics_output_contains_new_columns(self):
        from commercial.factor_diagnostics import compute_factor_diagnostics

        rng = np.random.default_rng(13)
        days = 160
        dates = pd.bdate_range("2025-01-01", periods=days)
        data = {}
        for index in range(40):
            code = f"sh.60{index:04d}"
            closes = 15 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, days)))
            preclose = np.concatenate([[closes[0]], closes[:-1]])
            volume = rng.lognormal(13, 0.3, days).round(0)
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
                "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, days)))).round(2),
                "open": 1000.0,
            }
        )
        summary, _corr, _meta = compute_factor_diagnostics(data, "2025-03-03", "2025-08-29")
        ok = summary[summary.get("status") == "ok"]
        self.assertFalse(ok.empty)
        for column in ("quantile_monotonic", "rank_autocorr"):
            self.assertIn(column, summary.columns)
        self.assertTrue(ok["rank_autocorr"].notna().any())
        self.assertTrue(ok["quantile_monotonic"].isin(["递增", "递减", "非单调"]).all())
        # 秩自相关取值合理（-1, 1]；同截面高度重叠时接近1。
        self.assertTrue(((ok["rank_autocorr"] > -1) & (ok["rank_autocorr"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
