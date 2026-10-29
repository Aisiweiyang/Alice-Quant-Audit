import math
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

# 核心逻辑测试不访问行情网络；本机未安装Baostock时提供导入占位。
sys.modules.setdefault("baostock", types.ModuleType("baostock"))

from engine.backtest import (
    _load_market_cache,
    _passes_strategy_filters,
    _rebalance_dates,
    calculate_metrics,
    run,
    select_candidates,
)
from data.cache_store import build_metadata, save_market_cache
from engine.broker import Broker
from engine.portfolio import Portfolio


class Config:
    initial_capital = 100_000
    max_positions = 10
    max_position_pct = 0.10
    commission_rate = 0.0003
    stamp_tax_rate = 0.001
    slippage = 0.001
    min_commission = 5.0
    consecutive_up_days = 2
    max_daily_gain_pct = 3.0
    min_score = 5.0
    factor_weights = {}
    use_pe_filter = False
    pe_min = 0
    pe_max = 80
    use_pb_filter = False
    pb_min = 0.5
    pb_max = 10
    top_volume_n = 500


def price_frame(periods=65):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    closes = pd.Series(range(periods), dtype=float) * 0.05 + 10
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": 1_000_000,
            "amount": 10_000_000,
            "pctChg": 1.0,
            "pe": 20.0,
            "pb": 2.0,
        }
    )
    return frame


class PortfolioTests(unittest.TestCase):
    def test_total_value_is_callable_and_trade_pnl_includes_buy_fee(self):
        broker = Broker(Config)
        portfolio = Portfolio(100_000, broker, 10, 0.10)

        self.assertTrue(portfolio.buy("sh.600000", 10.0, "2025-01-02", 100_000))
        self.assertFalse(portfolio.buy("sh.600000", 10.0, "2025-01-02", 100_000))
        self.assertGreater(portfolio.total_value({"sh.600000": 10.0}), 0)
        cost_basis = portfolio.holdings["sh.600000"]["cost_basis"]
        self.assertTrue(portfolio.sell("sh.600000", 10.0, "2025-01-03"))

        trade = portfolio.trades[-1]
        self.assertAlmostEqual(trade["pnl"], trade["revenue"] - cost_basis)
        self.assertLess(trade["pnl"], 0)

    def test_flat_equity_metrics_are_finite(self):
        portfolio = Portfolio(100_000, Broker(Config), 10, 0.10)
        portfolio.record("2025-01-02", {})
        portfolio.record("2025-01-03", {})

        metrics = calculate_metrics(portfolio)
        self.assertEqual(metrics["annual_volatility_pct"], 0)
        self.assertEqual(metrics["sharpe_ratio"], 0)
        for value in metrics.values():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))


class SelectionTests(unittest.TestCase):
    def test_strategy_filter_rejects_fast_gain(self):
        frame = price_frame()
        frame.loc[frame.index[-1], "pctChg"] = 4.0
        self.assertFalse(_passes_strategy_filters(frame, Config))

    def test_candidate_selection_ignores_future_rows(self):
        frame = price_frame(66)
        signal_date = frame.iloc[-2]["date"]
        frame.loc[frame.index[-1], ["pctChg", "amount"]] = [20.0, 0.0]

        with (
            patch("strategy.selection.compute_all", return_value={"momentum": 1}),
            patch("strategy.selection.score", return_value=6.0),
        ):
            selected = select_candidates({"sh.600000": frame}, signal_date, config=Config)

        self.assertEqual([code for code, _ in selected], ["sh.600000"])

    def test_weekly_rebalance_uses_first_trading_day_after_holiday(self):
        dates = [pd.Timestamp("2025-10-08"), pd.Timestamp("2025-10-09"), pd.Timestamp("2025-10-10")]
        self.assertEqual(_rebalance_dates(dates, "weekly", "monday"), {dates[0]})

    def test_offline_backtest_completes_end_to_end(self):
        frame = price_frame(90)
        data = {"sh.600000": frame}
        benchmark = frame[["date", "close"]].copy()

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch.dict(os.environ, {"ALICE_MARKET_CACHE_DIR": cache_dir}),
            patch("engine.backtest.get_liquid_universe", return_value=None),
            patch("engine.backtest.get_stock_list", return_value=["sh.600000"]),
            patch("engine.backtest.get_batch_data", return_value=data),
            patch("engine.backtest.get_index_data", return_value=benchmark),
            patch("engine.backtest.select_candidates", return_value=[("sh.600000", 6.0)]),
        ):
            portfolio, metrics = run("2025-04-01", "2025-04-30", verbose=False)

        self.assertIsNotNone(portfolio)
        self.assertGreaterEqual(metrics["total_trades"], 1)
        self.assertEqual(
            len({row["date"] for row in portfolio.equity_curve}), len(portfolio.equity_curve)
        )
        self.assertGreater(len(portfolio.rebalance_decisions), 0)
        selected = [
            item for decision in portfolio.rebalance_decisions for item in decision["selected"]
        ]
        self.assertTrue(all(item["score"] == 6.0 for item in selected))

    def test_formal_backtest_rejects_incomplete_cache(self):
        stock = price_frame(65)
        benchmark = stock[["date", "close"]].copy()
        data = {"sh.600000": stock, "sh.000300": benchmark}
        metadata = build_metadata(
            data,
            benchmark="sh.000300",
            adjust="2",
            universe_mode="file",
            source="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "market.pkl")
            save_market_cache(path, data, metadata)
            with self.assertRaises(ValueError):
                _load_market_cache(path, allow_incomplete_data=False)
            loaded = _load_market_cache(path, allow_incomplete_data=True)
        self.assertIn("sh.600000", loaded)


if __name__ == "__main__":
    unittest.main()
