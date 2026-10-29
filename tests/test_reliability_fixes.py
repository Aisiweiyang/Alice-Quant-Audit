"""商业化MVP前置的可信度修复回归测试。

覆盖：期末清仓真实性、因子回测执行规则统一、持仓缺价估值、
动量周期口径、turnover更名、标准ADX、短样本Sharpe防误导。
"""

import sys
import types
import unittest

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", types.ModuleType("baostock"))

from engine.backtest import calculate_metrics
from engine.broker import Broker
from engine.execution import closing_sale_is_executable
from engine.factor_backtest import run_factor_backtest
from engine.portfolio import Portfolio
from strategy.config import StrategyConfig
from strategy.factor_config import FactorStrategyConfig
from strategy.factors import compute_all, n_day_return, wilder_adx


def price_frame(periods=70, base=10.0, step=0.0):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    closes = base + step * np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes + 0.05,
            "low": closes - 0.05,
            "close": closes,
            "volume": 1_000_000,
            "amount": 10_000_000,
            "pctChg": 0.0 if step == 0 else step / base * 100,
            "preclose": closes - step,
            "tradestatus": 1,
            "isST": 0,
            "pe": 20.0,
            "pb": 2.0,
        }
    )


class MomentumDefinitionTests(unittest.TestCase):
    def test_n_day_return_spans_exactly_n_sessions(self):
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 16.0])
        # 5日涨跌幅 = 最新收盘 / 5个交易日前收盘 - 1
        self.assertAlmostEqual(n_day_return(closes, 5), (16.0 / 10.0 - 1) * 100)

    def test_n_day_return_needs_n_plus_one_prices(self):
        self.assertIsNone(n_day_return(np.array([10.0, 11.0]), 5))

    def test_compute_all_momentum_matches_compounded_pctchg(self):
        frame = price_frame(40)
        factors = compute_all(frame, StrategyConfig)
        expected = (frame["close"].iloc[-1] / frame["close"].iloc[-6] - 1) * 100
        self.assertAlmostEqual(factors["momentum"], expected, places=8)
        expected_20 = (frame["close"].iloc[-1] / frame["close"].iloc[-21] - 1) * 100
        self.assertAlmostEqual(factors["momentum_20"], expected_20, places=8)

    def test_compute_all_no_longer_emits_turnover_volume_alias(self):
        factors = compute_all(price_frame(40), StrategyConfig)
        self.assertNotIn("turnover", factors)
        self.assertIn("avg_volume_5d", factors)
        self.assertAlmostEqual(factors["avg_volume_5d"], 1.0)

    def test_wilder_adx_is_bounded_and_trend_sensitive(self):
        periods = 60
        dates = np.arange(periods, dtype=float)
        trending_close = 10.0 + dates * 0.1
        flat_close = np.full(periods, 10.0)
        high = close = trending_close
        low = trending_close - 0.05
        trending_adx = wilder_adx(high, low, close)
        flat_adx = wilder_adx(flat_close + 0.02, flat_close - 0.02, flat_close)
        self.assertIsNotNone(trending_adx)
        self.assertIsNotNone(flat_adx)
        for value in (trending_adx, flat_adx):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)
        self.assertGreater(trending_adx, flat_adx)

    def test_wilder_adx_insufficient_history_returns_none(self):
        self.assertIsNone(wilder_adx(np.full(10, 11.0), np.full(10, 9.0), np.full(10, 10.0)))


class ClosingSaleTests(unittest.TestCase):
    def test_limit_down_close_cannot_be_sold(self):
        row = {
            "date": pd.Timestamp("2025-01-02"),
            "open": 9.0,
            "high": 9.0,
            "low": 9.0,
            "close": 9.0,
            "preclose": 10.0,
            "volume": 1000,
            "tradestatus": 1,
            "isST": 0,
        }
        executable, reason = closing_sale_is_executable("sh.600000", row)
        self.assertFalse(executable)
        self.assertEqual(reason, "limit_down_at_close")

    def test_suspended_close_cannot_be_sold(self):
        row = {"close": 10.0, "volume": 0, "tradestatus": 0}
        executable, reason = closing_sale_is_executable("sh.600000", row)
        self.assertFalse(executable)
        self.assertEqual(reason, "suspended")

    def test_normal_close_can_be_sold(self):
        row = {
            "date": pd.Timestamp("2025-01-02"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "preclose": 10.0,
            "volume": 1000,
            "tradestatus": 1,
            "isST": 0,
        }
        self.assertTrue(closing_sale_is_executable("sh.600000", row)[0])

    def test_close_all_records_unsellable_positions(self):
        broker = Broker(FactorStrategyConfig())
        portfolio = Portfolio(100_000, broker, 5, 0.2)
        self.assertTrue(portfolio.buy("sh.600000", 10.0, "2025-01-02", 100_000))
        blocked_row = {
            "close": 9.0,
            "volume": 1000,
            "tradestatus": 1,
            "preclose": 10.0,
            "high": 9.0,
            "low": 9.0,
            "date": pd.Timestamp("2025-01-03"),
            "isST": 0,
        }
        portfolio.close_all("2025-01-03", {"sh.600000": 9.0}, rows={"sh.600000": blocked_row})
        portfolio.record("2025-01-03", {"sh.600000": 9.0})
        self.assertEqual(portfolio.final_unsold.get("sh.600000"), "limit_down_at_close")
        self.assertEqual(len(portfolio.holdings), 1)
        # 未卖出的持仓按最后已知价（跌停收盘价）估值，而不是被静默清零或假设卖出。
        expected = portfolio.cash + portfolio.holdings["sh.600000"]["shares"] * 9.0
        self.assertAlmostEqual(portfolio.total_value({}), expected, delta=1.0)
        self.assertGreater(portfolio.equity_curve[-1]["total"], 0)


class PortfolioValuationTests(unittest.TestCase):
    def test_missing_price_uses_last_known_price_and_is_disclosed(self):
        portfolio = Portfolio(100_000, Broker(FactorStrategyConfig()), 5, 0.2)
        self.assertTrue(portfolio.buy("sh.600000", 10.0, "2025-01-02", 100_000))
        shares = portfolio.holdings["sh.600000"]["shares"]
        portfolio.record("2025-01-02", {"sh.600000": 10.0})
        portfolio.record("2025-01-03", {})  # 当日停牌缺价
        total = portfolio.equity_curve[-1]["total"]
        expected = portfolio.cash + shares * 10.0
        self.assertAlmostEqual(total, expected, delta=1.0)
        self.assertEqual(portfolio.equity_curve[-1]["unpriced_positions"], 0)
        self.assertIn("sh.600000", portfolio.last_known_prices)

    def test_never_priced_holding_falls_back_to_cost_and_is_flagged(self):
        portfolio = Portfolio(100_000, Broker(FactorStrategyConfig()), 5, 0.2)
        self.assertTrue(portfolio.buy("sh.600000", 10.0, "2025-01-02", 100_000))
        portfolio.record("2025-01-02", {})
        self.assertEqual(portfolio.unpriced_positions.get("sh.600000"), "no_price_using_avg_cost")
        self.assertGreater(portfolio.equity_curve[-1]["total"], 0)


class MetricsReliabilityTests(unittest.TestCase):
    def _metrics_with_records(self, totals):
        portfolio = Portfolio(100_000, Broker(FactorStrategyConfig()), 5, 0.2)
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        for date, total in zip(dates, totals):
            portfolio.record(date.strftime("%Y-%m-%d"), {})
            portfolio.equity_curve[-1]["total"] = total
        return calculate_metrics(portfolio)

    def test_short_sample_flags_sharpe_as_unreliable(self):
        metrics = self._metrics_with_records([100_000, 101_000, 99_000])
        self.assertFalse(metrics["sharpe_reliable"])
        self.assertEqual(metrics["sharpe_observations"], 2)
        self.assertTrue(any("Sharpe" in warning for warning in metrics["reliability_warnings"]))

    def test_sufficient_sample_marks_sharpe_reliable(self):
        totals = [100_000 * (1 + 0.001 * ((index % 5) - 2)) for index in range(40)]
        metrics = self._metrics_with_records(totals)
        self.assertTrue(metrics["sharpe_reliable"])
        self.assertEqual(metrics["sharpe_observations"], 39)
        self.assertFalse(any("Sharpe" in warning for warning in metrics["reliability_warnings"]))


class FactorBacktestExecutionGateTests(unittest.TestCase):
    """因子研究必须与主回测使用同一套停牌/涨跌停执行规则。"""

    def _run(self, stock_frame):
        config = FactorStrategyConfig(
            initial_capital=100_000,
            use_market_timing=False,
            minimum_score=50.0,
            minimum_holding_days=30,
            exit_rank=1,
            max_positions=1,
            max_position_pct=0.5,
            stop_loss_pct=None,
            take_profit_pct=None,
            factor_weights={
                "momentum_rank": 1.0,
                "trend_quality_rank": 0.0,
                "value_rank": 0.0,
                "low_volatility_rank": 0.0,
                "volume_confirmation_rank": 0.0,
                "short_reversal_rank": 0.0,
                "alpha158_pv_rank": 0.0,
                "alpha158_ridge_rank": 0.0,
            },
        )
        benchmark = stock_frame[["date", "close", "open"]].copy()
        data = {"sh.000300": benchmark, "sh.600000": stock_frame}
        # 覆盖所有调仓信号日的截面：唯一候选，评分必然入选。
        trade_dates = sorted(pd.to_datetime(stock_frame["date"].unique()))
        store = {}
        for date in trade_dates[1:]:
            store[pd.Timestamp(date).normalize()] = pd.DataFrame(
                [
                    {
                        "code": "sh.600000",
                        "score": 100.0,
                        "amount_20": 10_000_000.0,
                        "momentum_rank": 0.9,
                        "trend_quality_rank": 0.9,
                        "value_rank": 0.9,
                        "low_volatility_rank": 0.9,
                        "volume_confirmation_rank": 0.9,
                        "short_reversal_rank": 0.9,
                        "alpha158_pv_rank": 0.9,
                        "alpha158_ridge_rank": 0.9,
                    }
                ]
            )
        portfolio, metrics = run_factor_backtest(
            data,
            trade_dates[0],
            trade_dates[-1],
            config=config,
            feature_store=store,
            verbose=False,
        )
        return portfolio, metrics

    def test_suspended_execution_date_blocks_buy(self):
        frame = price_frame(70)
        # 首个调仓执行日(2025-01-06)停牌：即便评分最高也不允许按开盘买入。
        frame.loc[frame.index[3], ["volume", "tradestatus"]] = [0, 0]
        portfolio, metrics = self._run(frame)
        self.assertGreaterEqual(portfolio.execution_audit["buy_blocked"], 1)
        self.assertEqual(metrics["unsold_final_positions"], 0)

    def test_limit_up_open_blocks_buy(self):
        frame = price_frame(70)
        # 首个调仓执行日开盘一字涨停：不允许按涨停价买入。
        frame.loc[frame.index[3], ["open", "high", "low", "close", "preclose", "pctChg"]] = [
            11.0,
            11.0,
            11.0,
            11.0,
            10.0,
            10.0,
        ]
        portfolio, _ = self._run(frame)
        self.assertGreaterEqual(portfolio.execution_audit["buy_blocked"], 1)

    def test_final_limit_down_close_is_not_sold(self):
        frame = price_frame(70)
        last = frame.index[-1]
        # 末日收盘跌停：期末清仓不能假设可以卖出。
        frame.loc[last, ["open", "high", "low", "close", "preclose", "pctChg"]] = [
            9.0,
            9.0,
            9.0,
            9.0,
            10.0,
            -10.0,
        ]
        portfolio, metrics = self._run(frame)
        buy_trades = [trade for trade in portfolio.trades if trade["action"] == "BUY"]
        self.assertGreaterEqual(len(buy_trades), 1)
        self.assertEqual(metrics["unsold_final_positions"], 1)
        self.assertEqual(portfolio.final_unsold.get("sh.600000"), "limit_down_at_close")


if __name__ == "__main__":
    unittest.main()
