import unittest
from unittest.mock import patch

import pandas as pd

from paper_trade.planner import build_order_plan


class Config:
    benchmark_index = "sh.000300"
    initial_capital = 100_000
    max_positions = 8
    max_position_pct = 0.12
    take_profit_pct = 0.10
    stop_loss_pct = 0.05
    rebalance_freq = "weekly"


def prices(close=10.0):
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=3),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000,
            "amount": 10_000,
        }
    )


class PaperTradeTests(unittest.TestCase):
    def test_order_plan_reuses_shared_candidate_selector(self):
        data = {"sh.600000": prices(), "sh.000300": prices(4_000)}
        account = {"cash": 100_000, "holdings": {}}
        with (
            patch("paper_trade.planner.market_allows_buy", return_value=(True, "ok")),
            patch(
                "paper_trade.planner.select_candidates", return_value=[("sh.600000", 6.5)]
            ) as selector,
        ):
            orders, timing = build_order_plan(data, "2025-01-03", "2025-01-06", account, Config)
        selector.assert_called_once()
        self.assertTrue(timing["can_buy"])
        self.assertEqual(orders[0]["action"], "BUY")
        self.assertEqual(orders[0]["estimated_shares"], 1_200)

    def test_stop_loss_order_is_emitted_even_when_market_blocks_buys(self):
        data = {"sh.600000": prices(9.0), "sh.000300": prices(4_000)}
        account = {
            "cash": 50_000,
            "holdings": {"sh.600000": {"shares": 1_000, "avg_cost": 10.0}},
        }
        with patch("paper_trade.planner.market_allows_buy", return_value=(False, "weak_market")):
            orders, timing = build_order_plan(data, "2025-01-03", "2025-01-06", account, Config)
        self.assertFalse(timing["can_buy"])
        self.assertEqual(orders[0]["action"], "SELL")
        self.assertEqual(orders[0]["reason"], "stop_loss")

    def test_non_rebalance_session_does_not_emit_buys(self):
        data = {"sh.600000": prices(), "sh.000300": prices(4_000)}
        account = {"cash": 100_000, "holdings": {}}
        with (
            patch("paper_trade.planner.market_allows_buy", return_value=(True, "ok")),
            patch("paper_trade.planner.select_candidates", return_value=[("sh.600000", 6.5)]),
        ):
            orders, timing = build_order_plan(data, "2025-01-06", "2025-01-07", account, Config)
        self.assertEqual(orders, [])
        self.assertEqual(timing["reason"], "not_rebalance_session")


if __name__ == "__main__":
    unittest.main()
