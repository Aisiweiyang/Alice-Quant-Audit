import unittest

import pandas as pd

from engine.execution import order_is_executable, price_limit_pct


class ExecutionRuleTests(unittest.TestCase):
    def test_board_and_st_price_limits(self):
        self.assertEqual(price_limit_pct("sh.600000", "2025-01-01"), 10.0)
        self.assertEqual(price_limit_pct("sh.600000", "2025-01-01", 1), 5.0)
        self.assertEqual(price_limit_pct("sz.300001", "2021-01-01"), 20.0)
        self.assertEqual(price_limit_pct("sz.300001", "2019-01-01"), 10.0)

    def test_suspended_order_is_blocked(self):
        row = {"date": pd.Timestamp("2025-01-02"), "open": 10, "volume": 0, "tradestatus": 0}
        self.assertEqual(order_is_executable("sh.600000", row, "BUY"), (False, "suspended"))

    def test_limit_up_blocks_buy_but_not_sell(self):
        row = {
            "date": pd.Timestamp("2025-01-02"),
            "open": 11.0,
            "high": 11.0,
            "low": 11.0,
            "preclose": 10.0,
            "volume": 1000,
            "tradestatus": 1,
            "isST": 0,
        }
        self.assertEqual(order_is_executable("sh.600000", row, "BUY")[1], "limit_up_at_open")
        self.assertTrue(order_is_executable("sh.600000", row, "SELL")[0])

    def test_limit_down_blocks_sell_but_not_buy(self):
        row = {
            "date": pd.Timestamp("2025-01-02"),
            "open": 9.0,
            "high": 9.0,
            "low": 9.0,
            "preclose": 10.0,
            "volume": 1000,
            "tradestatus": 1,
            "isST": 0,
        }
        self.assertEqual(order_is_executable("sh.600000", row, "SELL")[1], "limit_down_at_open")
        self.assertTrue(order_is_executable("sh.600000", row, "BUY")[0])


if __name__ == "__main__":
    unittest.main()
