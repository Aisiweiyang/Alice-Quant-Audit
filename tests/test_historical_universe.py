import unittest

import pandas as pd

from data.cleaner import filter_tradable
from data.universe import active_on, historical_main_board_codes, is_main_board_code


def security_master():
    return pd.DataFrame(
        [
            ["sh.600000", "active", "1999-01-01", "", "1", "1"],
            ["sz.000001", "delisted", "1991-01-01", "2018-06-01", "1", "0"],
            ["sh.600001", "future", "2027-01-01", "", "1", "1"],
            ["sh.688001", "star", "2019-07-22", "", "1", "1"],
            ["sh.000300", "index", "2005-04-08", "", "2", "1"],
        ],
        columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
    )


def price_frame(is_st=0, trade_status=1):
    dates = pd.bdate_range("2025-01-01", periods=65)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1_000,
            "amount": 10_000,
            "tradestatus": trade_status,
            "isST": is_st,
        }
    )


class HistoricalUniverseTests(unittest.TestCase):
    def test_period_universe_includes_delisted_stock(self):
        codes = historical_main_board_codes(security_master(), "2015-01-01", "2026-12-31")
        self.assertEqual(codes, ["sh.600000", "sz.000001"])

    def test_main_board_filter_excludes_star_and_index(self):
        self.assertTrue(is_main_board_code("sh.600000"))
        self.assertTrue(is_main_board_code("sz.002001"))
        self.assertFalse(is_main_board_code("sh.688001"))
        self.assertFalse(is_main_board_code("sh.000300"))

    def test_active_on_uses_ipo_and_out_dates(self):
        master = security_master()
        self.assertTrue(active_on(master, "sz.000001", "2017-01-01"))
        self.assertFalse(active_on(master, "sz.000001", "2019-01-01"))

    def test_historical_st_and_suspension_are_rejected(self):
        st = price_frame(is_st=1)
        suspended = price_frame(trade_status=0)
        date = st.iloc[-1]["date"]
        self.assertEqual(filter_tradable(st, date)[1], "历史ST状态")
        self.assertEqual(filter_tradable(suspended, date)[1], "停牌中")

    def test_missing_row_on_signal_date_is_rejected(self):
        frame = price_frame()
        after_last_row = frame.iloc[-1]["date"] + pd.Timedelta(days=1)
        self.assertFalse(filter_tradable(frame, after_last_row)[0])


if __name__ == "__main__":
    unittest.main()
