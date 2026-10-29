import unittest

import pandas as pd

from data.quality import (
    audit_market_data,
    completion_map,
    expected_period,
    pending_downloads,
)


def frame(start, periods):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "volume": 1000,
            "amount": 10000,
        }
    )


def master():
    return pd.DataFrame(
        [
            ["sh.600000", "A", "2025-01-06", "", "1", "1"],
        ],
        columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
    )


class DataQualityTests(unittest.TestCase):
    def test_expected_period_is_clipped_to_ipo(self):
        period = expected_period("sh.600000", "2025-01-01", "2025-01-31", master())
        self.assertEqual(period.start, pd.Timestamp("2025-01-06"))

    def test_pending_downloads_are_calculated_per_stock(self):
        data = {
            "sh.600000": frame("2025-01-01", 8),
            "sz.000001": frame("2025-01-01", 3),
        }
        jobs = pending_downloads(
            data,
            ["sh.600000", "sz.000001"],
            "2025-01-01",
            "2025-01-31",
            grace_days=2,
        )
        by_code = {code: (start, end, reason) for code, start, end, reason in jobs}
        self.assertEqual(by_code["sh.600000"][0], pd.Timestamp("2025-01-11"))
        self.assertEqual(by_code["sz.000001"][0], pd.Timestamp("2025-01-04"))
        self.assertEqual(by_code["sz.000001"][2], "tail")

    def test_completion_rejects_a_nonempty_but_truncated_stock(self):
        complete = completion_map(
            {"sh.600000": frame("2025-01-01", 3)},
            ["sh.600000"],
            "2025-01-01",
            "2025-01-31",
            grace_days=2,
        )
        self.assertFalse(complete["sh.600000"])

    def test_pending_downloads_detects_long_internal_gap(self):
        benchmark = frame("2025-01-01", 10)
        stock = benchmark.drop(index=[3, 4, 5, 6]).reset_index(drop=True)
        jobs = pending_downloads(
            {"sh.600000": stock},
            ["sh.600000"],
            "2025-01-01",
            "2025-01-14",
            benchmark_dates=benchmark["date"],
            internal_gap_sessions=3,
        )

        self.assertEqual(
            jobs,
            [
                (
                    "sh.600000",
                    pd.Timestamp("2025-01-06"),
                    pd.Timestamp("2025-01-09"),
                    "gap",
                )
            ],
        )

    def test_pending_downloads_ignores_short_internal_gap(self):
        benchmark = frame("2025-01-01", 10)
        stock = benchmark.drop(index=[3, 4]).reset_index(drop=True)
        jobs = pending_downloads(
            {"sh.600000": stock},
            ["sh.600000"],
            "2025-01-01",
            "2025-01-14",
            benchmark_dates=benchmark["date"],
            internal_gap_sessions=3,
        )

        self.assertEqual(jobs, [])

    def test_audit_detects_internal_session_gap_and_bad_ohlc(self):
        benchmark = frame("2025-01-01", 10)[["date", "close"]]
        stock = frame("2025-01-01", 10).drop(index=[3, 4, 5]).reset_index(drop=True)
        stock.loc[0, "high"] = 9.0
        report = audit_market_data(
            {"sh.000300": benchmark, "sh.600000": stock},
            benchmark="sh.000300",
            start="2025-01-01",
            end="2025-01-14",
            requested_codes=["sh.600000"],
            minimum_coverage=0.90,
            grace_days=2,
        )
        row = report["stocks"][0]
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["missing_sessions"], 3)
        self.assertEqual(row["longest_missing_run"], 3)
        self.assertEqual(row["invalid_ohlc_rows"], 1)
        self.assertFalse(report["summary"]["formal_ready"])

    def test_audit_accepts_unpriced_non_trading_marker(self):
        benchmark = frame("2025-01-01", 3)[["date", "close"]]
        stock = frame("2025-01-01", 3)
        stock["tradestatus"] = 1
        stock.loc[2, ["open", "high", "low", "close"]] = float("nan")
        stock.loc[2, "tradestatus"] = 0

        report = audit_market_data(
            {"sh.000300": benchmark, "sh.600000": stock},
            benchmark="sh.000300",
            start="2025-01-01",
            end="2025-01-03",
            requested_codes=["sh.600000"],
            minimum_coverage=1.0,
            grace_days=0,
        )

        row = report["stocks"][0]
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["invalid_ohlc_rows"], 0)
        self.assertEqual(row["unpriced_non_trading_rows"], 1)
        self.assertTrue(report["summary"]["formal_ready"])

    def test_audit_rejects_partial_prices_on_non_trading_marker(self):
        benchmark = frame("2025-01-01", 3)[["date", "close"]]
        stock = frame("2025-01-01", 3)
        stock["tradestatus"] = 1
        stock.loc[2, "open"] = float("nan")
        stock.loc[2, "tradestatus"] = 0

        report = audit_market_data(
            {"sh.000300": benchmark, "sh.600000": stock},
            benchmark="sh.000300",
            start="2025-01-01",
            end="2025-01-03",
            requested_codes=["sh.600000"],
            minimum_coverage=1.0,
            grace_days=0,
        )

        row = report["stocks"][0]
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["invalid_ohlc_rows"], 1)
        self.assertEqual(row["unpriced_non_trading_rows"], 0)


if __name__ == "__main__":
    unittest.main()
