import os
import tempfile
import unittest

import pandas as pd

from data.cache_store import build_metadata, save_market_cache
from scripts.doctor import inspect_project


def frame():
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=3),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "volume": 1000,
            "amount": 10000,
            "preclose": 10.0,
            "tradestatus": 1,
            "isST": 0,
            "pe": 20.0,
            "pb": 2.0,
        }
    )


class DoctorTests(unittest.TestCase):
    def test_no_cache_is_waiting_not_failure(self):
        statuses = {item["name"]: item["status"] for item in inspect_project()}
        self.assertEqual(statuses["market_cache"], "waiting")

    def test_complete_cache_passes_historical_status_check(self):
        data = {"sh.600000": frame(), "sh.000300": frame()[["date", "close"]]}
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
            statuses = {item["name"]: item["status"] for item in inspect_project(path)}
        self.assertEqual(statuses["historical_trade_status"], "ok")
        self.assertEqual(statuses["historical_valuation_fields"], "ok")


if __name__ == "__main__":
    unittest.main()
