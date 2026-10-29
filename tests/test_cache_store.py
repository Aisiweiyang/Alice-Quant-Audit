import json
import os
import tempfile
import unittest

import pandas as pd

from data.cache_store import (
    build_cache_key,
    build_metadata,
    has_historical_trade_status,
    load_market_cache,
    merge_market_data,
    metadata_path,
    save_market_cache,
)


def frame(start="2025-01-02", periods=3):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "volume": 1_000,
            "amount": 10_000,
        }
    )


class CacheStoreTests(unittest.TestCase):
    def test_cache_key_uses_actual_codes_not_only_count(self):
        common = dict(
            start_date="2025-01-01",
            end_date="2025-12-31",
            adjust="2",
            benchmark="sh.000300",
            universe_mode="file",
        )
        first = build_cache_key(codes=["sh.600000", "sz.000001"], **common)
        second = build_cache_key(codes=["sh.600001", "sz.000002"], **common)
        self.assertNotEqual(first, second)

    def test_round_trip_writes_metadata_sidecar(self):
        data = {"sh.600000": frame(), "sh.000300": frame()}
        metadata = build_metadata(
            data, benchmark="sh.000300", adjust="2", universe_mode="file", source="test"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "market.pkl")
            save_market_cache(path, data, metadata)
            loaded, loaded_metadata = load_market_cache(path, "sh.000300")
            self.assertEqual(set(loaded), set(data))
            self.assertTrue(os.path.isfile(metadata_path(path)))
            self.assertEqual(loaded_metadata["universe"]["stock_count"], 1)
            with open(metadata_path(path), "r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["schema_version"], 3)

    def test_incremental_merge_deduplicates_dates(self):
        existing = {"sh.600000": frame(periods=3)}
        updates = {"sh.600000": frame(start="2025-01-06", periods=3)}
        merged = merge_market_data(existing, updates)
        self.assertEqual(len(merged["sh.600000"]), 5)
        self.assertTrue(merged["sh.600000"]["date"].is_unique)

    def test_missing_valuation_is_recorded_in_capabilities(self):
        data = {"sh.600000": frame(), "sh.000300": frame()}
        metadata = build_metadata(
            data, benchmark="sh.000300", adjust="2", universe_mode="file", source="test"
        )
        self.assertFalse(metadata["capabilities"]["historical_valuation_fields"])
        self.assertEqual(metadata["capabilities"]["valuation_coverage"], 0.0)

    def test_non_trading_marker_does_not_require_preclose(self):
        stock = frame()
        stock["tradestatus"] = 1
        stock["isST"] = 0
        stock["preclose"] = 10.0
        stock.loc[2, "tradestatus"] = 0
        stock.loc[2, "preclose"] = float("nan")

        self.assertTrue(has_historical_trade_status({"sh.600000": stock}))

    def test_tradable_row_requires_preclose(self):
        stock = frame()
        stock["tradestatus"] = 1
        stock["isST"] = 0
        stock["preclose"] = 10.0
        stock.loc[2, "preclose"] = float("nan")

        self.assertFalse(has_historical_trade_status({"sh.600000": stock}))


if __name__ == "__main__":
    unittest.main()
