import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from scripts.update_market_data import (
    _batches,
    _group_pending_jobs,
    _ordered_codes,
    main,
)


def market_frame(code, start="2025-01-02", periods=7):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "preclose": 10.0,
            "volume": 1_000,
            "amount": 10_000,
            "pctChg": 0.0,
            "turn": 1.0,
            "pe": 20.0,
            "pb": 2.0,
            "tradestatus": 1,
            "isST": 0,
        }
    )


def master_frame():
    return pd.DataFrame(
        [
            ["sh.600000", "A", "1999-01-01", "", "1", "1"],
            ["sz.000001", "B", "1991-01-01", "", "1", "1"],
        ],
        columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
    )


class MarketDataUpdateTests(unittest.TestCase):
    def test_batches_cover_codes_without_duplicates(self):
        codes = [f"sh.{number:06d}" for number in range(205)]
        batches = list(_batches(codes, 100))
        self.assertEqual([len(batch) for batch in batches], [100, 100, 5])
        self.assertEqual([code for batch in batches for code in batch], codes)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            list(_batches(["sh.600000"], 0))

    def test_full_downloads_with_different_delist_dates_share_one_group(self):
        jobs = [
            ("sh.600000", pd.Timestamp("2024-05-29"), pd.Timestamp("2024-06-05"), "full"),
            ("sz.000001", pd.Timestamp("2024-05-29"), pd.Timestamp("2026-08-14"), "full"),
        ]
        groups = _group_pending_jobs(jobs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], pd.Timestamp("2024-05-29"))
        self.assertEqual(groups[0][1], pd.Timestamp("2026-08-14"))
        self.assertEqual(groups[0][3], ["sh.600000", "sz.000001"])

    def test_similar_internal_gaps_share_one_repair_group(self):
        jobs = [
            ("sh.600000", pd.Timestamp("2022-11-24"), pd.Timestamp("2024-05-28"), "gap"),
            ("sz.000001", pd.Timestamp("2022-11-25"), pd.Timestamp("2024-05-27"), "gap"),
        ]
        groups = _group_pending_jobs(jobs)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], pd.Timestamp("2022-11-24"))
        self.assertEqual(groups[0][1], pd.Timestamp("2024-05-28"))
        self.assertEqual(groups[0][2], "gap")
        self.assertEqual(groups[0][3], ["sh.600000", "sz.000001"])

    def test_priority_codes_are_downloaded_first_without_changing_universe(self):
        codes = ["sh.600000", "sz.000001", "sh.600519", "sz.000002"]
        ordered = _ordered_codes(codes, {"sh.600519", "sz.000001"})
        self.assertEqual(ordered[:2], ["sh.600519", "sz.000001"])
        self.assertEqual(set(ordered), set(codes))

    def test_historical_download_writes_resumable_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = os.path.join(temp_dir, "market.pkl")

            def fake_batch(codes, *_args, **_kwargs):
                return {code: market_frame(code) for code in codes}

            argv = [
                "update_market_data.py",
                "2025-01-01",
                "2025-01-10",
                "--cache",
                cache,
                "--batch-size",
                "1",
            ]
            provider = Mock()
            provider.name = "baostock"
            provider.warnings = []
            provider.supports_security_master = True
            provider.security_master.return_value = master_frame()
            provider.batch.side_effect = fake_batch
            provider.index.return_value = market_frame("sh.000300")
            with (
                patch.object(sys, "argv", argv + ["--provider", "baostock"]),
                patch("scripts.update_market_data.create_provider", return_value=provider),
            ):
                main()

            with open(f"{cache}.metadata.json", "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["download"]["status"], "complete")
            self.assertEqual(metadata["download"]["completed_codes"], 2)
            self.assertEqual(metadata["universe"]["requested_count"], 2)
            self.assertTrue(metadata["quality"]["formal_ready"])
            self.assertTrue(os.path.isfile(f"{cache}.security_master.csv"))
            self.assertTrue(os.path.isfile(f"{cache}.quality.json"))
            self.assertTrue(os.path.isfile(f"{cache}.quality.csv"))

    def test_akshare_large_universe_requires_explicit_confirmation(self):
        provider = Mock()
        provider.name = "akshare-tencent"
        provider.warnings = ["research-only"]
        provider.supports_security_master = False
        codes = [f"sh.{number:06d}" for number in range(600000, 600101)]
        with tempfile.TemporaryDirectory() as temp_dir:
            universe = os.path.join(temp_dir, "codes.txt")
            cache = os.path.join(temp_dir, "market.pkl")
            with open(universe, "w", encoding="utf-8") as handle:
                handle.write("\n".join(codes))
            argv = [
                "update_market_data.py",
                "2025-01-01",
                "2025-01-10",
                "--cache",
                cache,
                "--provider",
                "akshare-tencent",
                "--universe",
                "file",
                "--universe-file",
                universe,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("scripts.update_market_data.create_provider", return_value=provider),
                self.assertRaises(SystemExit) as raised,
            ):
                main()
            self.assertIn("可能运行数十到数百小时", str(raised.exception))

    def test_existing_cache_rejects_a_different_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = os.path.join(temp_dir, "market.pkl")
            data = {
                "sh.600000": market_frame("sh.600000"),
                "sh.000300": market_frame("sh.000300"),
            }
            from data.cache_store import build_metadata, save_market_cache

            metadata = build_metadata(
                data,
                benchmark="sh.000300",
                adjust="2",
                universe_mode="file",
                source="baostock",
            )
            metadata["universe"]["requested_codes"] = ["sh.600000"]
            save_market_cache(cache, data, metadata)
            provider = Mock()
            provider.name = "akshare-tencent"
            provider.warnings = []
            provider.supports_security_master = False
            argv = [
                "update_market_data.py",
                "2025-01-01",
                "2025-01-10",
                "--cache",
                cache,
                "--provider",
                "akshare-tencent",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch("scripts.update_market_data.create_provider", return_value=provider),
                self.assertRaises(SystemExit) as raised,
            ):
                main()
            self.assertIn("拒绝混合不同数据口径", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
