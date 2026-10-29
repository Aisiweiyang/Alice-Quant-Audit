import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import integration.observation_pool as observation_pool
from integration.candidate_feed import load_candidate_feed
from integration.observation_pool import (
    CandidateFeedConflictError,
    load_observation_pool,
    record_candidate_feed,
)


def candidate(code, name, rank, score):
    return {
        "rank": rank,
        "code": code,
        "name": name,
        "score": score,
        "risk": "中",
        "components": {"趋势": score},
        "reasons": ["测试候选"],
        "price": 10.5,
        "change_pct": 1.2,
    }


def payload(trade_date="2026-08-28", generated_at=None, candidates=None):
    generated_at = generated_at or f"{trade_date}T18:30:00+08:00"
    return {
        "schema_version": 1,
        "source_system": "a-share-mainboard-research",
        "strategy_version": "mainboard-factor-score-v1",
        "market": "CN_A_SHARE",
        "trade_date": trade_date,
        "generated_at": generated_at,
        "quality": {
            "passed": True,
            "universe_coverage": 1.0,
            "history_coverage": 0.99,
        },
        "candidates": candidates
        or [
            candidate("600000", "A", 1, 86.8),
            candidate("000001", "B", 2, 84.2),
        ],
    }


def load_feed(directory, filename, value):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    return load_candidate_feed(path)


class ObservationPoolTests(unittest.TestCase):
    def test_feed_is_recorded_once_without_rewriting_exact_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            feed = load_feed(directory, "feed.json", payload())
            pool_path = os.path.join(directory, "pool.json")
            first = record_candidate_feed(
                pool_path,
                feed,
                recorded_at=datetime.fromisoformat("2026-08-30T19:00:00+08:00"),
            )
            with open(pool_path, "rb") as handle:
                original = handle.read()
            repeated = record_candidate_feed(pool_path, feed)
            with open(pool_path, "rb") as handle:
                after_repeat = handle.read()

        self.assertTrue(first.imported)
        self.assertFalse(repeated.imported)
        self.assertEqual(original, after_repeat)
        self.assertEqual(repeated.summary["import_count"], 1)
        self.assertEqual(repeated.summary["unique_candidate_count"], 2)

    def test_summary_tracks_repeat_appearances_across_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            pool_path = os.path.join(directory, "pool.json")
            day_one = load_feed(directory, "day_one.json", payload())
            day_two = load_feed(
                directory,
                "day_two.json",
                payload(
                    trade_date="2026-08-29",
                    candidates=[
                        candidate("600000", "A", 1, 90.0),
                        candidate("600001", "C", 2, 82.0),
                    ],
                ),
            )
            record_candidate_feed(pool_path, day_one)
            result = record_candidate_feed(pool_path, day_two)

        summary = result.summary
        by_code = {item["code"]: item for item in summary["candidates"]}
        self.assertEqual(summary["import_count"], 2)
        self.assertEqual(summary["unique_candidate_count"], 3)
        self.assertEqual(by_code["sh.600000"]["appearances"], 2)
        self.assertEqual(by_code["sh.600000"]["first_trade_date"], "2026-08-28")
        self.assertEqual(by_code["sh.600000"]["last_trade_date"], "2026-08-29")
        self.assertEqual(by_code["sh.600000"]["latest_score"], 90.0)
        self.assertEqual(by_code["sh.600000"]["average_score"], 88.4)
        self.assertEqual(by_code["sh.600000"]["latest_attributes"]["price"], 10.5)

    def test_same_source_strategy_and_date_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            pool_path = os.path.join(directory, "pool.json")
            original = load_feed(directory, "original.json", payload())
            changed_value = payload()
            changed_value["candidates"][0]["score"] = 87.0
            changed = load_feed(directory, "changed.json", changed_value)
            record_candidate_feed(pool_path, original)
            with open(pool_path, "rb") as handle:
                before_conflict = handle.read()
            with self.assertRaisesRegex(CandidateFeedConflictError, "人工复核"):
                record_candidate_feed(pool_path, changed)
            with open(pool_path, "rb") as handle:
                after_conflict = handle.read()
        self.assertEqual(before_conflict, after_conflict)

    def test_invalid_existing_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pool_path = os.path.join(directory, "pool.json")
            with open(pool_path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "imports": [{"source_sha256": "bad"}]}, handle)
            with self.assertRaisesRegex(ValueError, "哈希"):
                load_observation_pool(pool_path)

    def test_recorded_at_requires_timezone(self):
        with tempfile.TemporaryDirectory() as directory:
            feed = load_feed(directory, "feed.json", payload())
            pool_path = os.path.join(directory, "pool.json")
            with self.assertRaisesRegex(ValueError, "时区"):
                record_candidate_feed(
                    pool_path,
                    feed,
                    recorded_at=datetime(2026, 8, 30, 19, 0, 0),
                )
            self.assertFalse(os.path.exists(pool_path))

    def test_pool_size_limit_is_checked_before_replacing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            feed = load_feed(directory, "feed.json", payload())
            pool_path = os.path.join(directory, "pool.json")
            with patch.object(observation_pool, "MAX_POOL_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "50 MB"):
                    record_candidate_feed(pool_path, feed)
            self.assertFalse(os.path.exists(pool_path))


if __name__ == "__main__":
    unittest.main()
