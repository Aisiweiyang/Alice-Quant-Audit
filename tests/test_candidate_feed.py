import json
import os
import tempfile
import unittest

from integration.candidate_feed import load_candidate_feed, normalise_candidate_code


def payload():
    return {
        "schema_version": 1,
        "source_system": "a-share-mainboard-research",
        "strategy_version": "candidate-score-1",
        "market": "CN_A_SHARE",
        "trade_date": "2026-08-28",
        "generated_at": "2026-08-28T18:30:00+08:00",
        "quality": {
            "passed": True,
            "universe_coverage": 1.0,
            "history_coverage": 0.99,
        },
        "candidates": [
            {
                "rank": 1,
                "code": "600000",
                "name": "A",
                "score": 86.8,
                "risk": "中",
                "components": {"趋势": 95},
                "reasons": ["趋势"],
                "price": 10.5,
                "change_pct": 1.2,
                "amount": 200000000,
                "pe": 12.3,
                "pb": 1.4,
            },
            {
                "rank": 2,
                "code": "SZ.000001",
                "name": "B",
                "score": 84.2,
                "components": {},
                "reasons": [],
            },
        ],
    }


def write_payload(directory, value):
    path = os.path.join(directory, "candidates.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    return path


class CandidateFeedTests(unittest.TestCase):
    def test_valid_feed_is_normalised_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, payload())
            feed = load_candidate_feed(path, expected_trade_date="2026-08-28")
        self.assertEqual([item.code for item in feed.candidates], ["sh.600000", "sz.000001"])
        self.assertEqual(len(feed.source_sha256), 64)
        self.assertEqual(feed.summary()["candidate_count"], 2)
        self.assertEqual(feed.candidates[0].attributes["price"], 10.5)
        self.assertEqual(feed.candidates[0].attributes["amount"], 200000000)

    def test_quality_must_pass_and_meet_coverage_thresholds(self):
        cases = [
            {"passed": False, "universe_coverage": 1.0, "history_coverage": 1.0},
            {"passed": True, "universe_coverage": 0.79, "history_coverage": 1.0},
            {"passed": True, "universe_coverage": 1.0, "history_coverage": 0.79},
        ]
        for quality in cases:
            with self.subTest(quality=quality), tempfile.TemporaryDirectory() as directory:
                value = payload()
                value["quality"] = quality
                path = write_payload(directory, value)
                with self.assertRaises(ValueError):
                    load_candidate_feed(path)
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, payload())
            with self.assertRaises(ValueError):
                load_candidate_feed(path, minimum_universe_coverage=-0.01)
            with self.assertRaises(ValueError):
                load_candidate_feed(path, minimum_history_coverage=1.01)

    def test_duplicate_candidates_are_rejected_after_normalisation(self):
        with tempfile.TemporaryDirectory() as directory:
            value = payload()
            value["candidates"][1]["code"] = "sh.600000"
            path = write_payload(directory, value)
            with self.assertRaisesRegex(ValueError, "重复"):
                load_candidate_feed(path)

    def test_non_main_board_codes_are_rejected(self):
        for code in ("300001", "688001", "bj.430001"):
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    normalise_candidate_code(code)

    def test_expected_trade_date_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, payload())
            with self.assertRaisesRegex(ValueError, "预期日期"):
                load_candidate_feed(path, expected_trade_date="2026-08-27")
            value = payload()
            value["trade_date"] = "20260828"
            path = write_payload(directory, value)
            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                load_candidate_feed(path)

    def test_generated_at_requires_timezone(self):
        with tempfile.TemporaryDirectory() as directory:
            value = payload()
            value["generated_at"] = "2026-08-28T18:30:00"
            path = write_payload(directory, value)
            with self.assertRaisesRegex(ValueError, "时区"):
                load_candidate_feed(path)

    def test_optional_market_attributes_are_range_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            value = payload()
            value["candidates"][0]["change_pct"] = float("inf")
            path = write_payload(directory, value)
            with self.assertRaisesRegex(ValueError, "candidate.change_pct"):
                load_candidate_feed(path)


if __name__ == "__main__":
    unittest.main()
