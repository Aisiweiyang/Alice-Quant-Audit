import json
import os
import tempfile
import unittest
from datetime import datetime

from integration.candidate_collection import (
    collect_candidate_directory,
    load_rejection_audit,
    load_trading_dates,
)


def candidate(code="600000", name="A", rank=1, score=86.8):
    return {
        "rank": rank,
        "code": code,
        "name": name,
        "score": score,
        "risk": "中",
        "components": {"趋势": score},
        "reasons": ["测试候选"],
    }


def payload(trade_date, candidates=None):
    return {
        "schema_version": 1,
        "source_system": "a-share-mainboard-research",
        "strategy_version": "mainboard-factor-score-v1",
        "market": "CN_A_SHARE",
        "trade_date": trade_date,
        "generated_at": f"{trade_date}T18:30:00+08:00",
        "quality": {
            "passed": True,
            "universe_coverage": 1.0,
            "history_coverage": 0.99,
        },
        "candidates": candidates or [candidate()],
    }


def write_json(directory, filename, value):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    return path


class CandidateCollectionTests(unittest.TestCase):
    def collect(self, source, pool, audit, **kwargs):
        return collect_candidate_directory(
            source,
            pool_path=pool,
            audit_path=audit,
            collected_at=datetime.fromisoformat("2026-08-30T20:00:00+08:00"),
            **kwargs,
        )

    def test_valid_directory_can_be_rerun_without_duplicate_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            os.makedirs(source)
            write_json(source, "candidates_20260828.json", payload("2026-08-28"))
            write_json(source, "candidates_20260829.json", payload("2026-08-29"))
            pool = os.path.join(directory, "pool.json")
            audit = os.path.join(directory, "audit.json")
            archive = os.path.join(directory, "archive")
            first = self.collect(source, pool, audit, archive_dir=archive)
            second = self.collect(source, pool, audit, archive_dir=archive)
            archived_files = os.listdir(archive)

        self.assertEqual(first.scanned, 2)
        self.assertEqual(first.imported, 2)
        self.assertEqual(second.existing, 2)
        self.assertEqual(second.pool_summary["import_count"], 2)
        self.assertEqual(first.archived_new, 2)
        self.assertEqual(second.archived_existing, 2)
        self.assertEqual(len(archived_files), 2)

    def test_rejection_is_audited_once_and_changed_file_is_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            os.makedirs(source)
            bad_path = os.path.join(source, "candidates_bad.json")
            with open(bad_path, "w", encoding="utf-8") as handle:
                handle.write("not-json")
            pool = os.path.join(directory, "pool.json")
            audit = os.path.join(directory, "audit.json")
            first = self.collect(source, pool, audit)
            repeated = self.collect(source, pool, audit)
            write_json(source, "candidates_bad.json", payload("2026-08-28"))
            changed = self.collect(source, pool, audit)
            audit_data = load_rejection_audit(audit)

        self.assertEqual(first.rejected, 1)
        self.assertEqual(repeated.known_rejected, 1)
        self.assertEqual(repeated.rejected, 0)
        self.assertEqual(changed.imported, 1)
        self.assertEqual(len(audit_data["rejections"]), 1)

    def test_conflicting_same_day_file_is_rejected_but_first_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            os.makedirs(source)
            write_json(source, "candidates_1.json", payload("2026-08-28"))
            write_json(
                source,
                "candidates_2.json",
                payload("2026-08-28", [candidate(score=90.0)]),
            )
            pool = os.path.join(directory, "pool.json")
            audit = os.path.join(directory, "audit.json")
            result = self.collect(source, pool, audit)
            audit_data = load_rejection_audit(audit)

        self.assertEqual(result.imported, 1)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(result.pool_summary["import_count"], 1)
        self.assertIn("人工复核", audit_data["rejections"][0]["reason"])

    def test_explicit_calendar_finds_missing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            os.makedirs(source)
            write_json(source, "candidates_1.json", payload("2026-08-28"))
            write_json(source, "candidates_3.json", payload("2026-09-01"))
            pool = os.path.join(directory, "pool.json")
            audit = os.path.join(directory, "audit.json")
            result = self.collect(
                source,
                pool,
                audit,
                trading_dates=("2026-08-28", "2026-08-31", "2026-09-01"),
            )

        self.assertEqual(result.missing_trade_dates, ("2026-08-31",))

    def test_calendar_loader_and_range_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            calendar = write_json(
                directory,
                "calendar.json",
                {"trade_dates": ["2026-08-31", "2026-08-28", "2026-08-31"]},
            )
            self.assertEqual(load_trading_dates(calendar), ("2026-08-28", "2026-08-31"))
            source = os.path.join(directory, "source")
            os.makedirs(source)
            with self.assertRaisesRegex(ValueError, "显式交易日历"):
                self.collect(
                    source,
                    os.path.join(directory, "pool.json"),
                    os.path.join(directory, "audit.json"),
                    through_date="2026-08-31",
                )
            with self.assertRaisesRegex(ValueError, "文件名模式"):
                self.collect(
                    source,
                    os.path.join(directory, "pool.json"),
                    os.path.join(directory, "audit.json"),
                    pattern="..\\*.json",
                )

    def test_invalid_pool_stops_batch_without_blame_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            os.makedirs(source)
            write_json(source, "candidates_1.json", payload("2026-08-28"))
            pool = write_json(directory, "pool.json", {"schema_version": 999})
            audit = os.path.join(directory, "audit.json")
            with self.assertRaisesRegex(ValueError, "观察池版本"):
                self.collect(source, pool, audit)
            self.assertFalse(os.path.exists(audit))
            with self.assertRaisesRegex(ValueError, "同一个文件"):
                self.collect(source, audit, audit)


if __name__ == "__main__":
    unittest.main()
