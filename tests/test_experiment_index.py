"""追加式实验索引（experiments_index.json）的测试。

要求：只追加不删除；原子写；损坏的历史索引不被覆盖；索引失败不影响报告本体。
"""

import json
import os
import tempfile
import unittest

from commercial.config import CommercialRequest
from commercial.experiment_index import append_experiment_index, experiment_entry


def _analysis(report_id, score=60, rating="READY", mode="full"):
    return {
        "report_id": report_id,
        "mode": mode,
        "request": CommercialRequest(
            cache="x.pkl",
            start="2025-01-01",
            end="2025-06-30",
            strategy="classic_trend",
        ),
        "score": {"score": score, "grade_label": "结论不稳定"},
        "data_audit": {"rating": rating},
        "blocked": False,
        "base": {
            "metrics": {
                "total_return_pct": 5.0,
                "max_drawdown_pct": -8.0,
                "sharpe_ratio": 1.0,
                "sharpe_ci_includes_zero": False,
                "total_trades": 12,
                "deflated_sharpe": 0.9,
                "omega_ratio": 1.2,
            }
        },
        "manifest": {"created_at": "2026-09-13T00:00:00+08:00"},
    }


class ExperimentIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.reports_root = os.path.join(self.temp.name, "commercial_reports")
        os.makedirs(self.reports_root)

    def tearDown(self):
        self.temp.cleanup()

    def _index_path(self):
        return os.path.join(self.reports_root, "experiments_index.json")

    def test_appends_without_touching_history(self):
        self.assertIsNotNone(
            append_experiment_index(
                os.path.join(self.reports_root, "rep_a"), _analysis("rep_a", score=61)
            )
        )
        self.assertIsNotNone(
            append_experiment_index(
                os.path.join(self.reports_root, "rep_b"), _analysis("rep_b", score=63)
            )
        )
        with open(self._index_path(), encoding="utf-8") as handle:
            records = json.load(handle)
        self.assertEqual([r["report_id"] for r in records], ["rep_a", "rep_b"])
        self.assertEqual(records[0]["score"], 61)

    def test_entry_fields_are_faithful(self):
        entry = experiment_entry(_analysis("rep_x"))
        self.assertEqual(entry["report_id"], "rep_x")
        self.assertEqual(entry["strategy"], "classic_trend")
        self.assertEqual(entry["start"], "2025-01-01")
        self.assertEqual(entry["data_rating"], "READY")
        self.assertEqual(entry["score"], 60)
        self.assertEqual(entry["key_metrics"]["total_return_pct"], 5.0)
        self.assertEqual(entry["key_metrics"]["omega_ratio"], 1.2)

    def test_compare_entry_records_plan_counts_and_pbo(self):
        analysis = _analysis("cmp_1", mode="compare")
        analysis["plan_results"] = [
            {"status": "ok"},
            {"status": "ok"},
            {"status": "insufficient_data"},
        ]
        analysis["overfitting_audit"] = {
            "n_trials": 2,
            "pbo": {"status": "ok", "pbo_probability": 0.4},
        }
        entry = experiment_entry(analysis)
        self.assertEqual(entry["plans_ok"], "2/3")
        self.assertEqual(entry["n_trials"], 2)
        self.assertEqual(entry["pbo_probability"], 0.4)

    def test_corrupted_index_is_never_overwritten(self):
        path = self._index_path()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertIsNone(
            append_experiment_index(os.path.join(self.reports_root, "rep_c"), _analysis("rep_c"))
        )
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "{not json")

    def test_non_list_index_is_never_overwritten(self):
        path = self._index_path()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"legacy": True}, handle)
        self.assertIsNone(
            append_experiment_index(os.path.join(self.reports_root, "rep_d"), _analysis("rep_d"))
        )
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"legacy": True})

    def test_isolated_from_report_rootless_call(self):
        # 报告根目录无法定位时如实None，不抛异常。
        self.assertIsNone(
            append_experiment_index(
                os.path.join(self.temp.name, "no_such_dir", "rep_e"),
                _analysis("rep_e"),
            )
        )


if __name__ == "__main__":
    unittest.main()
