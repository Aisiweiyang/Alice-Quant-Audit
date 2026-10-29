"""网页实验索引页（/experiments）的测试：只读渲染experiments_index.json。"""

import json
import os
import tempfile
import unittest

import commercial_web
from commercial_web import render_experiments, render_home


class ExperimentsPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_root = commercial_web.REPORTS_ROOT
        commercial_web.REPORTS_ROOT = self.temp.name

    def tearDown(self):
        commercial_web.REPORTS_ROOT = self.previous_root
        self.temp.cleanup()

    def test_missing_index_shows_honest_empty_state(self):
        html = render_experiments()
        self.assertIn("还没有记录", html)
        self.assertIn("不构成投资建议", html)

    def test_valid_index_renders_rows_newest_first(self):
        records = [
            {
                "report_id": "rep_old",
                "created_at": "2026-09-13T10:00:00+08:00",
                "mode": "full",
                "strategy": "classic_trend",
                "start": "2024-01-01",
                "end": "2026-06-30",
                "score": 71,
                "data_rating": "READY",
                "key_metrics": {"total_return_pct": 45.48},
            },
            {
                "report_id": "rep_new",
                "created_at": "2026-09-14T01:16:09+08:00",
                "mode": "compare",
                "strategy": "cross_sectional_alpha158",
                "start": "2025-07-01",
                "end": "2026-06-30",
                "score": None,
                "data_rating": "LIMITED",
                "key_metrics": {"total_return_pct": None},
            },
        ]
        path = os.path.join(commercial_web.REPORTS_ROOT, "experiments_index.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(records, handle)
        html = render_experiments()
        self.assertIn("rep_old", html)
        self.assertIn("rep_new", html)
        self.assertIn("45.48", html)
        self.assertIn("READY", html)
        # 新→旧：rep_new出现在rep_old之前
        self.assertLess(html.index("rep_new"), html.index("rep_old"))

    def test_corrupted_index_never_fakes_data(self):
        path = os.path.join(commercial_web.REPORTS_ROOT, "experiments_index.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{broken")
        html = render_experiments()
        self.assertIn("损坏", html)
        self.assertNotIn("<td>—</td><td>READY", html)

    def test_reports_page_links_to_experiments(self):
        html = render_home()
        self.assertIn('href="/experiments"', html)

    def _write_two_records(self):
        records = [
            {
                "report_id": "rep_trend",
                "created_at": "2026-09-13T10:00:00+08:00",
                "mode": "full",
                "strategy": "classic_trend",
                "start": "2024-01-01",
                "end": "2026-06-30",
                "score": 71,
                "data_rating": "READY",
                "key_metrics": {"total_return_pct": 45.48},
            },
            {
                "report_id": "rep_csx",
                "created_at": "2026-09-14T01:16:09+08:00",
                "mode": "compare",
                "strategy": "cross_sectional_alpha158",
                "start": "2025-07-01",
                "end": "2026-06-30",
                "score": None,
                "data_rating": "LIMITED",
                "key_metrics": {"total_return_pct": None},
            },
        ]
        path = os.path.join(commercial_web.REPORTS_ROOT, "experiments_index.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(records, handle)
        return records

    def test_strategy_filter_shows_matching_rows_and_banner(self):
        """?strategy=只读过滤，页面显示当前条件与清除链接。"""
        self._write_two_records()
        html = render_experiments(strategy="cross_sectional_alpha158")
        self.assertIn("rep_csx", html)
        self.assertNotIn("rep_trend", html)
        self.assertIn("当前过滤：策略=cross_sectional_alpha158", html)
        self.assertIn("清除过滤", html)
        # 大小写不敏感。
        self.assertNotIn("rep_trend", render_experiments(strategy="CROSS_SECTIONAL_ALPHA158"))

    def test_rating_filter_and_combined_no_match(self):
        self._write_two_records()
        html = render_experiments(rating="ready")
        self.assertIn("rep_trend", html)
        self.assertNotIn("rep_csx", html)
        self.assertIn("当前过滤：数据评级=ready", html)
        both = render_experiments(strategy="classic_trend", rating="LIMITED")
        self.assertIn("没有符合当前过滤条件的记录", both)
        self.assertNotIn("rep_trend", both)
        self.assertNotIn("rep_csx", both)

    def test_filter_does_not_modify_index_file(self):
        self._write_two_records()
        path = os.path.join(commercial_web.REPORTS_ROOT, "experiments_index.json")
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        render_experiments(strategy="classic_trend", rating="READY")
        with open(path, encoding="utf-8") as handle:
            after = handle.read()
        self.assertEqual(before, after, "过滤是只读行为，不得改写索引文件")

    def test_no_filter_keeps_all_rows(self):
        self._write_two_records()
        html = render_experiments()
        self.assertIn("rep_trend", html)
        self.assertIn("rep_csx", html)
        self.assertNotIn("当前过滤", html)


if __name__ == "__main__":
    unittest.main()
