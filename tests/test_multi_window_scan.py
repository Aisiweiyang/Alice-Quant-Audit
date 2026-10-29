"""多窗口扫描脚本的测试：窗口解析、报告聚合与markdown渲染。

跑批部分（subprocess调commercial_mvp.py）不在单测覆盖内——真实跑批由
output/realdata_runs/下的实测记录验证（见docs/真实数据参考基线.md 3.2节）。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from multi_window_scan import (
    aggregate_report,
    parse_windows,
    render_markdown,
    should_run,
)


class ShouldRunTests(unittest.TestCase):
    def test_existing_report_reused(self):
        """已存在报告默认复用（同编号重跑会撞compare防覆盖）。"""
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "not_yet_run"
            self.assertTrue(should_run(missing), "不存在则应重跑")
            existing = Path(temp) / "already_run"
            existing.mkdir()
            self.assertFalse(should_run(existing))


class ParseWindowsTests(unittest.TestCase):
    def test_parses_two_windows_in_order(self):
        windows = parse_windows("2022-07-01:2023-06-30, 2023-07-01:2024-06-30")
        self.assertEqual(
            windows,
            [
                ("2022-07-01", "2023-06-30"),
                ("2023-07-01", "2024-06-30"),
            ],
        )

    def test_rejects_reversed_or_malformed_window(self):
        with self.assertRaises(ValueError):
            parse_windows("2023-07-01:2022-07-01")
        with self.assertRaises(ValueError):
            parse_windows("2022-07-01")
        with self.assertRaises(ValueError):
            parse_windows("2022/07/01:2023-07-01")
        with self.assertRaises(ValueError):
            parse_windows("")


class AggregateReportTests(unittest.TestCase):
    def test_reads_metrics_csv_and_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with open(root / "metrics.json", "w", encoding="utf-8") as h:
                json.dump(
                    {
                        "overfitting_audit": {
                            "n_trials": 4,
                            "pbo": {"pbo_probability": 0.2525},
                            "dispersion": {"validity": "ok", "max_pair_correlation": 0.5072},
                        },
                    },
                    h,
                )
            with open(root / "data_quality.json", "w", encoding="utf-8") as h:
                json.dump({"rating": "READY"}, h)
            with open(root / "scenario_comparison.csv", "w", encoding="utf-8-sig") as h:
                h.write(
                    "label,total_return_pct,max_drawdown_pct,sharpe_ratio,"
                    "deflated_sharpe,total_trades\n"
                )
                h.write("价值主导,5.11,-6.29,0.56,0.3544,61\n")
            snap = aggregate_report(root)
            self.assertEqual(snap["rating"], "READY")
            self.assertEqual(snap["pbo"], 0.2525)
            self.assertEqual(snap["validity"], "ok")
            self.assertEqual(snap["plans"]["价值主导"]["total_return_pct"], "5.11")
            self.assertEqual(snap["plans"]["价值主导"]["deflated_sharpe"], "0.3544")

    def test_missing_files_yield_none_not_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            snap = aggregate_report(Path(temp))
            self.assertIsNone(snap["pbo"])
            self.assertIsNone(snap["rating"])
            self.assertEqual(snap["plans"], {})


class RenderMarkdownTests(unittest.TestCase):
    def test_table_contains_plan_rows_and_audit_lines(self):
        snapshots = [
            {
                "report_id": "demo_w202207_202306",
                "rating": "READY",
                "pbo": 0.2525,
                "n_trials": 4,
                "validity": "ok",
                "max_correlation": 0.5072,
                "plans": {
                    "A": {
                        "total_return_pct": "5.11",
                        "max_drawdown_pct": "-6.29",
                        "sharpe_ratio": "0.56",
                        "deflated_sharpe": "0.3544",
                        "total_trades": "61",
                    }
                },
            }
        ]
        text = render_markdown(snapshots, ["A"])
        self.assertIn("demo_w202207_202306", text)
        self.assertIn("| A | 5.11 / -6.29 / 0.56 / 0.3544 / 61 |", text)
        self.assertIn("| PBO | 0.2525 |", text)
        self.assertIn("| 分散度validity | ok |", text)
        self.assertIn("不可外推", text)
        self.assertIn("不构成投资建议", text)

    def test_missing_plan_shows_dash(self):
        snapshots = [
            {
                "report_id": "r1",
                "rating": None,
                "pbo": None,
                "n_trials": None,
                "validity": None,
                "max_correlation": None,
                "plans": {},
            }
        ]
        text = render_markdown(snapshots, ["缺席方案"])
        self.assertIn("| 缺席方案 | — |", text)
        self.assertIn("| PBO | — |", text)


if __name__ == "__main__":
    unittest.main()
