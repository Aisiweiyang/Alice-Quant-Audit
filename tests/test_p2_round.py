"""P2批次测试：滚动锚定窗口、全指标快照表、CLI --rolling-anchored接入。"""

import os
import sys
import tempfile
import unittest

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from commercial.config import CommercialRequest, request_from_args
from commercial.pipeline import run_full_analysis
from commercial.report import render_snapshot_table
from commercial.rolling_oos import run_rolling_oos, split_windows
from tests.test_night_round import make_data, write_cache


class AnchoredWindowTests(unittest.TestCase):
    def test_split_windows_default_non_overlapping(self):
        segments = split_windows("2025-01-01", "2025-07-01", 3)
        self.assertEqual(len(segments), 3)
        for index in range(2):
            self.assertLess(segments[index][2], segments[index + 1][1], "缺省窗口不应重叠")

    def test_split_windows_anchored_start_fixed_and_growing(self):
        segments = split_windows("2025-01-01", "2025-07-01", 3, anchored=True)
        self.assertEqual(len(segments), 3)
        starts = [s[1] for s in segments]
        ends = [s[2] for s in segments]
        self.assertEqual(len(set(starts)), 1, "锚定窗口起点应全部固定为区间起点")
        self.assertEqual(starts[0], "2025-01-01")
        self.assertLess(ends[0], ends[1])
        self.assertLess(ends[1], ends[2], "锚定窗口终点应递进")
        self.assertEqual(ends[-1], "2025-07-01")

    def test_run_rolling_oos_anchored_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=8), temp_dir)
            request = CommercialRequest(
                cache=cache,
                start="2025-02-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
            )
            request.validate()
            from commercial.config import build_engine_config

            config = build_engine_config(request)
            anchored = run_rolling_oos(
                request,
                config,
                cache,
                windows=3,
                data=make_data(stocks=8),
                anchored=True,
            )
            self.assertEqual(anchored.get("status"), "ok")
            self.assertTrue(anchored.get("anchored"))
            self.assertIn("锚定", anchored.get("note", ""))
            starts = {w["start"] for w in anchored["windows"] if w["status"] == "ok"}
            self.assertGreaterEqual(len(starts), 1)
            ends = [w["end"] for w in anchored["windows"] if w["status"] == "ok"]
            self.assertEqual(ends, sorted(ends), "锚定窗口终点应递增")


class SnapshotTableTests(unittest.TestCase):
    def test_renders_labels_and_dash_for_missing(self):
        html = render_snapshot_table(
            {
                "total_return_pct": 12.34,
                "sharpe_ratio": 1.1,
                "sharpe_ci_low": None,
            }
        )
        self.assertIn("全指标快照", html)
        self.assertIn("12.34%", html)
        self.assertIn("Deflated Sharpe", html)
        self.assertIn("信息比率", html)
        self.assertIn("—", html, "缺失字段应显示—")

    def test_full_report_contains_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=8), temp_dir)
            request = CommercialRequest(
                cache=cache,
                start="2025-05-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
                run_stress=False,
            )
            request.validate()
            analysis = run_full_analysis(request)
            with open(
                os.path.join(analysis["report_dir"], "index.html"), encoding="utf-8"
            ) as handle:
                html = handle.read()
            self.assertIn("全指标快照", html)
            self.assertIn("回收因子", html)


class AnchoredCliTests(unittest.TestCase):
    def test_rolling_anchored_flag_reaches_request(self):
        from commercial_mvp import build_parser

        args = build_parser().parse_args(
            [
                "full",
                "--cache",
                "x.pkl",
                "--start",
                "2025-01-01",
                "--end",
                "2025-02-01",
                "--rolling-anchored",
            ]
        )
        request = request_from_args(args)
        self.assertTrue(request.rolling_anchored)

    def test_default_not_anchored(self):
        from commercial_mvp import build_parser

        args = build_parser().parse_args(
            [
                "full",
                "--cache",
                "x.pkl",
                "--start",
                "2025-01-01",
                "--end",
                "2025-02-01",
            ]
        )
        request = request_from_args(args)
        self.assertFalse(request.rolling_anchored)


if __name__ == "__main__":
    unittest.main()
