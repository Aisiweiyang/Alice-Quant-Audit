"""reports --windows（只读查看最近一次多窗口扫描汇总）的测试。"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from commercial_mvp import _print_windows_scan


class WindowsScanReportTests(unittest.TestCase):
    def test_prints_newest_summary_and_its_source(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, text in [
                ("multi_window_scan_20260915_101010.md", "旧汇总 PBO=0.25"),
                ("multi_window_scan_20260916_121212.md", "新汇总 PBO=0.01"),
            ]:
                with open(os.path.join(temp, name), "w", encoding="utf-8") as h:
                    h.write(text)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = _print_windows_scan(scan_dir=temp)
            out = buffer.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("新汇总", out)
            self.assertNotIn("旧汇总", out, "应只输出最新一份")
            self.assertIn("multi_window_scan_20260916_121212.md", out)

    def test_no_summary_honest_message(self):
        with tempfile.TemporaryDirectory() as temp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = _print_windows_scan(scan_dir=temp)
            self.assertEqual(code, 0)
            self.assertIn("还没有多窗口扫描汇总", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
