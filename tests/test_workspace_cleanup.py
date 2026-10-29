"""工作区整理轮测试：归档目录排除、web_jobs迁移兼容。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from commercial_mvp import ARCHIVE_DIR_NAME, collect_reports


def _write_report(root, report_id, score=80):
    os.makedirs(os.path.join(root, report_id), exist_ok=True)
    with open(os.path.join(root, report_id, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": "2026-09-10T01:00:00+08:00",
                "blocked": False,
                "parameters": {"strategy": "classic_trend"},
            },
            handle,
        )
    with open(
        os.path.join(root, report_id, "strategy_audit.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"score": score, "grade_label": "可以继续研究"}, handle)


class ArchiveExclusionTests(unittest.TestCase):
    def test_collect_reports_skips_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_report(tmp, "live_report")
            _write_report(tmp, os.path.join(ARCHIVE_DIR_NAME, "old_report"))
            rows = collect_reports(tmp)
            ids = [row["report_id"] for row in rows]
            self.assertIn("live_report", ids)
            self.assertNotIn(
                os.path.join(ARCHIVE_DIR_NAME, "old_report"),
                ids,
                "归档目录内的报告不应出现在历史列表",
            )
            self.assertNotIn(ARCHIVE_DIR_NAME, ids)

    def test_archive_dir_constant(self):
        self.assertEqual(ARCHIVE_DIR_NAME, "_archive")


class WebJobsMigrationTests(unittest.TestCase):
    def test_legacy_jobs_migrated_to_state(self):
        import commercial_web

        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "legacy", "web_jobs.json")
            modern = os.path.join(tmp, "state", "web_jobs.json")
            os.makedirs(os.path.dirname(legacy))
            with open(legacy, "w", encoding="utf-8") as handle:
                json.dump({"job1": {"id": "job1", "status": "done"}}, handle)

            original = (
                commercial_web.JOBS_FILE,
                commercial_web.LEGACY_JOBS_FILE,
                commercial_web.STATE_DIR,
            )
            commercial_web.JOBS_FILE = modern
            commercial_web.LEGACY_JOBS_FILE = legacy
            commercial_web.STATE_DIR = os.path.dirname(modern)
            commercial_web.JOBS.clear()
            try:
                commercial_web._load_jobs()
                self.assertIn("job1", commercial_web.JOBS)
                commercial_web._persist_jobs()
                self.assertTrue(os.path.isfile(modern), "迁移后应写入新状态路径")
                self.assertFalse(
                    os.path.exists(legacy),
                    "旧文件在成功迁移后应被移除",
                )
            finally:
                (
                    commercial_web.JOBS_FILE,
                    commercial_web.LEGACY_JOBS_FILE,
                    commercial_web.STATE_DIR,
                ) = original
                commercial_web.JOBS.clear()

    def test_new_location_preferred_over_legacy(self):
        import commercial_web

        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "legacy", "web_jobs.json")
            modern = os.path.join(tmp, "state", "web_jobs.json")
            os.makedirs(os.path.dirname(legacy))
            os.makedirs(os.path.dirname(modern))
            with open(legacy, "w", encoding="utf-8") as handle:
                json.dump({"old": {"id": "old"}}, handle)
            with open(modern, "w", encoding="utf-8") as handle:
                json.dump({"new": {"id": "new"}}, handle)

            original = (
                commercial_web.JOBS_FILE,
                commercial_web.LEGACY_JOBS_FILE,
                commercial_web.STATE_DIR,
            )
            commercial_web.JOBS_FILE = modern
            commercial_web.LEGACY_JOBS_FILE = legacy
            commercial_web.STATE_DIR = os.path.dirname(modern)
            commercial_web.JOBS.clear()
            try:
                commercial_web._load_jobs()
                self.assertIn("new", commercial_web.JOBS)
                self.assertNotIn("old", commercial_web.JOBS)
            finally:
                (
                    commercial_web.JOBS_FILE,
                    commercial_web.LEGACY_JOBS_FILE,
                    commercial_web.STATE_DIR,
                ) = original
                commercial_web.JOBS.clear()


class PruneReportsTests(unittest.TestCase):
    def _make_root(self, tmp, count=5):
        for index in range(count):
            _write_report(tmp, f"rep_{index:02d}", score=60 + index)
            # 手动错开created_at便于排序断言
            manifest_path = os.path.join(tmp, f"rep_{index:02d}", "manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["created_at"] = f"2026-09-{10 + index:02d}T01:00:00+08:00"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False)

    def test_prune_preview_changes_nothing(self):
        from commercial_mvp import _prune_reports

        with tempfile.TemporaryDirectory() as tmp:
            self._make_root(tmp)
            _prune_reports(tmp, keep=2, execute=False)
            self.assertEqual(len(collect_reports(tmp)), 5, "预览不得移动任何报告")
            self.assertFalse(os.path.isdir(os.path.join(tmp, ARCHIVE_DIR_NAME)))

    def test_prune_archive_moves_and_zips(self):
        from commercial_mvp import _prune_reports

        with tempfile.TemporaryDirectory() as tmp:
            self._make_root(tmp)
            _prune_reports(tmp, keep=2, execute=True)
            rows = collect_reports(tmp)
            self.assertEqual(len(rows), 2, "归档后主目录只剩保留的最新2份")
            self.assertEqual(
                sorted(r["report_id"] for r in rows),
                ["rep_03", "rep_04"],
                "应保留created_at最新的报告",
            )
            archive_dir = os.path.join(tmp, ARCHIVE_DIR_NAME)
            self.assertTrue(os.path.isfile(os.path.join(archive_dir, "rep_00.zip")))
            self.assertTrue(os.path.isdir(os.path.join(archive_dir, "rep_00")))

    def test_prune_noop_when_under_keep(self):
        from commercial_mvp import _prune_reports

        with tempfile.TemporaryDirectory() as tmp:
            self._make_root(tmp, count=2)
            _prune_reports(tmp, keep=5, execute=True)
            self.assertEqual(len(collect_reports(tmp)), 2)


class FactorHeatmapChartTests(unittest.TestCase):
    def test_ic_by_date_and_quantile_heatmaps_written(self):

        from commercial.report import write_charts

        ic_by_date = {
            "momentum": [[f"2025-01-{d:02d}", 0.02 + d * 0.001] for d in range(1, 9)],
            "rsi": [[f"2025-01-{d:02d}", -0.01 + d * 0.0005] for d in range(1, 8)],
        }
        factors = pd.DataFrame(
            [
                dict(
                    feature="momentum",
                    status="ok",
                    **{f"q{j}_mean_fwd_pct": 0.2 * j - 0.5 for j in range(1, 6)},
                ),
                dict(
                    feature="rsi",
                    status="ok",
                    **{f"q{j}_mean_fwd_pct": 0.5 - 0.2 * j for j in range(1, 6)},
                ),
            ]
        )
        analysis = {
            "factor_diagnostics": factors,
            "factor_metadata": {"ic_by_date": ic_by_date},
        }
        with tempfile.TemporaryDirectory() as tmp:
            import contextlib

            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertIn("charts/ic_by_date.png", written)
            self.assertIn("charts/quantile_heatmap.png", written)
            for name in ("ic_by_date.png", "quantile_heatmap.png"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "charts", name)))

    def test_no_ic_heatmap_without_metadata(self):
        from commercial.report import write_charts

        analysis = {"factor_diagnostics": None, "factor_metadata": None}
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertNotIn("charts/ic_by_date.png", written)


class AuditEquityChartTests(unittest.TestCase):
    def _analysis_with_equity(self):
        import numpy as np

        rng = np.random.default_rng(5)
        totals = 100000 * np.cumprod(1 + rng.normal(0.0006, 0.01, 200))
        equity = pd.DataFrame(
            {
                "date": pd.bdate_range("2025-01-01", periods=200).strftime("%Y-%m-%d"),
                "cash": 20000.0,
                "holdings_value": totals * 0.8,
                "total": totals,
                "positions": 8,
                "unpriced_positions": 0,
            }
        )
        return {"base": {"equity": equity}}

    _EXPECTED = (
        "charts/drawdown.png",
        "charts/return_distribution.png",
        "charts/rolling_sharpe.png",
        "charts/exposure.png",
    )

    def test_audit_charts_written_from_equity(self):
        from commercial.report import write_charts

        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(
                tmp, self._analysis_with_equity(), charts_dir=Path(tmp) / "charts"
            )
            for name in self._EXPECTED:
                self.assertIn(name, written)
                self.assertTrue(os.path.isfile(os.path.join(tmp, "charts", os.path.basename(name))))

    def test_audit_charts_absent_without_equity(self):
        from commercial.report import write_charts

        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, {"base": {}}, charts_dir=Path(tmp) / "charts")
            for name in self._EXPECTED:
                self.assertNotIn(name, written)


if __name__ == "__main__":
    unittest.main()
