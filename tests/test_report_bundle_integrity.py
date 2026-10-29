"""报告产物完整性测试：图表四点闭环（生成→存在→嵌入→manifest哈希）与
manifest清单闭合（manifest.json自身不入清单，其余哈希与落盘文件一致）。

背景（2026-09-13 真实数据报告收尾发现）：
- equity_curve/monthly_returns/trade_distribution/scenario_deltas四张图有
  生成代码但从未嵌入HTML；
- manifest双写导致记录的manifest.json自引用哈希是已废弃前版的哈希，
  永远无法与落盘文件一致。
"""

import json
import os
import tempfile
import unittest

from commercial.report import render_html, write_report_bundle
from commercial.pipeline import run_full_analysis
from tests.test_night_round import make_data, make_request, write_cache

EMBEDDED_CHARTS = (
    "charts/drawdown.png",
    "charts/return_distribution.png",
    "charts/rolling_sharpe.png",
    "charts/exposure.png",
    "charts/equity_curve.png",
    "charts/monthly_returns.png",
    "charts/trade_distribution.png",
    "charts/scenario_deltas.png",
    "charts/rolling_windows.png",
    "charts/ic_decay.png",
    "charts/quantile_returns.png",
    "charts/ic_by_date.png",
    "charts/quantile_heatmap.png",
    "charts/long_short.png",
)


class ReportBundleIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        data = make_data()
        cache = write_cache(data, cls.temp_dir.name)
        request = make_request(cache, cls.temp_dir.name)  # 默认含压力测试
        cls.analysis = run_full_analysis(request, report_id="integrity_probe")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_all_written_charts_embedded_in_html(self):
        written = self.analysis.get("charts_written") or []
        self.assertTrue(written, "合成full运行应产出图表")
        html_path = os.path.join(self.analysis["report_dir"], "index.html")
        with open(html_path, encoding="utf-8") as handle:
            html = handle.read()
        missing = [name for name in written if f"<img src='{name}'" not in html]
        self.assertEqual(missing, [])

    def test_expected_core_charts_written(self):
        written = set(self.analysis.get("charts_written") or [])
        for name in (
            "charts/equity_curve.png",
            "charts/monthly_returns.png",
            "charts/trade_distribution.png",
            "charts/scenario_deltas.png",
        ):
            self.assertIn(name, written)

    def test_manifest_excludes_self_and_covers_all_outputs(self):
        report_dir = self.analysis["report_dir"]
        manifest = self.analysis["manifest"]
        outputs = manifest["outputs_sha256"]
        self.assertNotIn("manifest.json", outputs)
        self.assertIn("index.html", outputs)
        on_disk = set()
        for base, _dirs, files in os.walk(report_dir):
            for name in files:
                relative = os.path.relpath(os.path.join(base, name), report_dir).replace("\\", "/")
                if relative != "manifest.json":
                    on_disk.add(relative)
        self.assertEqual(set(outputs), on_disk)

    def test_manifest_hashes_match_files(self):
        from commercial.report import sha256_file

        report_dir = self.analysis["report_dir"]
        for name, recorded in self.analysis["manifest"]["outputs_sha256"].items():
            actual = sha256_file(os.path.join(report_dir, name))
            self.assertEqual(actual, recorded, f"{name} 哈希不一致")

    def test_experiment_index_created_next_to_report(self):
        # 报告落盘后，报告根目录自动出现追加式实验索引并含本报告摘要。
        index_path = os.path.join(
            os.path.dirname(self.analysis["report_dir"]),
            "experiments_index.json",
        )
        self.assertTrue(os.path.isfile(index_path))
        with open(index_path, encoding="utf-8") as handle:
            records = json.load(handle)
        entry = next(r for r in records if r["report_id"] == "integrity_probe")
        self.assertEqual(entry["mode"], "full")
        self.assertIsNotNone(entry["data_rating"])

    def test_second_bundle_write_stays_consistent(self):
        # 重跑/修复写同一报告目录：陈旧manifest.json不得混入新清单。
        analysis = dict(self.analysis)
        analysis["manifest"] = None
        manifest = write_report_bundle(self.analysis["report_dir"], analysis)
        self.assertNotIn("manifest.json", manifest["outputs_sha256"])


class ChartEmbeddingGuardTests(unittest.TestCase):
    """未生成的图不得出现img标签（防死链）；生成的图必须嵌入。"""

    def _analysis(self, charts_written):
        from commercial.config import CommercialRequest

        metrics = {
            "total_return_pct": 5.0,
            "cagr_pct": 5.0,
            "max_drawdown_pct": -8.0,
            "annual_volatility_pct": 15.0,
            "sharpe_ratio": 1.0,
            "sharpe_observations": 120,
            "sharpe_reliable": True,
            "benchmark_return_pct": 2.0,
            "excess_return_pct": 3.0,
            "total_trades": 12,
            "win_rate_pct": 50.0,
        }
        return {
            "report_id": "chart_guard",
            "request": CommercialRequest(cache="x.pkl", start="2025-01-01", end="2025-06-30"),
            "score": {"dimensions": [], "score": 60, "grade_label": "结论不稳定"},
            "data_audit": {"rating": "READY", "warnings": [], "blockers": []},
            "base": {"metrics": metrics, "statistics": {}},
            "charts_written": charts_written,
        }

    def test_scenario_deltas_embedded_only_when_written(self):
        with_chart = render_html(self._analysis(["charts/scenario_deltas.png"]))
        self.assertIn("charts/scenario_deltas.png", with_chart)
        without = render_html(self._analysis([]))
        self.assertNotIn("scenario_deltas.png", without)

    def test_core_charts_embedded_when_written(self):
        names = [
            "charts/equity_curve.png",
            "charts/monthly_returns.png",
            "charts/trade_distribution.png",
        ]
        html = render_html(self._analysis(names))
        for name in names:
            self.assertIn(f"<img src='{name}'", html)


if __name__ == "__main__":
    unittest.main()
