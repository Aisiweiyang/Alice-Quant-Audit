"""Publication-specific portability and reproducible synthetic-data checks."""

import json
from pathlib import Path
import tempfile
import unittest

from data.cache_store import load_market_cache
from commercial.config import CommercialRequest
from commercial.report import render_html
from examples.commercial_demo.generate_demo_data import generate
from run_metadata import PROJECT_DIR, WORKSPACE_DIR
from tests.golden_builder import GOLDEN_PATH, dataset_fingerprint


class PublicReleaseTests(unittest.TestCase):
    def test_audit_only_is_not_described_as_data_failure(self):
        request = CommercialRequest(cache="demo.pkl", start="2025-07-01", end="2025-12-31")
        document = render_html(
            {
                "request": request,
                "report_id": "audit",
                "blocked": True,
                "effective_config": {"mode": "data_audit_only"},
            }
        )
        self.assertIn("本次仅做数据审计，未运行回测", document)
        self.assertNotIn("数据缺陷足以扭曲回测结论", document)

    def test_synthetic_source_notice_precedes_score(self):
        request = CommercialRequest(cache="demo.pkl", start="2025-07-01", end="2025-12-31")
        document = render_html(
            {"request": request, "report_id": "demo", "data_source": "SYNTHETIC-DEMO:<script>"}
        )
        self.assertLess(document.index("SYNTHETIC 合成演示数据"), document.index("1. 一页结论"))
        self.assertIn("&lt;script&gt;", document)
        self.assertNotIn("SYNTHETIC-DEMO:<script>", document)

    def test_repository_metadata_uses_standalone_root(self):
        self.assertEqual(Path(WORKSPACE_DIR), Path(PROJECT_DIR))

    def test_generated_data_matches_original_logical_anchor(self):
        with open(GOLDEN_PATH, encoding="utf-8") as handle:
            expected = json.load(handle)["pipeline"]["dataset_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            generate(directory)
            data, _ = load_market_cache(str(Path(directory) / "demo_cache.pkl"), "sh.000300")
            self.assertEqual(dataset_fingerprint(data), expected)


if __name__ == "__main__":
    unittest.main()
