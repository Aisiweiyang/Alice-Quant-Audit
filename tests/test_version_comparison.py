import json
import tempfile
import unittest
from pathlib import Path

from analysis.version_comparison import load_version_history


class VersionComparisonTests(unittest.TestCase):
    def test_reproducible_results_override_v9_and_v10_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "history.json"
            comparison = tmp_path / "comparison.json"
            base = {
                "total_return_pct": 1,
                "max_drawdown_pct": -1,
                "sharpe_ratio": 1,
                "excess_return_pct": 1,
                "win_rate_pct": 50,
                "profit_factor": 1,
            }
            history.write_text(
                json.dumps({"versions": [{"version": "V9", **base}, {"version": "V10", **base}]}),
                encoding="utf-8",
            )
            comparison.write_text(
                json.dumps(
                    {
                        "legacy_full_period": {"total_return_pct": 52.28},
                        "full_period": {"total_return_pct": 49.3},
                    }
                ),
                encoding="utf-8",
            )
            versions = load_version_history(history, comparison)
            self.assertEqual(versions[0]["total_return_pct"], 52.28)
            self.assertEqual(versions[1]["total_return_pct"], 49.3)

    def test_duplicate_versions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            row = {
                "version": "V3",
                "total_return_pct": 1,
                "max_drawdown_pct": -1,
                "sharpe_ratio": 1,
                "excess_return_pct": 1,
                "win_rate_pct": 50,
                "profit_factor": 1,
            }
            path.write_text(json.dumps({"versions": [row, row]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不能重复"):
                load_version_history(path)


if __name__ == "__main__":
    unittest.main()
