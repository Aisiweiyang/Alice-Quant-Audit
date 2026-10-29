import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.parameter_comparison import write_parameter_comparison


class ParameterComparisonTests(unittest.TestCase):
    def _run(self, root: Path, name: str, sharpe: float, final: float) -> Path:
        run = root / "runs" / name
        reports = run / "reports"
        reports.mkdir(parents=True)
        metrics = {
            "total_return_pct": final / 1000 - 100,
            "cagr_pct": 10.0,
            "max_drawdown_pct": -8.0,
            "sharpe_ratio": sharpe,
            "excess_return_pct": 2.0,
            "total_trades": 4,
            "win_rate_pct": 50.0,
            "profit_factor": 1.2,
        }
        (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (reports / "run_statistics.json").write_text(
            json.dumps({"annualized_turnover": 3.2, "total_friction": 120.0}), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "date": pd.bdate_range("2025-01-02", periods=5),
                "total": [100_000, 101_000, 100_500, 102_000, final],
            }
        ).to_csv(reports / "equity.csv", index=False)
        (run / "index.html").write_text("single", encoding="utf-8")
        return run

    def test_comparison_writes_html_csv_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self._run(root, "first", 1.1, 105_000)
            second = self._run(root, "second", 1.4, 108_000)
            report = write_parameter_comparison(
                root,
                [
                    {"name": "基线", "description": "默认", "run_dir": first},
                    {"name": "分散", "description": "12只", "run_dir": second},
                ],
                "测试实验",
                "2025-01-02",
                "2025-01-08",
            )
            content = report.read_text(encoding="utf-8")
            self.assertTrue((root / "comparison.csv").exists())
            self.assertTrue((root / "comparison.json").exists())
        self.assertIn("测试实验", content)
        self.assertIn("风险—收益位置", content)


if __name__ == "__main__":
    unittest.main()
