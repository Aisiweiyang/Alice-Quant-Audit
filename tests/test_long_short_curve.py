"""多空分位累计收益曲线测试（alphalens口径long-short spread可视化）。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from commercial.factor_diagnostics import compute_factor_diagnostics
from commercial.report import write_charts


def build_universe(stocks=40, days=160, seed=13):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    data = {}
    for index in range(stocks):
        code = f"sh.60{index:04d}"
        closes = 15 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, days)))
        preclose = np.concatenate([[closes[0]], closes[:-1]])
        volume = rng.lognormal(13, 0.3, days).round(0)
        data[code] = pd.DataFrame(
            {
                "date": dates,
                "open": (closes * 1.001).round(3),
                "high": (closes * 1.012).round(3),
                "low": (closes * 0.99).round(3),
                "close": closes.round(3),
                "volume": volume,
                "amount": (volume * closes).round(0),
                "preclose": preclose.round(3),
                "pctChg": ((closes / preclose - 1) * 100).round(4),
                "tradestatus": 1,
                "isST": 0,
                "pe": 20.0,
                "pb": 2.0,
            }
        )
    data["sh.000300"] = pd.DataFrame(
        {
            "date": dates,
            "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, days)))).round(2),
            "open": 1000.0,
        }
    )
    return data


class LongShortSpreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = build_universe()
        cls.summary, _corr, cls.metadata = compute_factor_diagnostics(
            data, "2025-03-03", "2025-08-29"
        )

    def test_metadata_contains_spread_series_for_ok_factors(self):
        spreads = self.metadata.get("long_short_spread_by_date") or {}
        self.assertTrue(spreads, "ok因子应保留逐信号日多空spread序列")
        ok = self.summary[self.summary.get("status") == "ok"]
        for feature in ok["feature"]:
            self.assertIn(feature, spreads)
            self.assertGreaterEqual(len(spreads[feature]), 1)
            date_label, value = spreads[feature][0]
            self.assertTrue(isinstance(date_label, str) and len(date_label) == 10)
            self.assertTrue(np.isfinite(value))

    def test_cumulative_curve_end_consistent_with_quintile_spread_mean(self):
        """自测：累计曲线终值 = spread之和 = 均值×期数，与既有
        quintile_spread_pct（均值口径）自洽。"""
        spreads = self.metadata.get("long_short_spread_by_date") or {}
        ok = self.summary[self.summary.get("status") == "ok"]
        matched = 0
        for _, row in ok.iterrows():
            spread_pct = row.get("quintile_spread_pct")
            points = spreads.get(row["feature"]) or []
            if not np.isfinite(spread_pct) or len(points) < 2:
                continue
            matched += 1
            values = [p[1] for p in points]
            final = float(np.sum(values))
            expected = float(spread_pct) * len(values)
            self.assertAlmostEqual(
                final,
                expected,
                delta=0.05,
                msg=f"{row['feature']} 累计终值与均值×期数不自洽",
            )
        self.assertGreater(matched, 0, "应至少有一个ok因子参与自洽校验")


class LongShortChartTests(unittest.TestCase):
    @staticmethod
    def _analysis_with(spread_by_date):
        """write_charts的因子图块要求factor_diagnostics非空；spread来自metadata。"""
        return {
            "factor_diagnostics": pd.DataFrame(
                [
                    {"feature": "momentum", "status": "ok"},
                    {"feature": "reversal", "status": "ok"},
                ]
            ),
            "factor_metadata": {"long_short_spread_by_date": spread_by_date},
        }

    def test_chart_generated_from_metadata(self):
        analysis = self._analysis_with(
            {
                "momentum": [["2025-03-03", 0.4], ["2025-03-10", -0.2], ["2025-03-17", 0.6]],
                "reversal": [["2025-03-03", -0.1], ["2025-03-10", 0.1]],
                "weak": [["2025-03-03", 0.2]],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertIn("charts/long_short.png", written)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "charts", "long_short.png")))

    def test_no_series_no_chart(self):
        analysis = self._analysis_with({})
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertNotIn("charts/long_short.png", written)

    def test_end_to_end_chart_embedded_and_hashed(self):
        from commercial.config import CommercialRequest
        from commercial.pipeline import run_full_analysis
        from tests.test_night_round import make_data, write_cache

        # 分位/spread计算要求每个信号日截面≥10只股票，故用14只。
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(make_data(stocks=14), temp_dir)
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
            root = analysis["report_dir"]
            self.assertIn("charts/long_short.png", analysis.get("charts_written") or [])
            self.assertTrue(os.path.isfile(os.path.join(root, "charts", "long_short.png")))
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("charts/long_short.png", html, "多空曲线应嵌入HTML因子小节")
            manifest = analysis.get("manifest") or {}
            self.assertIn("charts/long_short.png", manifest.get("outputs_sha256") or {})


if __name__ == "__main__":
    unittest.main()
