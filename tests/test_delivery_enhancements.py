"""交付增强轮测试：reports --trend、网页方案校验、rolling_windows.png、抽样角标。"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from commercial.compare import ParameterError, parse_plan_payload
from commercial.factor_diagnostics import render_factor_section_html
from commercial.report import write_charts
from commercial_mvp import _print_trend


VALID_PLAN = json.dumps(
    {
        "plans": [
            {"label": "基线", "overrides": {}},
            {"label": "高止损", "overrides": {"stop_loss_pct": 0.08}},
        ]
    },
    ensure_ascii=False,
)


class ComparePlanPayloadTests(unittest.TestCase):
    def test_valid_plan_passes(self):
        plans = parse_plan_payload(json.loads(VALID_PLAN))
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[1]["overrides"]["stop_loss_pct"], 0.08)

    def test_unknown_override_rejected(self):
        payload = {"plans": [{"label": "a"}, {"label": "b", "overrides": {"hack": 1}}]}
        with self.assertRaises(ParameterError):
            parse_plan_payload(payload)

    def test_plan_count_bounds(self):
        with self.assertRaises(ParameterError):
            parse_plan_payload({"plans": [{"label": "a"}]})
        with self.assertRaises(ParameterError):
            parse_plan_payload({"plans": [{"label": str(i)} for i in range(9)]})


class ReportsTrendTests(unittest.TestCase):
    def test_print_trend_chronological_and_declaration(self):
        rows = [
            {
                "report_id": "new",
                "created_at": "2026-09-10T01:00:00+08:00",
                "score": 82,
                "grade_label": "可以继续研究",
                "rating": "READY",
                "blocked": False,
            },
            {
                "report_id": "old",
                "created_at": "2026-09-08T01:00:00+08:00",
                "score": 74,
                "grade_label": "可以继续研究",
                "rating": "LIMITED",
                "blocked": False,
            },
        ]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _print_trend(rows)
        output = buffer.getvalue()
        old_pos = output.index("old")
        new_pos = output.index("new")
        self.assertLess(old_pos, new_pos, "趋势表应按时间正序（旧→新）")
        self.assertIn("不评价未来收益", output)
        self.assertIn("74", output)
        self.assertIn("82", output)


class WebValidatePageTests(unittest.TestCase):
    def test_validate_page_accepts_valid_plan(self):
        import commercial_web

        html = commercial_web.handle_validate({"plan_json": VALID_PLAN})
        self.assertIn("校验通过", html)
        self.assertIn("不执行任何回测", html)

    def test_validate_page_reports_errors(self):
        import commercial_web

        html = commercial_web.handle_validate({"plan_json": "{bad json"})
        self.assertIn("JSON解析失败", html)
        payload = json.loads('{"plans": [{"label": "a"}, {"label": "b", "overrides": {"x": 1}}]}')
        html = commercial_web.handle_validate({"plan_json": json.dumps(payload)})
        self.assertIn("校验未通过", html)
        html = commercial_web.handle_validate({"plan_json": ""})
        self.assertIn("内容为空", html)

    def test_trend_page_renders_empty(self):
        import commercial_web

        html = commercial_web.render_reports_trend()
        self.assertIn("趋势", html)


class RollingChartTests(unittest.TestCase):
    def test_rolling_windows_chart_written(self):
        analysis = {
            "rolling_oos": {
                "status": "ok",
                "windows": [
                    {"window": "W1", "status": "ok", "total_return_pct": 3.2},
                    {"window": "W2", "status": "ok", "total_return_pct": -1.1},
                    {"window": "W3", "status": "insufficient_data"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertIn("charts/rolling_windows.png", written)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "charts", "rolling_windows.png")))
            self.assertIn("charts/rolling_windows.png", analysis["charts_written"])

    def test_no_rolling_no_chart(self):
        analysis = {"rolling_oos": {"status": "insufficient_data", "windows": []}}
        with tempfile.TemporaryDirectory() as tmp:
            written = write_charts(tmp, analysis, charts_dir=Path(tmp) / "charts")
            self.assertNotIn("charts/rolling_windows.png", written)


class SamplingBadgeTests(unittest.TestCase):
    def test_sampling_badge_rendered_when_applied(self):
        summary = pd.DataFrame(
            [
                {
                    "feature": "momentum",
                    "status": "ok",
                    "coverage_pct": 95.0,
                    "mean_rank_ic": 0.02,
                    "rank_icir": 0.1,
                    "positive_ic_rate_pct": 55.0,
                    "ic_5d": 0.01,
                    "ic_20d": 0.0,
                    "direction_flip": None,
                },
            ]
        )
        metadata = {
            "sampling": {
                "applied": True,
                "requested": 3400,
                "used": 800,
                "note": "股票池3400只超过上限800，按代码排序等距抽样800只参与因子诊断",
            },
            "signal_rule": "每周第一个交易日为信号日",
            "signal_count": 30,
            "duplicate_pairs": [],
            "exposure_note": "",
            "disclaimer": "不构成投资建议。",
        }
        html = render_factor_section_html(summary, pd.DataFrame(), metadata)
        self.assertIn("抽样说明", html)
        self.assertIn("抽样估计", html)

    def test_no_badge_when_not_sampled(self):
        summary = pd.DataFrame(
            [
                {
                    "feature": "momentum",
                    "status": "ok",
                    "coverage_pct": 95.0,
                    "mean_rank_ic": 0.02,
                    "rank_icir": 0.1,
                    "positive_ic_rate_pct": 55.0,
                    "ic_5d": 0.01,
                    "ic_20d": 0.0,
                    "direction_flip": None,
                },
            ]
        )
        metadata = {
            "sampling": {"applied": False, "requested": 400, "used": 400},
            "signal_rule": "每周第一个交易日为信号日",
            "signal_count": 30,
            "duplicate_pairs": [],
            "exposure_note": "",
            "disclaimer": "不构成投资建议。",
        }
        html = render_factor_section_html(summary, pd.DataFrame(), metadata)
        self.assertNotIn("抽样说明", html)


if __name__ == "__main__":
    unittest.main()
