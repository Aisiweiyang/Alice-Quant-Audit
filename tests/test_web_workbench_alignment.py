"""网页工作台对齐CLI能力的测试：表单新参数接线 + compare状态页过拟合摘要。

背景（2026-09-12轮）：CLI已有 strategy/trials/rolling-anchored/no-segment/
strategy-param 与compare过拟合审计，网页表单与状态页此前未暴露这些能力。
要求：所有参数必须经过request.validate()白名单，防止"argparse加了但没接线"。
"""

import threading
import unittest
from http.server import ThreadingHTTPServer

from commercial.config import ParameterError
from commercial_web import (
    AuditHandler,
    JOBS,
    JOBS_LOCK,
    _overfitting_summary,
    parse_form,
    render_home,
    render_status,
    request_from_form,
)


def form_body(pairs: dict) -> dict:
    """模拟parse_form的输出：值为字符串。"""
    return {k: str(v) for k, v in pairs.items()}


class FormWiringTests(unittest.TestCase):
    base = {
        "cache": "a.pkl",
        "start": "2025-01-01",
        "end": "2025-06-30",
    }

    def test_strategy_and_trials_wired(self):
        form = parse_form(
            b"cache=a.pkl&start=2025-01-01&end=2025-06-30"
            b"&strategy=cross_sectional_alpha158&trials=5"
        )
        request = request_from_form(form)
        self.assertEqual(request.strategy, "cross_sectional_alpha158")
        self.assertEqual(request.trials, 5)

    def test_anchored_and_segment_flags_wired(self):
        form = parse_form(
            b"cache=a.pkl&start=2025-01-01&end=2025-06-30&no_segment=1&rolling_anchored=1"
        )
        request = request_from_form(form)
        self.assertFalse(request.run_segment_consistency)
        self.assertTrue(request.rolling_anchored)
        # 未勾选时保持默认开启/关闭
        default = request_from_form(parse_form(b"cache=a.pkl&start=2025-01-01&end=2025-06-30"))
        self.assertTrue(default.run_segment_consistency)
        self.assertFalse(default.rolling_anchored)

    def test_strategy_params_parsed_and_validated(self):
        form = parse_form(
            b"cache=a.pkl&start=2025-01-01&end=2025-06-30"
            b"&strategy=cross_sectional_alpha158"
            b"&strategy_params=minimum_score%3D60%2Cexit_rank%3D15"
        )
        request = request_from_form(form)
        self.assertEqual(request.strategy_params["minimum_score"], 60.0)
        self.assertEqual(request.strategy_params["exit_rank"], 15)

    def test_strategy_params_bad_format_rejected(self):
        form = form_body({**self.base, "strategy_params": "minimum_score"})
        with self.assertRaises(ParameterError):
            request_from_form(form)

    def test_strategy_params_unknown_key_rejected_by_whitelist(self):
        form = form_body(
            {
                **self.base,
                "strategy": "cross_sectional_alpha158",
                "strategy_params": "not_in_whitelist=3",
            }
        )
        with self.assertRaises(ParameterError):
            request_from_form(form)

    def test_strategy_params_rejected_for_classic_trend(self):
        form = form_body(
            {
                **self.base,
                "strategy": "classic_trend",
                "strategy_params": "minimum_score=60",
            }
        )
        with self.assertRaises(ParameterError):
            request_from_form(form)

    def test_trials_out_of_range_rejected(self):
        form = form_body({**self.base, "trials": "1001"})
        with self.assertRaises(ParameterError):
            request_from_form(form)


class HomePageContentTests(unittest.TestCase):
    def test_home_has_new_controls_and_disclaimer(self):
        html = render_home()
        for needle in (
            'name="strategy"',
            'value="cross_sectional_alpha158"',
            'name="trials"',
            'name="strategy_params"',
            'name="rolling_anchored"',
            'name="no_segment"',
            "不构成投资建议",
        ):
            self.assertIn(needle, html)


class OverfittingSummaryTests(unittest.TestCase):
    def test_summary_extraction_from_analysis(self):
        analysis = {
            "mode": "compare",
            "overfitting_audit": {
                "n_trials": 3,
                "pbo": {
                    "status": "ok",
                    "pbo_probability": 0.42,
                    "cscv_combinations": 12870,
                },
                "dispersion": {
                    "validity": "ok",
                    "max_pair_correlation": 0.42,
                },
            },
            "plan_results": [
                {
                    "label": "基线",
                    "status": "ok",
                    "deflated_sharpe": {
                        "status": "ok",
                        "deflated_sharpe": 0.9,
                        "sr_observed": 0.08,
                    },
                },
                {"label": "坏方案", "status": "insufficient_data", "detail": "失败"},
            ],
        }
        summary = _overfitting_summary(analysis)
        self.assertEqual(summary["n_trials"], 3)
        self.assertEqual(summary["pbo_probability"], 0.42)
        self.assertEqual(summary["cscv_combinations"], 12870)
        self.assertEqual(summary["dispersion_validity"], "ok")
        self.assertEqual(len(summary["plans"]), 1)
        self.assertEqual(summary["plans"][0]["label"], "基线")
        self.assertEqual(summary["plans"][0]["dsr"], 0.9)

    def test_insufficient_pbo_kept_in_summary(self):
        analysis = {
            "mode": "compare",
            "overfitting_audit": {
                "n_trials": 2,
                "pbo": {
                    "status": "insufficient_data",
                    "pbo_probability": None,
                    "cscv_combinations": None,
                    "detail": "需要≥3个方案",
                },
            },
            "plan_results": [],
        }
        summary = _overfitting_summary(analysis)
        self.assertEqual(summary["pbo_status"], "insufficient_data")
        self.assertIsNone(summary["pbo_probability"])


class StatusPageTests(unittest.TestCase):
    def tearDown(self):
        with JOBS_LOCK:
            JOBS.clear()

    def _put_done_compare_job(self, overfitting):
        with JOBS_LOCK:
            JOBS["jobof1"] = {
                "id": "jobof1",
                "kind": "compare",
                "status": "done",
                "steps": ["完成"],
                "report_dir": "x",
                "report_id": "rep1",
                "error": None,
                "score": 60,
                "grade": "结论不稳定",
                "rating": "READY",
                "blocked": False,
                "compare_summary": "3/3个方案完成",
                "overfitting": overfitting,
            }

    def test_status_renders_overfitting_panel(self):
        self._put_done_compare_job(
            {
                "n_trials": 3,
                "pbo_status": "ok",
                "pbo_probability": 0.42,
                "cscv_combinations": 12870,
                "pbo_detail": None,
                "dispersion_validity": "ok",
                "dispersion_detail": "方案两两日收益相关最高0.42（≤0.98），无退化方案，PBO具备参考意义",
                "plans": [{"label": "基线", "dsr": 0.9, "sr_observed": 0.08, "dsr_status": "ok"}],
            }
        )
        html = render_status("jobof1")
        self.assertIn("过拟合审计摘要", html)
        self.assertIn("PBO=0.42", html)
        self.assertIn("12870", html)
        self.assertIn("基线", html)
        self.assertIn("0.9", html)
        self.assertIn("不是对未来收益的预测", html)
        self.assertIn("不构成投资建议", html)
        self.assertIn("PBO具备参考意义", html)

    def test_status_pbo_note_warns_against_single_window_extrapolation(self):
        """PBO跨窗口会大幅波动，状态页备注必须提示不可单窗口外推。"""
        self._put_done_compare_job(
            {
                "n_trials": 3,
                "pbo_status": "ok",
                "pbo_probability": 0.42,
                "cscv_combinations": 12870,
                "pbo_detail": None,
                "dispersion_validity": "ok",
                "dispersion_detail": "相关度可接受",
                "plans": [],
            }
        )
        html = render_status("jobof1")
        self.assertIn("单窗口数字不可外推", html)

    def test_status_shows_degradation_regression_when_present(self):
        """有退化回归数据时状态页显示斜率/截距/组合数。"""
        self._put_done_compare_job(
            {
                "n_trials": 3,
                "pbo_status": "ok",
                "pbo_probability": 0.42,
                "cscv_combinations": 12870,
                "pbo_detail": None,
                "dispersion_validity": "ok",
                "dispersion_detail": "相关度可接受",
                "degradation_slope": 0.35,
                "degradation_intercept": 0.02,
                "degradation_pairs": 12870,
                "plans": [],
            }
        )
        html = render_status("jobof1")
        self.assertIn("性能退化回归", html)
        self.assertIn("0.35", html)
        self.assertIn("12870", html)
        # 没有回归数据时不应出现该小节（full旧报告/数据不足场景）。
        self._put_done_compare_job(
            {
                "n_trials": 3,
                "pbo_status": "ok",
                "pbo_probability": 0.42,
                "cscv_combinations": 12870,
                "pbo_detail": None,
                "dispersion_validity": "ok",
                "dispersion_detail": "相关度可接受",
                "plans": [],
            }
        )
        self.assertNotIn("性能退化回归", render_status("jobof1"))

    def test_status_warns_when_dispersion_insufficient(self):
        self._put_done_compare_job(
            {
                "n_trials": 3,
                "pbo_status": "ok",
                "pbo_probability": 0.95,
                "cscv_combinations": 12870,
                "pbo_detail": None,
                "dispersion_validity": "insufficient_dispersion",
                "dispersion_detail": "方案组缺乏分散度：两两日收益相关最高0.999，且存在退化方案['ms80']；PBO无统计意义，不可作为过拟合证据引用",
                "plans": [],
            }
        )
        html = render_status("jobof1")
        self.assertIn("PBO=0.95", html)
        self.assertIn("PBO无统计意义", html)
        self.assertIn("不可作为过拟合证据引用", html)

    def test_status_shows_insufficient_pbo_honestly(self):
        self._put_done_compare_job(
            {
                "n_trials": 2,
                "pbo_status": "insufficient_data",
                "pbo_probability": None,
                "cscv_combinations": None,
                "pbo_detail": "完成回测的方案仅2个，PBO（CSCV）需要≥3个方案",
                "plans": [],
            }
        )
        html = render_status("jobof1")
        self.assertIn("数据不足", html)
        self.assertIn("需要≥3个方案", html)

    def test_status_without_overfitting_renders_normally(self):
        self._put_done_compare_job(None)
        html = render_status("jobof1")
        self.assertIn("分析完成", html)
        self.assertNotIn("过拟合审计摘要", html)


class WebServerIntegrationTests(unittest.TestCase):
    """走HTTP层的端到端检查：表单能提交带新参数的full任务。"""

    server = None
    thread = None
    base = None

    @classmethod
    def setUpClass(cls):
        import commercial_web

        cls.previous_reports_root = commercial_web.REPORTS_ROOT
        import tempfile

        cls.temp_dir = tempfile.TemporaryDirectory()
        commercial_web.REPORTS_ROOT = cls.temp_dir.name
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        import commercial_web

        commercial_web.REPORTS_ROOT = cls.previous_reports_root
        cls.temp_dir.cleanup()

    def tearDown(self):
        with JOBS_LOCK:
            JOBS.clear()

    def test_run_rejects_bad_strategy_params_over_http(self):
        from urllib.parse import urlencode
        from urllib.request import (
            HTTPError,
            Request,
            urlopen,
        )

        body = urlencode(
            {
                "cache": "a.pkl",
                "start": "2025-01-01",
                "end": "2025-06-30",
                "mode": "full",
                "strategy": "cross_sectional_alpha158",
                "strategy_params": "bad_key=1",
            }
        ).encode()
        request = Request(
            f"{self.base}/run",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            urlopen(request, timeout=10)
            raised = None
        except HTTPError as exc:
            raised = exc
        self.assertIsNotNone(raised)
        self.assertEqual(raised.code, 400)
        page = raised.read().decode("utf-8")
        self.assertIn("参数错误", page)


if __name__ == "__main__":
    unittest.main()
