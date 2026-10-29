"""商业化MVP优化轮测试：多窗口滚动样本外、报告历史、本地网页入口。"""

import json
import os
import sys
import tempfile
import threading
import types
import unittest
import urllib.request
from unittest.mock import patch

import numpy as np
import pandas as pd
from http.server import ThreadingHTTPServer

sys.modules.setdefault("baostock", types.ModuleType("baostock"))

from commercial.config import CommercialRequest
from commercial.reliability_score import compute_reliability_score
from commercial.rolling_oos import (
    MAX_WINDOWS,
    rolling_frame,
    run_rolling_oos,
    split_windows,
)
from commercial_web import (
    AuditHandler,
    parse_form,
    request_from_form,
)
from data.cache_store import build_metadata, save_market_cache


def make_data(stocks=12, days=200, start="2025-01-01"):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range(start, periods=days)
    data = {}
    for index in range(stocks):
        code = f"sh.60{index:04d}" if index % 2 == 0 else f"sz.00{index:04d}"
        steps = rng.normal(0.0003, 0.012, size=days)
        closes = 15 * np.exp(np.cumsum(steps))
        preclose = np.concatenate([[closes[0]], closes[:-1]])
        volume = rng.lognormal(13, 0.3, size=days).round(0)
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
            "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=days)))).round(2),
            "open": 1000.0,
        }
    )
    return data


def write_cache(data, directory, name="market.pkl"):
    metadata = build_metadata(
        data,
        benchmark="sh.000300",
        adjust="2",
        universe_mode="file",
        source="unittest",
    )
    path = os.path.join(directory, name)
    save_market_cache(path, data, metadata)
    return path


def make_request(cache, output_root, **overrides):
    defaults = dict(
        cache=cache,
        start="2025-05-01",
        end="2025-09-30",
        output_root=output_root,
        quiet=True,
    )
    defaults.update(overrides)
    request = CommercialRequest(**defaults)
    request.validate()
    return request


class SplitWindowTests(unittest.TestCase):
    def test_windows_are_contiguous_and_labelled(self):
        windows = split_windows("2025-01-01", "2025-12-31", 3)
        self.assertEqual(len(windows), 3)
        self.assertEqual([w[0] for w in windows], ["window_1", "window_2", "window_3"])
        self.assertEqual(windows[0][1], "2025-01-01")
        self.assertEqual(windows[-1][2], "2025-12-31")
        # 连续性：上一窗口结束的次日是下一窗口开始
        for (_, _, prev_end), (_, next_start, _) in zip(windows, windows[1:]):
            self.assertEqual(
                pd.Timestamp(next_start), pd.Timestamp(prev_end) + pd.Timedelta(days=1)
            )

    def test_invalid_inputs_return_empty(self):
        self.assertEqual(split_windows("2025-06-01", "2025-06-01", 3), [])
        self.assertEqual(split_windows("2025-01-01", "2025-12-31", 1), [])
        self.assertLessEqual(MAX_WINDOWS, 4)


def _fake_engine(metrics):
    def fake_run(start, end, **kwargs):
        return object(), metrics

    return fake_run


class RollingOosTests(unittest.TestCase):
    def _request(self, temp_dir):
        return make_request("x.pkl", temp_dir)

    def test_stable_verdict_when_all_windows_positive(self):
        metrics = {
            "total_return_pct": 5.0,
            "sharpe_ratio": 1.1,
            "sharpe_reliable": True,
            "max_drawdown_pct": -6.0,
            "total_trades": 10,
        }
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("commercial.rolling_oos.run_classic_backtest", _fake_engine(metrics)),
        ):
            result = run_rolling_oos(self._request(temp_dir), object(), "x.pkl", 3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verdict"], "stable")
        self.assertEqual(len(result["windows"]), 3)

    def test_unstable_verdict_when_most_windows_negative(self):
        metrics = {
            "total_return_pct": -5.0,
            "sharpe_ratio": -0.8,
            "sharpe_reliable": True,
            "max_drawdown_pct": -12.0,
            "total_trades": 8,
        }
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("commercial.rolling_oos.run_classic_backtest", _fake_engine(metrics)),
        ):
            result = run_rolling_oos(self._request(temp_dir), object(), "x.pkl", 3)
        self.assertEqual(result["verdict"], "unstable")

    def test_insufficient_when_window_has_no_data(self):
        metrics = {
            "total_return_pct": 5.0,
            "sharpe_ratio": 1.0,
            "sharpe_reliable": True,
            "max_drawdown_pct": -5.0,
            "total_trades": 9,
        }

        def fake_run(start, end, **kwargs):
            if "window" in str(kwargs):
                pass
            if pd.Timestamp(start) >= pd.Timestamp("2025-08-01"):
                return None, None
            return object(), metrics

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("commercial.rolling_oos.run_classic_backtest", fake_run),
        ):
            result = run_rolling_oos(self._request(temp_dir), object(), "x.pkl", 3)
        self.assertEqual(result["status"], "insufficient_data")

    def test_integration_on_synthetic_cache_is_honest(self):
        data = make_data(stocks=12, days=200)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir, start="2025-03-01", end="2025-09-30")
            config = type("Cfg", (), {})()  # 滚动测试中引擎config会被使用
            from commercial.config import build_engine_config

            # 商业审计已做门禁，放行引擎的metadata质量检查（与管线行为一致）
            request.allow_incomplete_data = True
            result = run_rolling_oos(request, build_engine_config(request), cache, 2)
        self.assertIn(result["status"], {"ok", "insufficient_data"})
        self.assertTrue(result.get("windows"))

    def test_rolling_frame_serialisation(self):
        rolling = {
            "status": "ok",
            "verdict": "stable",
            "windows": [
                {
                    "window": "window_1",
                    "start": "2025-01-01",
                    "end": "2025-04-30",
                    "status": "ok",
                    "total_return_pct": 3.0,
                    "max_drawdown_pct": -4.0,
                    "sharpe_ratio": 1.0,
                    "sharpe_reliable": True,
                    "total_trades": 9,
                },
                {
                    "window": "window_2",
                    "start": "2025-05-01",
                    "end": "2025-08-31",
                    "status": "insufficient_data",
                },
            ],
        }
        frame = rolling_frame(rolling)
        self.assertEqual(len(frame), 2)
        self.assertIn("total_return_pct", frame.columns)


class RollingScoreIntegrationTests(unittest.TestCase):
    def _audit(self):
        return {
            "rating": "READY",
            "blockers": [],
            "warnings": [],
            "coverage": {"median_session_coverage": 1.0},
            "fields": {"historical_trade_status": True},
        }

    def test_rolling_stable_earns_full_points(self):
        rolling = {
            "status": "ok",
            "verdict": "stable",
            "positive_share": 1.0,
            "sharpe_std": 0.2,
            "windows": [1, 2, 3],
            "detail": "稳定",
            "note": "",
        }
        score = compute_reliability_score(
            data_audit=self._audit(),
            base_result=None,
            segment_consistency={"status": "insufficient_data"},
            stress_summary={"status": "insufficient_data"},
            rolling_oos=rolling,
        )
        oos = next(d for d in score["dimensions"] if d["key"] == "out_of_sample")
        self.assertEqual(oos["status"], "pass")
        self.assertEqual(oos["earned_points"], 20)

    def test_rolling_unstable_fails_dimension(self):
        rolling = {
            "status": "ok",
            "verdict": "unstable",
            "positive_share": 0.0,
            "sharpe_std": 1.5,
            "windows": [1, 2, 3],
            "detail": "不稳定",
            "note": "",
        }
        score = compute_reliability_score(
            data_audit=self._audit(),
            base_result=None,
            segment_consistency={"status": "insufficient_data"},
            stress_summary={"status": "insufficient_data"},
            rolling_oos=rolling,
        )
        oos = next(d for d in score["dimensions"] if d["key"] == "out_of_sample")
        self.assertEqual(oos["status"], "fail")
        self.assertLessEqual(oos["earned_points"], 5)


class FormParsingTests(unittest.TestCase):
    def test_parse_form_and_request(self):
        form = parse_form(
            b"cache=a.pkl&start=2025-01-01&end=2025-06-30&initial_capital=200000"
            b"&max_positions=5&rebalance_freq=monthly&no_stress=1"
        )
        request = request_from_form(form)
        self.assertEqual(request.initial_capital, 200_000)
        self.assertEqual(request.max_positions, 5)
        self.assertFalse(request.run_stress)
        self.assertTrue(request.run_factor_diagnostics)

    def test_invalid_form_raises_parameter_error(self):
        form = parse_form(b"cache=a.pkl&start=bad-date&end=2025-06-30")
        with self.assertRaises(Exception):
            request_from_form(form)


class WebServerTests(unittest.TestCase):
    server = None
    thread = None
    base = None
    temp_dir = None
    previous_reports_root = None

    @classmethod
    def setUpClass(cls):
        import commercial_web

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.previous_reports_root = commercial_web.REPORTS_ROOT
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

    def test_home_and_reports_pages_render(self):
        with urllib.request.urlopen(f"{self.base}/", timeout=10) as response:
            home = response.read().decode("utf-8")
        self.assertIn("新建分析", home)
        self.assertIn("不构成投资建议", home)
        with urllib.request.urlopen(f"{self.base}/reports", timeout=10) as response:
            reports = response.read().decode("utf-8")
        self.assertIn("历史报告", reports)

    def test_bad_parameters_return_400(self):
        data = "cache=a.pkl&start=not-a-date&end=2025-06-30".encode()
        request = urllib.request.Request(f"{self.base}/run", data=data, method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)
            self.fail("应当返回400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_full_job_lifecycle(self):
        data = make_data(stocks=12, days=200)
        cache = write_cache(data, self.temp_dir.name)
        form = (
            f"cache={cache}&start=2025-05-01&end=2025-09-30&mode=full"
            "&no_stress=1&no_factor=1&no_rolling=1"
        )
        request = urllib.request.Request(f"{self.base}/run", data=form.encode(), method="POST")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(request, timeout=10)
            self.fail("应当返回303")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 303)
            location = exc.headers.get("Location", "")
        self.assertTrue(location.startswith("/status?job="))
        job_id = location.split("job=")[1]

        import time

        deadline = time.time() + 120
        job = None
        while time.time() < deadline:
            with urllib.request.urlopen(
                f"{self.base}/api/status?job={job_id}", timeout=10
            ) as response:
                job = json.loads(response.read().decode("utf-8"))
            if job.get("status") in {"done", "error"}:
                break
            time.sleep(1.0)
        self.assertEqual(job.get("status"), "done", f"任务未完成: {job.get('error')}")
        self.assertIsNotNone(job.get("report_id"))
        self.assertEqual(job.get("rating"), "READY")
        self.assertTrue(job.get("steps"))


if __name__ == "__main__":
    unittest.main()
