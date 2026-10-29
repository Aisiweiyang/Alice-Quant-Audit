"""收益质量指标测试（pyfolio/quantstats/empyrical口径的alpha/beta/IR/
tail_ratio/up-down capture/recovery_factor + pyfolio口径的前5大回撤明细）。"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from engine.backtest import _top_drawdown_periods, calculate_metrics
from tests.test_oss_practices import fake_portfolio

QUALITY_KEYS = (
    "alpha_annualized_pct",
    "beta",
    "information_ratio",
    "tail_ratio",
    "up_capture_pct",
    "down_capture_pct",
    "recovery_factor",
)


def build_pair(strat_returns, bench_returns, initial=100000.0, start="2025-01-01"):
    """由日收益序列构造净值与基准帧（日历完全一致，便于确定性断言）。

    首日为起点（净值=initial、基准=1000），其后逐日按收益复利推进；
    因此策略日收益与基准日收益逐日一一对应。返回(portfolio, benchmark_df, totals)。
    """
    dates = pd.bdate_range(start, periods=len(strat_returns) + 1)
    totals = initial * np.concatenate([[1.0], np.cumprod(1 + np.asarray(strat_returns))])
    benchmark_df = pd.DataFrame(
        {
            "date": dates,
            "close": (
                1000 * np.concatenate([[1.0], np.cumprod(1 + np.asarray(bench_returns))])
            ).round(4),
        }
    )
    return fake_portfolio(totals, dates), benchmark_df, totals


class AlphaBetaCaptureTests(unittest.TestCase):
    def setUp(self):
        # 固定基准日收益（含明确的上/下行日），策略=0.5×基准+0.001，无噪声：
        # 回归斜率恰为0.5、截距恰为0.001、超额恒为0.001（标准差0→IR如实None）。
        rng = np.random.default_rng(11)
        self.bench_returns = rng.normal(0.0005, 0.008, 120)
        self.strat_returns = 0.5 * self.bench_returns + 0.001
        portfolio, benchmark_df, self.totals = build_pair(self.strat_returns, self.bench_returns)
        self.metrics = calculate_metrics(portfolio, benchmark_df)

    def test_beta_exact_and_alpha_annualized(self):
        self.assertAlmostEqual(self.metrics["beta"], 0.5, places=3)
        expected_alpha = ((1.001**252) - 1) * 100
        self.assertAlmostEqual(
            self.metrics["alpha_annualized_pct"], round(expected_alpha, 2), places=1
        )

    def test_information_ratio_matches_definition(self):
        """IR=超额(策略-基准)日收益均值/标准差×√252；本设定下超额=0.001-0.5×基准，
        可精确复算。"""
        excess = self.strat_returns - self.bench_returns
        expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(
            self.metrics["information_ratio"], round(float(expected), 3), places=2
        )

    def test_up_down_capture_values(self):
        bench = self.bench_returns
        strat = self.strat_returns
        up = bench > 0
        down = bench < 0
        expected_up = strat[up].mean() / bench[up].mean() * 100
        expected_down = strat[down].mean() / bench[down].mean() * 100
        self.assertAlmostEqual(self.metrics["up_capture_pct"], round(expected_up, 2), places=1)
        self.assertAlmostEqual(self.metrics["down_capture_pct"], round(expected_down, 2), places=1)

    def test_tail_ratio_matches_percentile_definition(self):
        p95 = np.percentile(self.strat_returns, 95)
        p05 = np.percentile(self.strat_returns, 5)
        self.assertAlmostEqual(self.metrics["tail_ratio"], round(abs(p95) / abs(p05), 3), places=3)

    def test_recovery_factor_consistent_with_return_and_drawdown(self):
        total = (self.totals[-1] / self.totals[0] - 1) * 100
        peak = np.maximum.accumulate(self.totals)
        max_dd = ((self.totals - peak) / peak).min() * 100
        expected = round((total / 100) / abs(max_dd / 100), 3)
        self.assertAlmostEqual(self.metrics["recovery_factor"], expected, places=3)

    def test_drawdown_deepest_equals_max_drawdown(self):
        deepest = min(p["depth_pct"] for p in self.metrics["drawdown_periods"])
        self.assertEqual(deepest, self.metrics["max_drawdown_pct"])


class InformationRatioTests(unittest.TestCase):
    def test_flat_benchmark_degenerate_fields_none(self):
        """基准完全走平（方差0）时回归类指标如实None：beta/alpha不可定义，
        无上涨日→上行捕获None；信息比率退化为策略自身夏普口径仍可计算。"""
        rng = np.random.default_rng(19)
        strat_returns = rng.normal(0.0006, 0.008, 130)
        dates = pd.bdate_range("2025-01-01", periods=len(strat_returns) + 1)
        totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + strat_returns)])
        benchmark_df = pd.DataFrame(
            {
                "date": dates,
                "close": 1000.0,
            }
        )
        metrics = calculate_metrics(fake_portfolio(totals, dates), benchmark_df)
        self.assertIsNone(metrics["beta"])
        self.assertIsNone(metrics["alpha_annualized_pct"])
        self.assertIsNone(metrics["up_capture_pct"])
        self.assertIsNone(metrics["down_capture_pct"])
        expected_ir = strat_returns.mean() / strat_returns.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(metrics["information_ratio"], round(float(expected_ir), 3), places=2)

    def test_ir_with_noisy_excess(self):
        """有噪声超额时，IR=超额均值/超额标准差×√252（quantstats口径）。"""
        rng = np.random.default_rng(23)
        bench_returns = rng.normal(0.0002, 0.009, 150)
        noise = rng.normal(0.0, 0.003, 150)
        strat_returns = 0.8 * bench_returns + 0.0006 + noise
        portfolio, benchmark_df, _totals = build_pair(strat_returns, bench_returns)
        metrics = calculate_metrics(portfolio, benchmark_df)
        excess = strat_returns - bench_returns
        expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
        self.assertIsNotNone(metrics["information_ratio"])
        self.assertAlmostEqual(metrics["information_ratio"], round(float(expected), 3), places=2)


class MissingBenchmarkTests(unittest.TestCase):
    def test_all_quality_fields_none_without_benchmark(self):
        rng = np.random.default_rng(31)
        strat_returns = rng.normal(0.0008, 0.01, 120)
        totals = 100000 * np.cumprod(1 + strat_returns)
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        metrics = calculate_metrics(fake_portfolio(totals, dates))
        for key in QUALITY_KEYS:
            self.assertIsNone(metrics[key])
        # 回撤明细不依赖基准，仍然输出。
        self.assertTrue(metrics["drawdown_periods"])

    def test_too_few_aligned_observations_none(self):
        rng = np.random.default_rng(37)
        strat_returns = rng.normal(0.0008, 0.01, 120)
        # 基准只有9个交易日（8个重叠收益日，<20门槛），回归不可靠，如实None。
        dates = pd.bdate_range("2025-01-01", periods=len(strat_returns) + 1)
        totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + strat_returns)])
        benchmark_df = pd.DataFrame(
            {
                "date": dates[:9],
                "close": (
                    1000 * np.concatenate([[1.0], np.cumprod(1 + rng.normal(0, 0.008, 8))])
                ).round(4),
            }
        )
        portfolio = fake_portfolio(totals, dates)
        metrics = calculate_metrics(portfolio, benchmark_df)
        for key in QUALITY_KEYS:
            self.assertIsNone(metrics[key], key)


class DrawdownPeriodTests(unittest.TestCase):
    def test_handcrafted_path_top_periods(self):
        # 手工推导路径：两段完整回撤 + 一段未恢复。
        totals = np.array([100, 110, 90, 95, 110, 120, 100, 119, 120.5])
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        periods = _top_drawdown_periods(totals, dates)
        self.assertEqual(len(periods), 2)
        first, second = periods
        self.assertEqual(first["peak_date"], dates[1].strftime("%Y-%m-%d"))
        self.assertEqual(first["valley_date"], dates[2].strftime("%Y-%m-%d"))
        self.assertEqual(first["recovery_date"], dates[4].strftime("%Y-%m-%d"))
        self.assertEqual(first["depth_pct"], -18.18)
        self.assertEqual(first["decline_days"], 1)
        self.assertEqual(first["recovery_days"], 2)
        self.assertEqual(second["peak_date"], dates[5].strftime("%Y-%m-%d"))
        self.assertEqual(second["valley_date"], dates[6].strftime("%Y-%m-%d"))
        self.assertEqual(second["depth_pct"], round((100 / 120 - 1) * 100, 2))
        # 深度从深到浅排序。
        self.assertLessEqual(first["depth_pct"], second["depth_pct"])

    def test_open_drawdown_marked_unrecovered(self):
        totals = np.array([100, 120, 90, 95])
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        periods = _top_drawdown_periods(totals, dates)
        self.assertEqual(len(periods), 1)
        self.assertIsNone(periods[0]["recovery_date"])
        self.assertIsNone(periods[0]["recovery_days"])
        self.assertEqual(periods[0]["decline_days"], 1)

    def test_top5_limit(self):
        rng = np.random.default_rng(41)
        totals = 100000 * np.cumprod(1 + rng.normal(0, 0.01, 300))
        dates = pd.bdate_range("2025-01-01", periods=len(totals))
        periods = _top_drawdown_periods(totals, dates)
        self.assertLessEqual(len(periods), 5)


class ReportIntegrationTests(unittest.TestCase):
    def test_report_contains_new_sections_and_metrics_json_machine_readable(self):
        from commercial.config import CommercialRequest
        from commercial.pipeline import run_full_analysis

        rng = np.random.default_rng(21)
        days = 150
        dates = pd.bdate_range("2025-01-01", periods=days)
        data = {}
        for index in range(8):
            code = f"sh.60{index:04d}" if index % 2 == 0 else f"sz.00{index:04d}"
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
        bench_closes = 1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, days)))
        data["sh.000300"] = pd.DataFrame(
            {
                "date": dates,
                "close": bench_closes.round(2),
                "open": 1000.0,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            from data.cache_store import build_metadata, save_market_cache

            cache = os.path.join(temp_dir, "market.pkl")
            save_market_cache(
                cache,
                data,
                build_metadata(
                    data,
                    benchmark="sh.000300",
                    adjust="2",
                    universe_mode="file",
                    source="unittest",
                ),
            )
            request = CommercialRequest(
                cache=cache,
                start="2025-05-01",
                end="2025-07-15",
                output_root=temp_dir,
                quiet=True,
                run_stress=False,
            )
            request.validate()
            analysis = run_full_analysis(request, report_id="quality_check")
            root = analysis["report_dir"]
            with open(os.path.join(root, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("最大回撤明细", html)
            self.assertIn("信息比率", html)
            self.assertIn("Alpha(年化)", html)
            self.assertIn("上行/下行捕获", html)
            self.assertIn("尾部比率", html)
            self.assertIn("回收因子", html)
            self.assertIn("仅供研究", html)
            with open(os.path.join(root, "metrics.json"), encoding="utf-8") as handle:
                import json

                payload = json.load(handle)
            for key in QUALITY_KEYS + ("drawdown_periods",):
                self.assertIn(key, payload)
            if payload.get("drawdown_periods"):
                self.assertEqual(
                    min(p["depth_pct"] for p in payload["drawdown_periods"]),
                    payload["max_drawdown_pct"],
                )


if __name__ == "__main__":
    unittest.main()
