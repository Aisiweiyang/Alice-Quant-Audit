"""交易级蒙特卡洛(tmc_*)、Omega比率与reports --list过拟合审计标记的测试。

口径出处：
- Omega：empyrical stats.omega_ratio（2026-09-12在线核对GitHub源码）；
- tmc：本引擎自有口径（回合收益随机重排+复利，固定种子）；
- 过拟合审计标记：metrics.json中overfitting_audit(compare)/deflated_sharpe(full)存在性。
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from commercial.report import render_html
from commercial_mvp import collect_reports
from engine.backtest import _omega_ratio, _tmc_drawdown_stats, calculate_metrics
from tests.test_oss_practices import fake_portfolio
from tests.test_risk_metrics import fake_trades


class OmegaRatioTests(unittest.TestCase):
    def test_known_answer_gains_heavier(self):
        # 上方收益和0.03 / 下方损失绝对值0.01 = 3.0
        self.assertAlmostEqual(_omega_ratio([0.01, 0.02, -0.01]), 3.0)

    def test_known_answer_losses_heavier(self):
        self.assertAlmostEqual(_omega_ratio([0.01, -0.01, -0.02]), 1.0 / 3.0)

    def test_no_losses_returns_none(self):
        self.assertIsNone(_omega_ratio([0.01, 0.02, 0.005]))

    def test_nan_inputs_return_none(self):
        self.assertIsNone(_omega_ratio([0.01, np.nan, -0.01]))

    def test_threshold_shifts_split(self):
        # 阈值0.01：超额为[0.01, -0.005, -0.02] → 0.01/(0.005+0.02)=0.4
        self.assertAlmostEqual(_omega_ratio([0.02, 0.005, -0.01], threshold=0.01), 0.4)

    def test_metrics_gate_at_20_observations(self):
        # calculate_metrics里样本<20时omega_ratio必须为None，≥20时有值。
        rng = np.random.default_rng(11)
        for days, expect_none in ((25, False), (19, True)):
            returns = rng.normal(0.0005, 0.01, days)
            totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + returns)])
            dates = pd.bdate_range("2025-01-01", periods=days + 1)
            portfolio = fake_portfolio(totals, dates)
            portfolio.trades = []
            metrics = calculate_metrics(portfolio)
            if expect_none:
                self.assertIsNone(metrics["omega_ratio"])
            else:
                self.assertIsNotNone(metrics["omega_ratio"])


class TradeMonteCarloTests(unittest.TestCase):
    """tmc口径（2026-09-13真实数据审读后修正）：重排每笔净盈亏（金额），
    加法不变性保证终值恒等于已实现总收益；此前的"收益率全仓复利"口径
    会丢掉仓位管理与现金拖累（203笔实测终值+1628% vs 实际+45%），已废弃。
    """

    @staticmethod
    def _trips(pnls):
        return [{"pnl": pnl} for pnl in pnls]

    def test_below_30_round_trips_is_none(self):
        self.assertIsNone(_tmc_drawdown_stats(self._trips([10.0] * 29), 100_000.0))
        self.assertIsNone(_tmc_drawdown_stats([], 100_000.0))
        # pnl缺失的回合应被过滤后再计数
        self.assertIsNone(
            _tmc_drawdown_stats(self._trips([10.0] * 29) + [{"pnl": None}] * 5, 100_000.0)
        )
        self.assertIsNone(_tmc_drawdown_stats(self._trips([10.0] * 40), 0.0))
        self.assertIsNone(_tmc_drawdown_stats(self._trips([10.0] * 40), None))

    def test_terminal_equals_realized_return(self):
        # 加法不变性：任意重排终值相同，p5就是已实现总收益。
        pnls = list(np.round(np.random.default_rng(3).normal(120, 400, 40), 2))
        stats = _tmc_drawdown_stats(self._trips(pnls), 100_000.0)
        self.assertIsNotNone(stats)
        realized = sum(pnls) / 100_000.0 * 100
        self.assertAlmostEqual(stats["tmc_terminal_return_p5_pct"], round(realized, 2), places=1)

    def test_all_positive_pnl_has_zero_drawdown(self):
        stats = _tmc_drawdown_stats(self._trips([50.0] * 40), 1_000.0)
        self.assertEqual(stats["tmc_max_drawdown_pct_p50"], 0.0)
        self.assertEqual(stats["tmc_max_drawdown_pct_p95"], 0.0)
        self.assertAlmostEqual(stats["tmc_terminal_return_p5_pct"], 200.0, places=6)

    def test_bankrupt_path_counts_as_full_loss(self):
        # 全亏在前、盈利在后的重排会中途穿零：该路径回撤按-100%计，
        # 终值仍由加法不变性保持；p50/p95落在[-100, 0]内且可复现。
        pnls = [-200.0] * 10 + [300.0] * 30
        stats = _tmc_drawdown_stats(self._trips(pnls), 2_000.0)
        self.assertIsNotNone(stats)
        self.assertLessEqual(stats["tmc_max_drawdown_pct_p95"], 0.0)
        self.assertGreaterEqual(stats["tmc_max_drawdown_pct_p50"], -100.0)

    def test_total_loss_returns_none(self):
        # 盈亏合计亏光本金：终值分位数无意义，如实None。
        self.assertIsNone(_tmc_drawdown_stats(self._trips([-100.0] * 30), 2_000.0))

    def test_deterministic_with_fixed_seed(self):
        pnls = list(np.round(np.random.default_rng(5).normal(30, 150, 50), 2))
        first = _tmc_drawdown_stats(self._trips(pnls), 100_000.0)
        second = _tmc_drawdown_stats(self._trips(pnls), 100_000.0)
        self.assertEqual(first, second)

    def test_metrics_wire_tmc_keys(self):
        rng = np.random.default_rng(7)
        days = 120
        returns = rng.normal(0.0005, 0.01, days)
        totals = 100000 * np.concatenate([[1.0], np.cumprod(1 + returns)])
        dates = pd.bdate_range("2025-01-01", periods=days + 1)
        portfolio = fake_portfolio(totals, dates)
        trades = []
        for index in range(32):
            day = str(dates[(index % (days - 2)) + 1].date())
            trades.append((day, f"sh.6000{index % 10}", "BUY", 10.0, 100, None, 1000.0))
            trades.append(
                (day, f"sh.6000{index % 10}", "SELL", 10.5, 100, round(rng.normal(20, 60), 2), None)
            )
        portfolio.trades = fake_trades(trades)
        metrics = calculate_metrics(portfolio)
        for key in (
            "tmc_max_drawdown_pct_p50",
            "tmc_max_drawdown_pct_p95",
            "tmc_terminal_return_p5_pct",
            "tmc_round_trips",
            "tmc_paths",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["tmc_round_trips"], 32)
        self.assertEqual(metrics["tmc_paths"], 1000)


class ReportRenderingTests(unittest.TestCase):
    def _analysis(self, metrics_extra):
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
            "mc_max_drawdown_pct_p50": -9.0,
            "mc_max_drawdown_pct_p95": -18.0,
            "mc_terminal_return_p5_pct": -5.0,
            "mc_paths": 1000,
            "tmc_max_drawdown_pct_p50": -6.0,
            "tmc_max_drawdown_pct_p95": -14.0,
            "tmc_terminal_return_p5_pct": 5.0,
            "tmc_round_trips": 40,
            "tmc_paths": 1000,
            "omega_ratio": 1.15,
        }
        metrics.update(metrics_extra)
        request = CommercialRequest(
            cache="x.pkl",
            start="2025-01-01",
            end="2025-06-30",
        )
        return {
            "report_id": "test_report",
            "request": request,
            "score": {"dimensions": [], "score": 60, "grade_label": "结论不稳定"},
            "data_audit": {"rating": "READY", "warnings": [], "blockers": []},
            "base": {"metrics": metrics, "statistics": {}},
            "charts_written": [],
        }

    def test_html_shows_both_mc_flavours_and_omega(self):
        html = render_html(self._analysis({}))
        self.assertIn("蒙特卡洛重构：回撤与终值的分布", html)
        self.assertIn("交易级蒙特卡洛重排", html)
        self.assertIn("加法不变性", html)
        self.assertIn("Omega比率", html)

    def test_html_omits_tmc_section_when_absent(self):
        # 旧报告没有tmc_*键时不应渲染交易MC小节，也不能报错。
        html = render_html(
            self._analysis(
                {
                    "tmc_max_drawdown_pct_p50": None,
                    "tmc_max_drawdown_pct_p95": None,
                    "tmc_terminal_return_p5_pct": None,
                    "tmc_round_trips": None,
                    "tmc_paths": None,
                }
            )
        )
        self.assertNotIn("交易级蒙特卡洛重排", html)
        self.assertIn("蒙特卡洛重构：回撤与终值的分布", html)


class OverfittingFlagTests(unittest.TestCase):
    def _write_report(self, root, name, metrics_payload):
        report_dir = os.path.join(root, name)
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"created_at": "2026-09-12T00:00:00+08:00", "blocked": False, "parameters": {}},
                handle,
            )
        if metrics_payload is not None:
            with open(os.path.join(report_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle)

    def test_flag_from_overfitting_audit_or_dsr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_report(temp_dir, "cmp_report", {"mode": "compare", "overfitting_audit": {}})
            self._write_report(
                temp_dir, "full_report", {"sharpe_ratio": 1.0, "deflated_sharpe": 0.9}
            )
            self._write_report(temp_dir, "plain_report", {"sharpe_ratio": 1.0})
            self._write_report(temp_dir, "no_metrics", None)
            rows = {r["report_id"]: r for r in collect_reports(temp_dir)}
            self.assertTrue(rows["cmp_report"]["has_overfitting_audit"])
            self.assertTrue(rows["full_report"]["has_overfitting_audit"])
            self.assertFalse(rows["plain_report"]["has_overfitting_audit"])
            self.assertFalse(rows["no_metrics"]["has_overfitting_audit"])


if __name__ == "__main__":
    unittest.main()
