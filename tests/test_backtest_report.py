import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.backtest_report import (
    derive_decision_statistics,
    derive_run_statistics,
    write_backtest_report,
)
from strategy.config import StrategyConfig


def equity_frame():
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=25),
            "cash": 10_000.0,
            "holdings_value": [90_000 + index * 500 for index in range(25)],
            "total": [100_000 + index * 500 for index in range(25)],
            "positions": 8,
        }
    )


def trade_frame():
    return pd.DataFrame(
        {
            "action": ["BUY", "SELL"],
            "cost": [10_020.0, None],
            "revenue": [None, 10_500.0],
            "total_friction": [20.0, 25.0],
        }
    )


def decision_rows():
    return [
        {
            "signal_date": "2025-01-03",
            "execution_date": "2025-01-06",
            "market_allowed": True,
            "market_reason": "ok",
            "candidate_count": 12,
            "selected": [
                {"code": "sh.600000", "score": 6.2},
                {"code": "sz.000001", "score": 5.8},
            ],
        },
        {
            "signal_date": "2025-01-10",
            "execution_date": "2025-01-13",
            "market_allowed": False,
            "market_reason": "benchmark_below_ma20",
            "candidate_count": 0,
            "selected": [],
        },
    ]


class BacktestReportTests(unittest.TestCase):
    def test_statistics_include_turnover_and_friction(self):
        stats = derive_run_statistics(equity_frame(), trade_frame())
        self.assertGreater(stats["annualized_turnover"], 0)
        self.assertEqual(stats["total_friction"], 45.0)
        self.assertEqual(stats["maximum_positions"], 8)

    def test_decision_statistics_show_market_blocks_and_score_range(self):
        stats = derive_decision_statistics(decision_rows())
        self.assertEqual(stats["rebalance_count"], 2)
        self.assertEqual(stats["market_blocked_count"], 1)
        self.assertEqual(stats["selected_score_min"], 5.8)
        self.assertEqual(stats["selected_score_median"], 6.0)

    def test_html_and_machine_readable_statistics_are_written(self):
        metrics = {
            "total_return_pct": 12.0,
            "cagr_pct": 10.0,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": -5.0,
            "excess_return_pct": 3.0,
            "win_rate_pct": 45.0,
            "profit_factor": 1.3,
            "total_trades": 1,
        }
        benchmark = equity_frame()[["date", "total"]].rename(columns={"total": "close"})
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_backtest_report(
                temp_dir,
                metrics,
                equity_frame(),
                trade_frame(),
                benchmark,
                StrategyConfig,
                "2025-01-02",
                "2025-02-05",
                decisions=decision_rows(),
            )
            content = path.read_text(encoding="utf-8")
            stats = json.loads(
                (Path(temp_dir) / "reports" / "run_statistics.json").read_text(encoding="utf-8")
            )
            decision_stats = json.loads(
                (Path(temp_dir) / "reports" / "decision_statistics.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertIn("Alice 单次回测报告", content)
        self.assertIn("净值与回撤", content)
        self.assertIn("调仓决策审计", content)
        self.assertEqual(stats["total_friction"], 45.0)
        self.assertEqual(decision_stats["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
