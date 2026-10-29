"""属性式不变量测试（手写随机循环，不引入hypothesis等外部库）。

在多组随机合成数据上断言两条执行层不变量：
1. T+1成交不变量：买入成交价=执行日开盘价×(1+滑点)，且执行日严格晚于
   信号日——信号日当天的价格变动不可能影响当日成交价（成交发生在次日开盘）；
2. 期末清仓后无持仓残留：合成数据无停牌/跌停时，期末所有持仓必须按
   末日收盘真实卖出，holdings清空、final_unsold为空。
"""

import sys
import unittest

import numpy as np
import pandas as pd

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from commercial.config import CommercialRequest, build_engine_config
from engine.backtest import run as run_classic_backtest


def build_random_data(seed, stocks=10, days=140):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    data = {}
    for index in range(stocks):
        code = f"sh.60{index:04d}" if index % 2 == 0 else f"sz.00{index:04d}"
        closes = 12 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, days)))
        preclose = np.concatenate([[closes[0]], closes[:-1]])
        volume = rng.lognormal(13, 0.3, days).round(0)
        data[code] = pd.DataFrame(
            {
                "date": dates,
                "open": (closes * (1 + rng.normal(0, 0.004, days))).round(3),
                "high": (closes * 1.01).round(3),
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


class ExecutionInvariantsTests(unittest.TestCase):
    def test_t_plus_1_fill_and_no_residual_holdings(self):
        """随机种子循环：T+1成交不变量 + 期末无持仓残留。"""
        request = CommercialRequest(
            cache="unused",
            start="2025-03-01",
            end="2025-06-30",
            quiet=True,
            run_stress=False,
        )
        request.validate()
        config = build_engine_config(request)
        checked_trades_total = 0
        for seed in range(5):
            data = build_random_data(seed)
            portfolio, metrics = run_classic_backtest(
                request.start,
                request.end,
                verbose=False,
                allow_incomplete_data=True,
                config=config,
                data=data,
            )
            self.assertIsNotNone(portfolio)
            # 不变量2：期末清仓后无持仓残留（合成数据全部可交易）。
            self.assertEqual(
                len(portfolio.holdings),
                0,
                f"seed={seed} 期末仍有持仓残留",
            )
            self.assertEqual(
                getattr(portfolio, "final_unsold", {}) or {},
                {},
                f"seed={seed} 期末存在未售出持仓",
            )

            # 不变量1：每次买入成交价=执行日开盘×(1+滑点)，执行日>信号日。
            decisions = {
                (d["execution_date"], trade["code"]): d["signal_date"]
                for d in portfolio.rebalance_decisions
                for trade in d.get("selected", [])
            }
            slippage = float(config.slippage)
            buy_trades = [t for t in portfolio.trades if t["action"] == "BUY"]
            for trade in buy_trades:
                execution_date = pd.Timestamp(trade["date"])
                signal_date = (
                    pd.Timestamp(decisions.get((trade["date"], trade["code"]), ""))
                    if (trade["date"], trade["code"]) in decisions
                    else None
                )
                # 开盘成交价核验（含滑点；允许极小浮点误差）。
                frame = data[trade["code"]]
                opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy()
                dates = pd.DatetimeIndex(frame["date"])
                position = int(dates.searchsorted(execution_date))
                if position < len(opens) and dates[position] == execution_date:
                    expected = opens[position] * (1 + slippage)
                    self.assertAlmostEqual(
                        float(trade["price"]),
                        expected,
                        places=6,
                        msg=f"seed={seed} {trade['code']}@{trade['date']} "
                        "成交价≠执行日开盘价(含滑点)",
                    )
                # T+1：执行日必须严格晚于信号日。
                if signal_date is not None:
                    self.assertGreater(
                        execution_date,
                        signal_date,
                        f"seed={seed} 买入在信号日当日成交，违反T+1",
                    )
                checked_trades_total += 1
        # 保证随机循环确实覆盖到了交易（防止空转式测试）。
        self.assertGreater(checked_trades_total, 0)


if __name__ == "__main__":
    unittest.main()
