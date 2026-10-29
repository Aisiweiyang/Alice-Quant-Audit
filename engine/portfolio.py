"""
持仓管理模块 —— 管理资金、持仓、交易记录
"""

import pandas as pd

from engine.execution import closing_sale_is_executable


class Portfolio:
    """投资组合管理器"""

    def __init__(self, initial_cash, broker, max_positions=10, max_position_pct=0.10):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.broker = broker
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct

        # 持仓: {code: {'shares': int, 'avg_cost': float}}
        self.holdings = {}
        # 交易记录
        self.trades = []
        # 每日资金记录
        self.equity_curve = []
        # 调仓决策审计：记录市场门禁、候选分数和实际买入，不参与交易逻辑。
        self.rebalance_decisions = []
        # 最后已知收盘价：持仓当日缺价时按最后已知价估值，而不是按零。
        self.last_known_prices = {}
        # 最近一次估值中按最后已知价/成本估值的持仓，用于报告披露。
        self.unpriced_positions = {}
        # 期末未能真实卖出的持仓及其原因，由 close_all 填充。
        self.final_unsold = {}

    def total_value(self, prices=None):
        """当前总资产 = 现金 + 持仓市值

        当日没有价格的持仓按最后已知收盘价估值；从未有过价格的持仓按成本估值。
        这两类持仓都会记录到 unpriced_positions，供报告披露，而不是静默计零。
        ``prices`` 可以是 dict 或按股票代码索引的 Series。
        """
        if prices is not None and len(prices):
            for code, price in dict(prices).items():
                if price is not None and pd.notna(price) and float(price) > 0:
                    self.last_known_prices[str(code)] = float(price)
        self.unpriced_positions = {}
        mv = self.cash
        for code, h in self.holdings.items():
            price = self.last_known_prices.get(code)
            if price:
                mv += h["shares"] * price
            else:
                mv += h["shares"] * h["avg_cost"]
                self.unpriced_positions[code] = "no_price_using_avg_cost"
        return mv

    def can_buy(self):
        """是否还有仓位空间"""
        return len(self.holdings) < self.max_positions

    def buy(self, code, price, date, portfolio_value=None):
        """买入股票"""
        if code in self.holdings or not self.can_buy():
            return False
        if portfolio_value is None:
            portfolio_value = self.total_value()
        budget = min(self.cash, portfolio_value * self.max_position_pct)
        shares = self.broker.buy_shares(price, budget, 1.0, date)
        if shares == 0:
            return False
        actual_price, cost = self.broker.buy(price, shares, date)
        if cost > self.cash:
            return False
        self.cash -= cost
        self.holdings[code] = {
            "shares": shares,
            "avg_cost": actual_price,
            "cost_basis": cost,
            "buy_date": date,
        }
        fee_detail = self.broker.last_cost.to_dict()
        self.trades.append(
            {
                "date": date,
                "code": code,
                "action": "BUY",
                "price": actual_price,
                "shares": shares,
                "cost": cost,
                "commission": fee_detail["commission"],
                "stamp_tax": fee_detail["stamp_tax"],
                "slippage_cost": fee_detail["slippage_cost"],
                "total_friction": fee_detail["total_friction"],
            }
        )
        return True

    def sell(self, code, price, date):
        """卖出股票"""
        if code not in self.holdings:
            return False
        h = self.holdings[code]
        actual_price, revenue = self.broker.sell(price, h["shares"], date)
        self.cash += revenue
        # 成本基数包含买入佣金，避免高估每笔交易收益。
        cost_basis = h.get("cost_basis", h["shares"] * h["avg_cost"])
        pnl = revenue - cost_basis
        fee_detail = self.broker.last_cost.to_dict()
        self.trades.append(
            {
                "date": date,
                "code": code,
                "action": "SELL",
                "price": actual_price,
                "shares": h["shares"],
                "revenue": revenue,
                "pnl": pnl,
                "commission": fee_detail["commission"],
                "stamp_tax": fee_detail["stamp_tax"],
                "slippage_cost": fee_detail["slippage_cost"],
                "total_friction": fee_detail["total_friction"],
            }
        )
        del self.holdings[code]
        return True

    def record(self, date, prices):
        """记录当日净值"""
        total = self.total_value(prices)
        self.equity_curve.append(
            {
                "date": date,
                "cash": self.cash,
                "holdings_value": total - self.cash,
                "total": total,
                "positions": len(self.holdings),
                "unpriced_positions": len(self.unpriced_positions),
            }
        )

    def close_all(self, date, prices, rows=None):
        """清仓（回测结束时）。

        提供 ``rows``（code -> 当日行情行）时，只有停牌与跌停检查通过的持仓才按
        收盘价卖出；卖不出的持仓保留在 ``final_unsold`` 中按最后已知价估值，
        不再假设"回测结束就能成交"。
        """
        self.final_unsold = {}
        for code in list(self.holdings.keys()):
            if code not in prices:
                self.final_unsold[code] = "no_price_on_final_date"
                continue
            if rows is not None:
                row = rows.get(code)
                if row is None:
                    self.final_unsold[code] = "no_price_on_final_date"
                    continue
                executable, reason = closing_sale_is_executable(code, row)
                if not executable:
                    self.final_unsold[code] = reason
                    continue
            self.sell(code, prices[code], date)

    def summary(self):
        """交易汇总"""
        if not self.trades:
            return None
        df = pd.DataFrame(self.trades)
        sells = df[df["action"] == "SELL"]
        if sells.empty:
            return {"total_trades": 0, "win_rate": 0, "avg_return": 0}
        wins = sells[sells["pnl"] > 0]
        losses = sells[sells["pnl"] < 0]
        return {
            "total_trades": len(sells),
            "win_trades": len(wins),
            "win_rate": len(wins) / len(sells) * 100,
            "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
            "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
            "gross_profit": wins["pnl"].sum() if len(wins) > 0 else 0,
            "gross_loss": losses["pnl"].sum() if len(losses) > 0 else 0,
            "total_pnl": sells["pnl"].sum(),
            "avg_pnl": sells["pnl"].mean(),
        }
