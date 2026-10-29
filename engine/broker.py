"""模拟券商：委托成交价格与交易成本计算。"""

from engine.costs import ChinaAStockCostModel


class Broker:
    """执行成交计算并保留最近一笔成本明细。"""

    def __init__(self, cfg):
        self.cost_model = ChinaAStockCostModel(cfg)
        self.slippage = self.cost_model.slippage
        self.last_cost = None

    def buy(self, price, shares, trade_date=None):
        """返回买入实际成交价与现金支出。"""
        self.last_cost = self.cost_model.calculate("BUY", price, shares, trade_date)
        return self.last_cost.execution_price, self.last_cost.cash_amount

    def sell(self, price, shares, trade_date=None):
        """返回卖出实际成交价与现金净收入。"""
        self.last_cost = self.cost_model.calculate("SELL", price, shares, trade_date)
        return self.last_cost.execution_price, self.last_cost.cash_amount

    def buy_shares(self, price, cash, max_pct, trade_date=None):
        """根据可用资金和仓位限制，计算可买入股数
        Args:
            price: 当前价格
            cash: 可用现金
            max_pct: 单只股票最大仓位比例
        Returns:
            int: 可买股数（100股整数倍）
        """
        if price <= 0 or cash <= 0 or max_pct <= 0:
            return 0

        max_cost = min(cash, cash * max_pct)
        raw_shares = int(max_cost / (price * (1 + self.slippage)))
        shares = (raw_shares // 100) * 100

        # 佣金（尤其最低5元）也必须落在预算内。
        while shares > 0:
            _, total_cost = self.buy(price, shares, trade_date)
            if total_cost <= max_cost:
                return shares
            shares -= 100
        return 0
