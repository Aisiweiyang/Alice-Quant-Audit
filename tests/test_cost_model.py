import unittest

from engine.broker import Broker
from engine.costs import ChinaAStockCostModel


class ZeroCommissionConfig:
    commission_rate = 0.0
    min_commission = 0.0
    slippage = 0.0
    stamp_tax_rate = 0.0005
    stamp_tax_rate_before_20230828 = 0.001
    stamp_tax_rate_after_20230828 = 0.0005


class CostModelTests(unittest.TestCase):
    def test_stamp_tax_uses_trade_date(self):
        model = ChinaAStockCostModel(ZeroCommissionConfig)
        old = model.calculate("SELL", 10.0, 100, "2023-08-25")
        new = model.calculate("SELL", 10.0, 100, "2023-08-28")
        self.assertAlmostEqual(old.stamp_tax, 1.0)
        self.assertAlmostEqual(new.stamp_tax, 0.5)
        self.assertAlmostEqual(new.cash_amount - old.cash_amount, 0.5)

    def test_buy_never_charges_stamp_tax(self):
        model = ChinaAStockCostModel(ZeroCommissionConfig)
        trade = model.calculate("BUY", 10.0, 100, "2022-01-04")
        self.assertEqual(trade.stamp_tax, 0.0)

    def test_broker_keeps_auditable_cost_breakdown(self):
        broker = Broker(ZeroCommissionConfig)
        _, revenue = broker.sell(10.0, 100, "2025-01-02")
        self.assertAlmostEqual(revenue, 999.5)
        self.assertEqual(broker.last_cost.trade_date, "2025-01-02")
        self.assertIn("total_friction", broker.last_cost.to_dict())


if __name__ == "__main__":
    unittest.main()
