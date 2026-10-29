import unittest

from paper_trade.adapters import AccountSnapshot
from paper_trade.reconciliation import reconcile_account


def account(cash=90_000, shares=1_000):
    return AccountSnapshot(
        cash=cash,
        total_assets=100_000,
        holdings={
            "sh.600000": {
                "code": "sh.600000",
                "shares": shares,
                "avg_cost": 10.0,
                "market_price": 10.0,
            }
        },
    )


def ledger_snapshot():
    return {
        "schema_version": 1,
        "orders": {
            "order-1": {
                "status": "FILLED",
                "code": "sh.600000",
                "action": "BUY",
                "estimated_shares": 1_000,
                "filled_shares": 1_000,
            }
        },
        "fills": {
            "fill-1": {
                "fill_id": "fill-1",
                "order_id": "order-1",
                "trade_date": "2025-01-06",
                "code": "sh.600000",
                "action": "BUY",
                "shares": 1_000,
                "price": 10.0,
                "commission": 5.0,
                "stamp_tax": 0.0,
                "other_fee": 0.0,
                "payload_hash": "ignored",
            }
        },
    }


def broker_orders():
    return [
        {
            "order_id": "order-1",
            "status": "FILLED",
            "code": "sh.600000",
            "action": "BUY",
            "ordered_shares": 1_000,
            "filled_shares": 1_000,
        }
    ]


def broker_fills():
    return [
        {
            "fill_id": "fill-1",
            "order_id": "order-1",
            "trade_date": "2025-01-06",
            "code": "sh.600000",
            "action": "BUY",
            "shares": 1_000,
            "price": 10.0,
            "commission": 5.0,
            "stamp_tax": 0.0,
            "other_fee": 0.0,
        }
    ]


class AccountReconciliationTests(unittest.TestCase):
    def test_identical_snapshots_pass(self):
        report = reconcile_account(
            account(),
            account(),
            ledger_snapshot(),
            broker_orders(),
            broker_fills(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["blocking_issue_count"], 0)

    def test_cash_position_order_and_fill_differences_block(self):
        actual_orders = broker_orders()
        actual_orders[0]["status"] = "SUBMITTED"
        actual_fills = broker_fills()
        actual_fills[0]["price"] = 10.5
        report = reconcile_account(
            account(),
            account(cash=89_000, shares=900),
            ledger_snapshot(),
            actual_orders,
            actual_fills,
        )
        categories = {issue["category"] for issue in report["issues"]}
        self.assertFalse(report["passed"])
        self.assertIn("CASH", categories)
        self.assertIn("POSITION_SHARES", categories)
        self.assertIn("ORDER_STATUS", categories)
        self.assertIn("FILL_PRICE", categories)

    def test_unknown_broker_order_blocks(self):
        orders = broker_orders()
        orders.append(
            {
                "order_id": "external-order",
                "status": "SUBMITTED",
                "code": "sz.000001",
                "action": "BUY",
                "ordered_shares": 100,
                "filled_shares": 0,
            }
        )
        report = reconcile_account(
            account(),
            account(),
            ledger_snapshot(),
            orders,
            broker_fills(),
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"][0]["category"], "ORDER_MISSING")


if __name__ == "__main__":
    unittest.main()
