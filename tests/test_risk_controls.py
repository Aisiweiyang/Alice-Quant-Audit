import unittest

from paper_trade.risk import RiskLimits, evaluate_orders


def order(action="BUY", code="sh.600000", shares=1_000, price=10.0):
    return {
        "order_id": f"{action}-{code}",
        "action": action,
        "code": code,
        "estimated_shares": shares,
        "reference_close": price,
    }


def account(
    cash=100_000,
    holdings=None,
    start_of_day_equity=100_000,
    current_equity=100_000,
):
    return {
        "cash": cash,
        "holdings": holdings or {},
        "start_of_day_equity": start_of_day_equity,
        "current_equity": current_equity,
    }


def ledger(active=0):
    return {"orders": {f"active-{index}": {"status": "SUBMITTED"} for index in range(active)}}


class RiskControlTests(unittest.TestCase):
    def test_emergency_stop_blocks_buy_but_allows_position_reducing_sell(self):
        holdings = {
            "sh.600000": {
                "shares": 1_000,
                "avg_cost": 10.0,
                "market_price": 10.0,
            }
        }
        limits = RiskLimits(emergency_stop=True)
        result = evaluate_orders(
            [
                order("BUY", "sz.000001", 1_000, 10),
                order("SELL", "sh.600000", 1_000, 10),
            ],
            account(cash=90_000, holdings=holdings),
            ledger(),
            limits,
        )
        self.assertEqual([item["action"] for item in result.allowed_orders], ["SELL"])
        self.assertIn("emergency_stop", result.rejected_orders[0]["reasons"])

    def test_failed_reconciliation_blocks_new_buys(self):
        result = evaluate_orders(
            [order()],
            account(),
            ledger(),
            RiskLimits(),
            reconciliation_passed=False,
        )
        self.assertEqual(result.allowed_orders, ())
        self.assertIn(
            "account_reconciliation_failed",
            result.rejected_orders[0]["reasons"],
        )

    def test_daily_loss_limit_blocks_new_buys(self):
        result = evaluate_orders(
            [order()],
            account(start_of_day_equity=100_000, current_equity=96_000),
            ledger(),
            RiskLimits(daily_loss_limit_pct=0.03),
        )
        self.assertIn("daily_loss_limit", result.rejected_orders[0]["reasons"])

    def test_active_order_limit_is_enforced(self):
        result = evaluate_orders(
            [order()],
            account(),
            ledger(active=2),
            RiskLimits(max_active_orders=2),
        )
        self.assertIn("active_order_limit", result.rejected_orders[0]["reasons"])

    def test_single_and_total_position_limits_are_enforced(self):
        holdings = {
            "sh.600000": {
                "shares": 8_500,
                "avg_cost": 10.0,
                "market_price": 10.0,
            }
        }
        result = evaluate_orders(
            [order("BUY", "sh.600000", 1_000, 10)],
            account(cash=15_000, holdings=holdings),
            ledger(),
            RiskLimits(
                max_position_pct=0.90,
                max_total_position_pct=0.90,
                max_order_value_pct=0.20,
            ),
        )
        reasons = result.rejected_orders[0]["reasons"]
        self.assertIn("single_position_limit", reasons)
        self.assertIn("total_position_limit", reasons)

    def test_sell_cannot_exceed_position(self):
        result = evaluate_orders(
            [order("SELL", "sh.600000", 1_000, 10)],
            account(),
            ledger(),
            RiskLimits(),
        )
        self.assertIn("insufficient_position", result.rejected_orders[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
