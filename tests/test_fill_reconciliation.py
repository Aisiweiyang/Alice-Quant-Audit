import os
import tempfile
import unittest

from paper_trade.order_identity import attach_order_ids
from paper_trade.order_ledger import (
    IdempotencyConflictError,
    OrderLedger,
    OrderStateError,
)


def planned_order():
    raw = {
        "signal_date": "2025-01-03",
        "action": "BUY",
        "code": "sh.600000",
        "estimated_shares": 1_000,
        "reference_close": 10.0,
        "score": 6.5,
        "reason": "ranked_candidate",
    }
    return attach_order_ids([raw], "2025-01-06")[0]


def fill(order, fill_id, shares, price=10.0):
    return {
        "fill_id": fill_id,
        "order_id": order["order_id"],
        "trade_date": "2025-01-06",
        "code": order["code"],
        "action": order["action"],
        "shares": shares,
        "price": price,
        "commission": 5.0,
        "stamp_tax": 0.0,
        "other_fee": 0.0,
    }


class FillReconciliationTests(unittest.TestCase):
    def make_submitted_ledger(self, directory):
        order = planned_order()
        ledger = OrderLedger(os.path.join(directory, "ledger.json"))
        ledger.reserve([order])
        ledger.transition(order["order_id"], "REVIEWED")
        ledger.transition(order["order_id"], "SUBMITTED")
        return ledger, order

    def test_partial_and_final_fills_update_cumulative_values(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, order = self.make_submitted_ledger(directory)
            first = ledger.import_fills([fill(order, "fill-1", 400, 10.0)])
            second = ledger.import_fills([fill(order, "fill-2", 600, 10.2)])
            saved = ledger.get(order["order_id"])

        self.assertEqual(first.updated_order_ids, (order["order_id"],))
        self.assertEqual(second.updated_order_ids, (order["order_id"],))
        self.assertEqual(saved["status"], "FILLED")
        self.assertEqual(saved["filled_shares"], 1_000)
        self.assertAlmostEqual(saved["average_fill_price"], 10.12)
        self.assertAlmostEqual(saved["fill_fees"], 10.0)

    def test_exact_fill_repeat_does_not_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, order = self.make_submitted_ledger(directory)
            report = fill(order, "fill-1", 400)
            ledger.import_fills([report])
            repeated = ledger.import_fills([report])
            saved = ledger.get(order["order_id"])

        self.assertEqual(repeated.existing_fill_ids, ("fill-1",))
        self.assertEqual(repeated.updated_order_ids, ())
        self.assertEqual(saved["filled_shares"], 400)

    def test_changed_content_under_same_fill_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, order = self.make_submitted_ledger(directory)
            ledger.import_fills([fill(order, "fill-1", 400)])
            with self.assertRaises(IdempotencyConflictError):
                ledger.import_fills([fill(order, "fill-1", 500)])
            saved = ledger.get(order["order_id"])

        self.assertEqual(saved["filled_shares"], 400)

    def test_batch_with_overfill_is_rejected_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, order = self.make_submitted_ledger(directory)
            reports = [
                fill(order, "fill-1", 600),
                fill(order, "fill-2", 500),
            ]
            with self.assertRaises(ValueError):
                ledger.import_fills(reports)
            saved = ledger.get(order["order_id"])

        self.assertEqual(saved["status"], "SUBMITTED")
        self.assertNotIn("filled_shares", saved)

    def test_fill_cannot_arrive_before_order_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            order = planned_order()
            ledger = OrderLedger(os.path.join(directory, "ledger.json"))
            ledger.reserve([order])
            with self.assertRaises(OrderStateError):
                ledger.import_fills([fill(order, "fill-1", 100)])

    def test_fill_date_cannot_precede_execution_date(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, order = self.make_submitted_ledger(directory)
            report = fill(order, "fill-1", 100)
            report["trade_date"] = "2025-01-03"
            with self.assertRaises(ValueError):
                ledger.import_fills([report])
            saved = ledger.get(order["order_id"])

        self.assertEqual(saved["status"], "SUBMITTED")
        self.assertNotIn("filled_shares", saved)


if __name__ == "__main__":
    unittest.main()
