import json
import os
import tempfile
import unittest

from paper_trade.order_identity import attach_order_ids
from paper_trade.order_ledger import (
    IdempotencyConflictError,
    OrderLedger,
    OrderStateError,
)


def sample_order(shares=1_000):
    return {
        "signal_date": "2025-01-03",
        "action": "BUY",
        "code": "sh.600000",
        "estimated_shares": shares,
        "reference_close": 10.0,
        "score": 6.5,
        "reason": "ranked_candidate",
    }


class OrderIdempotencyTests(unittest.TestCase):
    def test_same_intent_gets_same_id_across_reruns(self):
        first = attach_order_ids([sample_order()], "2025-01-06")[0]
        second = attach_order_ids([sample_order()], "2025-01-06")[0]
        self.assertEqual(first["order_id"], second["order_id"])
        self.assertEqual(first["execution_date"], "2025-01-06")

    def test_ledger_reports_exact_repeat_without_adding_a_second_record(self):
        order = attach_order_ids([sample_order()], "2025-01-06")[0]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.json")
            ledger = OrderLedger(path)
            first = ledger.reserve([order])
            second = ledger.reserve([order])
            with open(path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)

        self.assertEqual(first.new_order_ids, (order["order_id"],))
        self.assertEqual(second.existing_order_ids, (order["order_id"],))
        self.assertEqual(len(saved["orders"]), 1)

    def test_changed_payload_under_same_id_is_rejected(self):
        original = attach_order_ids([sample_order()], "2025-01-06")[0]
        changed = attach_order_ids([sample_order(shares=2_000)], "2025-01-06")[0]
        self.assertEqual(original["order_id"], changed["order_id"])

        with tempfile.TemporaryDirectory() as directory:
            ledger = OrderLedger(os.path.join(directory, "ledger.json"))
            ledger.reserve([original])
            with self.assertRaises(IdempotencyConflictError):
                ledger.reserve([changed])

    def test_order_state_requires_review_before_submission(self):
        order = attach_order_ids([sample_order()], "2025-01-06")[0]
        with tempfile.TemporaryDirectory() as directory:
            ledger = OrderLedger(os.path.join(directory, "ledger.json"))
            ledger.reserve([order])
            with self.assertRaises(OrderStateError):
                ledger.transition(order["order_id"], "SUBMITTED")
            self.assertEqual(ledger.get(order["order_id"])["status"], "PLANNED")

    def test_valid_order_lifecycle_is_saved_with_audit_events(self):
        order = attach_order_ids([sample_order()], "2025-01-06")[0]
        with tempfile.TemporaryDirectory() as directory:
            ledger = OrderLedger(os.path.join(directory, "ledger.json"))
            ledger.reserve([order])
            ledger.transition(order["order_id"], "REVIEWED", {"reviewer": "manual"})
            ledger.transition(order["order_id"], "SUBMITTED")
            filled = ledger.transition(
                order["order_id"],
                "FILLED",
                {"filled_shares": 1_000, "fill_price": 10.08},
            )

        self.assertEqual(filled["status"], "FILLED")
        self.assertEqual(
            [event["to"] for event in filled["events"]],
            [
                "REVIEWED",
                "SUBMITTED",
                "FILLED",
            ],
        )

    def test_repeating_same_state_is_idempotent(self):
        order = attach_order_ids([sample_order()], "2025-01-06")[0]
        with tempfile.TemporaryDirectory() as directory:
            ledger = OrderLedger(os.path.join(directory, "ledger.json"))
            ledger.reserve([order])
            first = ledger.transition(order["order_id"], "REVIEWED")
            second = ledger.transition(order["order_id"], "REVIEWED")

        self.assertEqual(first, second)
        self.assertEqual(len(second["events"]), 1)


if __name__ == "__main__":
    unittest.main()
