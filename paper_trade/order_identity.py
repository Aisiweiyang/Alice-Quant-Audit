"""Stable paper-order identities and payload fingerprints."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


ORDER_ID_SCHEMA = "alice-paper-order-v1"


def _date_text(value) -> str:
    return str(pd.Timestamp(value).date())


def _plain_value(value):
    """Convert common scalar values to stable JSON-compatible values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return _date_text(value)
    return value


def _digest(payload: dict) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_order_id(order: dict, execution_date) -> str:
    """Return a deterministic client order id for one trading intent.

    Quantity and price are deliberately excluded from the identity. If the
    same intent is regenerated with a changed payload, the ledger treats it as
    a conflict instead of silently creating a second order.
    """
    payload = {
        "schema": ORDER_ID_SCHEMA,
        "signal_date": _date_text(order["signal_date"]),
        "execution_date": _date_text(execution_date),
        "action": str(order["action"]).upper(),
        "code": str(order["code"]).lower(),
    }
    return f"alice-{_digest(payload)[:24]}"


def order_payload_hash(order: dict) -> str:
    """Fingerprint the material fields protected by an order id."""
    fields = (
        "signal_date",
        "execution_date",
        "action",
        "code",
        "estimated_shares",
        "reference_close",
        "score",
        "reason",
    )
    payload = {key: _plain_value(order.get(key)) for key in fields}
    return _digest(payload)


def attach_order_ids(orders, execution_date):
    """Copy orders, attach stable ids, and reject duplicates in one plan."""
    result = []
    seen = set()
    execution_text = _date_text(execution_date)
    for raw_order in orders:
        order = dict(raw_order)
        order["signal_date"] = _date_text(order["signal_date"])
        order["execution_date"] = execution_text
        order["action"] = str(order["action"]).upper()
        order["code"] = str(order["code"]).lower()
        order_id = make_order_id(order, execution_text)
        if order_id in seen:
            raise ValueError(f"订单计划存在重复交易意图: {order_id}")
        seen.add(order_id)
        order["order_id"] = order_id
        result.append(order)
    return result
