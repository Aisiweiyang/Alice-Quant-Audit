"""Atomic local ledger used to guard paper orders against duplicate plans."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass

import pandas as pd

from paper_trade.fills import fill_payload_hash, normalize_fill
from paper_trade.order_identity import order_payload_hash


LEDGER_SCHEMA = 1
ORDER_STATES = frozenset(
    {
        "PLANNED",
        "REVIEWED",
        "SUBMITTED",
        "PARTIALLY_FILLED",
        "FILLED",
        "REJECTED",
        "CANCELLED",
    }
)
ALLOWED_TRANSITIONS = {
    "PLANNED": frozenset({"REVIEWED", "CANCELLED"}),
    "REVIEWED": frozenset({"SUBMITTED", "CANCELLED"}),
    "SUBMITTED": frozenset(
        {
            "PARTIALLY_FILLED",
            "FILLED",
            "REJECTED",
            "CANCELLED",
        }
    ),
    "PARTIALLY_FILLED": frozenset({"FILLED", "CANCELLED"}),
    "FILLED": frozenset(),
    "REJECTED": frozenset(),
    "CANCELLED": frozenset(),
}


class IdempotencyConflictError(ValueError):
    """The same order id was regenerated with different material fields."""


class OrderStateError(ValueError):
    """An order is missing or attempted an illegal lifecycle transition."""


@dataclass(frozen=True)
class ReservationResult:
    new_order_ids: tuple[str, ...]
    existing_order_ids: tuple[str, ...]


@dataclass(frozen=True)
class FillImportResult:
    new_fill_ids: tuple[str, ...]
    existing_fill_ids: tuple[str, ...]
    updated_order_ids: tuple[str, ...]


class OrderLedger:
    """Persist planned order ids without connecting to any broker."""

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def _load(self):
        if not os.path.exists(self.path):
            return {"schema_version": LEDGER_SCHEMA, "orders": {}, "fills": {}}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema_version") != LEDGER_SCHEMA:
            raise ValueError("不支持的订单台账版本")
        if not isinstance(data.get("orders"), dict):
            raise ValueError("订单台账格式无效")
        data.setdefault("fills", {})
        if not isinstance(data["fills"], dict):
            raise ValueError("成交台账格式无效")
        return data

    def _write(self, data):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".order_ledger_", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def reserve(self, orders) -> ReservationResult:
        """Reserve new ids and report exact repeats without duplicating them."""
        data = self._load()
        records = data["orders"]
        additions = {}
        new_ids = []
        existing_ids = []

        for order in orders:
            order_id = str(order.get("order_id") or "")
            if not order_id:
                raise ValueError("订单缺少order_id")
            fingerprint = order_payload_hash(order)
            known = records.get(order_id) or additions.get(order_id)
            if known is not None:
                if known.get("payload_hash") != fingerprint:
                    raise IdempotencyConflictError(
                        f"订单编号对应的内容发生变化，请人工复核: {order_id}"
                    )
                existing_ids.append(order_id)
                continue
            additions[order_id] = {
                "payload_hash": fingerprint,
                "status": "PLANNED",
                "signal_date": order["signal_date"],
                "execution_date": order["execution_date"],
                "action": order["action"],
                "code": order["code"],
                "estimated_shares": int(order["estimated_shares"]),
            }
            new_ids.append(order_id)

        if additions:
            records.update(additions)
            self._write(data)
        return ReservationResult(tuple(new_ids), tuple(existing_ids))

    def get(self, order_id):
        """Return a copy of one order record, or None when it is unknown."""
        record = self._load()["orders"].get(str(order_id))
        return dict(record) if record is not None else None

    def snapshot(self):
        """Return an isolated copy for reporting and reconciliation."""
        return copy.deepcopy(self._load())

    def transition(self, order_id, new_status, details=None):
        """Apply one validated, atomic and idempotent lifecycle transition."""
        data = self._load()
        record = data["orders"].get(str(order_id))
        if record is None:
            raise OrderStateError(f"订单台账中不存在该编号: {order_id}")

        current = str(record.get("status") or "").upper()
        target = str(new_status or "").upper()
        if current not in ORDER_STATES:
            raise OrderStateError(f"订单当前状态无效: {current}")
        if target not in ORDER_STATES:
            raise OrderStateError(f"目标订单状态无效: {target}")
        if target == current:
            return dict(record)
        if target not in ALLOWED_TRANSITIONS[current]:
            raise OrderStateError(f"不允许的订单状态迁移: {current} -> {target}")

        events = list(record.get("events") or [])
        events.append(
            {
                "sequence": len(events) + 1,
                "from": current,
                "to": target,
                "details": dict(details or {}),
            }
        )
        record["status"] = target
        record["events"] = events
        self._write(data)
        return dict(record)

    def import_fills(self, fills) -> FillImportResult:
        """Atomically reconcile offline fills against planned order records."""
        data = self._load()
        orders = data["orders"]
        known_fills = data["fills"]
        new_fill_ids = []
        existing_fill_ids = []
        updated_order_ids = []

        for raw_fill in fills:
            fill = normalize_fill(raw_fill)
            fill_id = fill["fill_id"]
            fingerprint = fill_payload_hash(fill)
            known = known_fills.get(fill_id)
            if known is not None:
                if known.get("payload_hash") != fingerprint:
                    raise IdempotencyConflictError(
                        f"成交编号对应的内容发生变化，请人工复核: {fill_id}"
                    )
                existing_fill_ids.append(fill_id)
                continue

            order_id = fill["order_id"]
            order = orders.get(order_id)
            if order is None:
                raise OrderStateError(f"成交回报引用了未知订单: {order_id}")
            current = str(order.get("status") or "").upper()
            if current not in {"SUBMITTED", "PARTIALLY_FILLED"}:
                raise OrderStateError(f"订单状态不允许导入成交回报: {order_id} ({current})")
            if fill["code"] != str(order["code"]).lower():
                raise ValueError(f"成交股票与订单不一致: {fill_id}")
            if fill["action"] != str(order["action"]).upper():
                raise ValueError(f"成交方向与订单不一致: {fill_id}")
            if pd.Timestamp(fill["trade_date"]) < pd.Timestamp(order["execution_date"]):
                raise ValueError(f"成交日期早于计划执行日期: {fill_id}")

            ordered_shares = int(order["estimated_shares"])
            prior_shares = int(order.get("filled_shares", 0))
            total_shares = prior_shares + fill["shares"]
            if total_shares > ordered_shares:
                raise ValueError(
                    f"累计成交数量超过订单数量: {order_id} ({total_shares}>{ordered_shares})"
                )

            prior_value = float(order.get("filled_value", 0))
            total_value = prior_value + fill["shares"] * fill["price"]
            target = "FILLED" if total_shares == ordered_shares else "PARTIALLY_FILLED"
            if target != current and target not in ALLOWED_TRANSITIONS[current]:
                raise OrderStateError(f"不允许的订单状态迁移: {current} -> {target}")

            total_fees = (
                float(order.get("fill_fees", 0))
                + fill["commission"]
                + fill["stamp_tax"]
                + fill["other_fee"]
            )
            events = list(order.get("events") or [])
            events.append(
                {
                    "sequence": len(events) + 1,
                    "from": current,
                    "to": target,
                    "details": {
                        "type": "FILL_RECONCILIATION",
                        "fill_id": fill_id,
                        "shares": fill["shares"],
                        "price": fill["price"],
                    },
                }
            )
            order["status"] = target
            order["filled_shares"] = total_shares
            order["filled_value"] = total_value
            order["average_fill_price"] = total_value / total_shares
            order["fill_fees"] = total_fees
            order["events"] = events
            known_fills[fill_id] = {
                **fill,
                "payload_hash": fingerprint,
            }
            new_fill_ids.append(fill_id)
            if order_id not in updated_order_ids:
                updated_order_ids.append(order_id)

        if new_fill_ids:
            self._write(data)
        return FillImportResult(
            tuple(new_fill_ids),
            tuple(existing_fill_ids),
            tuple(updated_order_ids),
        )
