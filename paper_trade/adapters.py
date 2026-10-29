"""Read-only broker snapshot adapters for the manual paper-trading workflow."""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from paper_trade.fills import normalize_fill


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    holdings: dict
    total_assets: float | None = None

    def as_dict(self):
        return {
            "cash": self.cash,
            "holdings": self.holdings,
            "total_assets": self.total_assets,
        }


class BrokerAdapter(ABC):
    """Stable boundary for snapshots and any future broker operations."""

    @abstractmethod
    def load_account(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def load_positions(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def load_orders(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def load_fills(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order):
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id):
        raise NotImplementedError


def _number(value, field, minimum=0):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}必须是数字") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field}必须大于等于{minimum}")
    return result


def _integer(value, field, minimum=0):
    result = _number(value, field, minimum)
    if not result.is_integer():
        raise ValueError(f"{field}必须是整数")
    return int(result)


def _read_records(path, key):
    if not path:
        return []
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        frame = frame.astype(object).where(pd.notna(frame), None)
        return frame.to_dict(orient="records")
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        records = data.get(key, []) if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError(f"{key}文件必须包含记录列表")
        return records
    raise ValueError(f"{key}文件必须是CSV或JSON")


def _position(record):
    code = str(record.get("code") or "").strip().lower()
    if not code:
        raise ValueError("持仓记录缺少code")
    return {
        "code": code,
        "shares": _integer(record.get("shares"), "shares"),
        "avg_cost": _number(record.get("avg_cost", 0), "avg_cost"),
        "market_price": _number(
            record.get("market_price", record.get("last_price", 0)),
            "market_price",
        ),
    }


_STATUS_ALIASES = {
    "NEW": "SUBMITTED",
    "PENDING": "SUBMITTED",
    "ACCEPTED": "SUBMITTED",
    "PART_FILLED": "PARTIALLY_FILLED",
    "PARTIAL_FILLED": "PARTIALLY_FILLED",
    "CANCELED": "CANCELLED",
}


def _order(record):
    order_id = str(record.get("order_id") or "").strip()
    if not order_id:
        raise ValueError("订单记录缺少order_id")
    status = str(record.get("status") or "").strip().upper()
    status = _STATUS_ALIASES.get(status, status)
    action = str(record.get("action", record.get("side", ""))).strip().upper()
    code = str(record.get("code") or "").strip().lower()
    if not status or not action or not code:
        raise ValueError(f"订单记录字段不完整: {order_id}")
    ordered = record.get(
        "ordered_shares",
        record.get("estimated_shares", record.get("shares")),
    )
    return {
        "order_id": order_id,
        "status": status,
        "code": code,
        "action": action,
        "ordered_shares": _integer(ordered, "ordered_shares"),
        "filled_shares": _integer(record.get("filled_shares", 0), "filled_shares"),
    }


class ManualCsvAdapter(BrokerAdapter):
    """Load account state exported by the user; never connects to a broker."""

    def __init__(
        self,
        account_path,
        positions_path=None,
        orders_path=None,
        fills_path=None,
    ):
        self.account_path = os.path.abspath(account_path)
        self.positions_path = positions_path
        self.orders_path = orders_path
        self.fills_path = fills_path

    def load_positions(self):
        if self.positions_path:
            records = _read_records(self.positions_path, "positions")
            return [_position(record) for record in records]
        with open(self.account_path, "r", encoding="utf-8") as handle:
            account = json.load(handle)
        holdings = account.get("holdings") or {}
        records = []
        for code, position in holdings.items():
            records.append(_position({"code": code, **position}))
        return records

    def load_account(self):
        with open(self.account_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("账户文件必须是JSON对象")
        positions = self.load_positions()
        holdings = {position["code"]: position for position in positions}
        total_assets = data.get("total_assets")
        return AccountSnapshot(
            cash=_number(data.get("cash"), "cash"),
            holdings=holdings,
            total_assets=(None if total_assets is None else _number(total_assets, "total_assets")),
        )

    def load_orders(self):
        return [_order(record) for record in _read_records(self.orders_path, "orders")]

    def load_fills(self):
        return [normalize_fill(record) for record in _read_records(self.fills_path, "fills")]

    def submit_order(self, order):
        raise PermissionError("ManualCsvAdapter是只读适配器，禁止提交订单")

    def cancel_order(self, order_id):
        raise PermissionError("ManualCsvAdapter是只读适配器，禁止撤销订单")
