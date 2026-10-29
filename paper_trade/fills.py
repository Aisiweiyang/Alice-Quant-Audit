"""Validation and stable fingerprints for offline paper-trade fills."""

from __future__ import annotations

import hashlib
import json
import math

import pandas as pd


class FillValidationError(ValueError):
    """A fill report is incomplete or contains unsafe values."""


def _required_text(fill, key):
    value = str(fill.get(key) or "").strip()
    if not value:
        raise FillValidationError(f"成交回报缺少{key}")
    return value


def _positive_int(value, key):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FillValidationError(f"{key}必须是正整数") from exc
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise FillValidationError(f"{key}必须是正整数")
    return int(number)


def _nonnegative_number(value, key):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FillValidationError(f"{key}必须是非负数") from exc
    if not math.isfinite(number) or number < 0:
        raise FillValidationError(f"{key}必须是非负数")
    return number


def normalize_fill(raw_fill):
    """Return a strict JSON-compatible fill record."""
    fill = dict(raw_fill)
    action = _required_text(fill, "action").upper()
    if action not in {"BUY", "SELL"}:
        raise FillValidationError("action必须是BUY或SELL")
    price = _nonnegative_number(fill.get("price"), "price")
    if price <= 0:
        raise FillValidationError("price必须大于0")

    trade_date = _required_text(fill, "trade_date")
    try:
        trade_date = str(pd.Timestamp(trade_date).date())
    except (TypeError, ValueError) as exc:
        raise FillValidationError("trade_date必须是有效日期") from exc

    return {
        "fill_id": _required_text(fill, "fill_id"),
        "order_id": _required_text(fill, "order_id"),
        "trade_date": trade_date,
        "code": _required_text(fill, "code").lower(),
        "action": action,
        "shares": _positive_int(fill.get("shares"), "shares"),
        "price": price,
        "commission": _nonnegative_number(fill.get("commission", 0), "commission"),
        "stamp_tax": _nonnegative_number(fill.get("stamp_tax", 0), "stamp_tax"),
        "other_fee": _nonnegative_number(fill.get("other_fee", 0), "other_fee"),
    }


def fill_payload_hash(fill):
    """Fingerprint a normalized fill so repeated imports are idempotent."""
    text = json.dumps(
        normalize_fill(fill),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
