"""Conservative A-share order-execution constraints."""

from __future__ import annotations

import math

import pandas as pd


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def price_limit_pct(code: str, date, is_st=False) -> float:
    """Return the normal daily price-limit percentage for the security."""
    if bool(int(_number(is_st, 0))):
        return 5.0
    tail = str(code).split(".")[-1]
    target = pd.Timestamp(date).normalize()
    if tail.startswith("688"):
        return 20.0
    if tail.startswith(("300", "301")) and target >= pd.Timestamp("2020-08-24"):
        return 20.0
    if tail.startswith(("4", "8", "92")):
        return 30.0
    return 10.0


def order_is_executable(code: str, row, side: str):
    """Check suspension and open-price limit constraints for an order."""
    action = str(side).upper()
    if action not in {"BUY", "SELL"}:
        raise ValueError("side必须是BUY或SELL")

    trade_status = _number(row.get("tradestatus"), 1)
    volume = _number(row.get("volume"), 0)
    open_price = _number(row.get("open"), 0)
    if trade_status == 0 or volume <= 0:
        return False, "suspended"
    if open_price <= 0:
        return False, "invalid_open"

    preclose = _number(row.get("preclose"))
    limit_pct = price_limit_pct(code, row.get("date"), row.get("isST", 0))
    if preclose and preclose > 0:
        if action == "BUY":
            limit_price = preclose * (1 + limit_pct / 100)
            if open_price >= limit_price * 0.998:
                return False, "limit_up_at_open"
        else:
            limit_price = preclose * (1 - limit_pct / 100)
            if open_price <= limit_price * 1.002:
                return False, "limit_down_at_open"
    else:
        pct_change = _number(row.get("pctChg"), 0)
        high = _number(row.get("high"), open_price)
        low = _number(row.get("low"), open_price)
        one_price = abs(high - low) <= max(abs(open_price), 1) * 1e-6
        if action == "BUY" and one_price and pct_change >= limit_pct - 0.2:
            return False, "limit_up_at_open"
        if action == "SELL" and one_price and pct_change <= -limit_pct + 0.2:
            return False, "limit_down_at_open"
    return True, "ok"


def closing_sale_is_executable(code: str, row):
    """Check whether an end-of-period sale at the close price could really fill.

    Used for forced liquidation at the end of a backtest: a suspended stock or
    a close pinned at the limit-down price cannot be sold at that close.
    """
    trade_status = _number(row.get("tradestatus"), 1)
    volume = _number(row.get("volume"), 0)
    close_price = _number(row.get("close"), 0)
    if trade_status == 0 or volume <= 0:
        return False, "suspended"
    if close_price <= 0:
        return False, "invalid_close"

    preclose = _number(row.get("preclose"))
    limit_pct = price_limit_pct(code, row.get("date"), row.get("isST", 0))
    if preclose and preclose > 0:
        limit_price = preclose * (1 - limit_pct / 100)
        if close_price <= limit_price * 1.002:
            return False, "limit_down_at_close"
    else:
        pct_change = _number(row.get("pctChg"), 0)
        high = _number(row.get("high"), close_price)
        low = _number(row.get("low"), close_price)
        one_price = abs(high - low) <= max(abs(close_price), 1) * 1e-6
        if one_price and pct_change <= -limit_pct + 0.2:
            return False, "limit_down_at_close"
    return True, "ok"
