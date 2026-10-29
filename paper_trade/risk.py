"""Pre-trade controls for reviewable paper-order plans."""

from __future__ import annotations

import math
from dataclasses import dataclass


ACTIVE_ORDER_STATES = frozenset({"SUBMITTED", "PARTIALLY_FILLED"})


@dataclass(frozen=True)
class RiskLimits:
    emergency_stop: bool = False
    max_total_position_pct: float = 0.95
    max_position_pct: float = 0.12
    max_order_value_pct: float = 0.15
    max_active_orders: int = 20
    daily_loss_limit_pct: float = 0.03

    @classmethod
    def from_config(cls, config, emergency_stop=None):
        return cls(
            emergency_stop=(
                bool(config.emergency_stop) if emergency_stop is None else bool(emergency_stop)
            ),
            max_total_position_pct=float(config.max_total_position_pct),
            max_position_pct=float(config.max_position_pct),
            max_order_value_pct=float(config.max_order_value_pct),
            max_active_orders=int(config.max_active_orders),
            daily_loss_limit_pct=float(config.daily_loss_limit_pct),
        )


@dataclass(frozen=True)
class RiskAssessment:
    allowed_orders: tuple[dict, ...]
    rejected_orders: tuple[dict, ...]
    metrics: dict

    def as_dict(self):
        return {
            "passed": not self.rejected_orders,
            "allowed_order_ids": [order["order_id"] for order in self.allowed_orders],
            "rejected_orders": list(self.rejected_orders),
            "metrics": self.metrics,
        }


def _finite_number(value, default=0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _holding_value(position):
    shares = int(_finite_number(position.get("shares"), 0))
    price = _finite_number(
        position.get(
            "market_price",
            position.get("last_price", position.get("avg_cost", 0)),
        ),
        0,
    )
    return max(0, shares) * max(0, price)


def _active_order_count(ledger_snapshot):
    return sum(
        1
        for order in ledger_snapshot.get("orders", {}).values()
        if order.get("status") in ACTIVE_ORDER_STATES
    )


def evaluate_orders(
    orders,
    account,
    ledger_snapshot,
    limits,
    reconciliation_passed=True,
):
    """Filter an order plan through deterministic, fail-closed controls."""
    holdings = dict(account.get("holdings") or {})
    cash = _finite_number(account.get("cash"), 0)
    values = {str(code).lower(): _holding_value(position) for code, position in holdings.items()}
    shares_by_code = {
        str(code).lower(): int(_finite_number(position.get("shares"), 0))
        for code, position in holdings.items()
    }
    exposure = sum(values.values())
    calculated_equity = cash + exposure
    current_equity = _finite_number(
        account.get("current_equity"),
        calculated_equity,
    )
    start_equity = _finite_number(
        account.get("start_of_day_equity"),
        current_equity,
    )
    if current_equity <= 0:
        raise ValueError("账户总权益必须大于0")
    loss_pct = 0 if start_equity <= 0 else max(0, (start_equity - current_equity) / start_equity)
    daily_loss_block = loss_pct >= limits.daily_loss_limit_pct
    active_orders = _active_order_count(ledger_snapshot)
    allowed = []
    rejected = []

    for raw_order in orders:
        order = dict(raw_order)
        order_id = str(order.get("order_id") or "")
        action = str(order.get("action") or "").upper()
        code = str(order.get("code") or "").lower()
        shares = int(_finite_number(order.get("estimated_shares"), 0))
        price = _finite_number(order.get("reference_close"), 0)
        notional = shares * price
        reasons = []

        if action not in {"BUY", "SELL"}:
            reasons.append("invalid_action")
        if shares <= 0 or price <= 0:
            reasons.append("invalid_order_value")

        if action == "BUY":
            if not reconciliation_passed:
                reasons.append("account_reconciliation_failed")
            if limits.emergency_stop:
                reasons.append("emergency_stop")
            if daily_loss_block:
                reasons.append("daily_loss_limit")
            if active_orders >= limits.max_active_orders:
                reasons.append("active_order_limit")
            if notional / current_equity > limits.max_order_value_pct + 1e-12:
                reasons.append("order_value_limit")
            code_value = values.get(code, 0) + notional
            if code_value / current_equity > limits.max_position_pct + 1e-12:
                reasons.append("single_position_limit")
            if (exposure + notional) / current_equity > limits.max_total_position_pct + 1e-12:
                reasons.append("total_position_limit")
        elif action == "SELL":
            if shares > shares_by_code.get(code, 0):
                reasons.append("insufficient_position")

        if reasons:
            rejected.append(
                {
                    "order_id": order_id,
                    "action": action,
                    "code": code,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue

        allowed.append(order)
        active_orders += 1
        if action == "BUY":
            values[code] = values.get(code, 0) + notional
            exposure += notional
            shares_by_code[code] = shares_by_code.get(code, 0) + shares
        else:
            reduction = min(values.get(code, 0), notional)
            values[code] = max(0, values.get(code, 0) - reduction)
            exposure = max(0, exposure - reduction)
            shares_by_code[code] = max(0, shares_by_code.get(code, 0) - shares)

    return RiskAssessment(
        tuple(allowed),
        tuple(rejected),
        {
            "emergency_stop": limits.emergency_stop,
            "reconciliation_passed": bool(reconciliation_passed),
            "current_equity": current_equity,
            "start_of_day_equity": start_equity,
            "daily_loss_pct": loss_pct,
            "daily_loss_block": daily_loss_block,
            "active_orders_after_plan": active_orders,
            "projected_position_pct": exposure / current_equity,
        },
    )
