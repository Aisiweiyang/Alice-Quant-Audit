"""A-share transaction cost models.

The model is intentionally independent from the broker and portfolio so that
historical fee schedules can be tested without running a full backtest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime


STAMP_TAX_CHANGE_DATE = date(2023, 8, 28)


def _as_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class CostBreakdown:
    side: str
    trade_date: str | None
    reference_price: float
    execution_price: float
    shares: int
    gross_value: float
    commission: float
    stamp_tax: float
    slippage_cost: float
    cash_amount: float

    @property
    def explicit_fees(self) -> float:
        return self.commission + self.stamp_tax

    @property
    def total_friction(self) -> float:
        return self.explicit_fees + self.slippage_cost

    def to_dict(self) -> dict:
        result = asdict(self)
        result["explicit_fees"] = self.explicit_fees
        result["total_friction"] = self.total_friction
        return result


class ChinaAStockCostModel:
    """Calculate commission, slippage and dated sell-side stamp tax."""

    def __init__(self, config):
        self.commission_rate = float(config.commission_rate)
        self.min_commission = float(config.min_commission)
        self.slippage = float(config.slippage)
        legacy_rate = float(getattr(config, "stamp_tax_rate", 0.001))
        self.stamp_tax_rate_before = float(
            getattr(config, "stamp_tax_rate_before_20230828", legacy_rate)
        )
        self.stamp_tax_rate_after = float(
            getattr(config, "stamp_tax_rate_after_20230828", legacy_rate)
        )

    def stamp_tax_rate(self, trade_date=None) -> float:
        current = _as_date(trade_date)
        if current is None:
            return self.stamp_tax_rate_after
        if current < STAMP_TAX_CHANGE_DATE:
            return self.stamp_tax_rate_before
        return self.stamp_tax_rate_after

    def calculate(self, side, price, shares, trade_date=None) -> CostBreakdown:
        normalized_side = str(side).upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError(f"不支持的交易方向: {side}")
        price = float(price)
        shares = int(shares)
        if price <= 0 or shares <= 0:
            raise ValueError("price和shares必须大于0")

        direction = 1 if normalized_side == "BUY" else -1
        execution_price = price * (1 + direction * self.slippage)
        gross_value = execution_price * shares
        commission = max(gross_value * self.commission_rate, self.min_commission)
        stamp_tax = (
            gross_value * self.stamp_tax_rate(trade_date) if normalized_side == "SELL" else 0.0
        )
        slippage_cost = abs(execution_price - price) * shares
        cash_amount = (
            gross_value + commission
            if normalized_side == "BUY"
            else gross_value - commission - stamp_tax
        )
        normalized_date = _as_date(trade_date)
        return CostBreakdown(
            side=normalized_side,
            trade_date=normalized_date.isoformat() if normalized_date else None,
            reference_price=price,
            execution_price=execution_price,
            shares=shares,
            gross_value=gross_value,
            commission=commission,
            stamp_tax=stamp_tax,
            slippage_cost=slippage_cost,
            cash_amount=cash_amount,
        )
