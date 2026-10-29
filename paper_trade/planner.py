"""Create reviewable next-session orders without connecting to a broker."""

from __future__ import annotations

import pandas as pd

from paper_trade.order_identity import attach_order_ids
from strategy.config import StrategyConfig as cfg
from strategy.selection import market_allows_buy, select_candidates


def _last_close(frame, signal_date):
    rows = frame.loc[frame["date"] <= pd.Timestamp(signal_date)]
    if rows.empty:
        return None
    value = float(rows.iloc[-1]["close"])
    return value if value > 0 else None


def next_session_is_rebalance(signal_date, execution_date, config=cfg):
    """Return whether the next session starts a configured rebalance period."""
    signal = pd.Timestamp(signal_date)
    execution = pd.Timestamp(execution_date)
    if execution <= signal:
        raise ValueError("execution_date必须晚于signal_date")
    if config.rebalance_freq == "daily":
        return True
    if config.rebalance_freq == "monthly":
        return (signal.year, signal.month) != (execution.year, execution.month)
    if config.rebalance_freq == "weekly":
        signal_week = signal.isocalendar()
        execution_week = execution.isocalendar()
        return (signal_week.year, signal_week.week) != (execution_week.year, execution_week.week)
    raise ValueError(f"不支持的调仓频率: {config.rebalance_freq}")


def build_order_plan(data, signal_date, execution_date, account, config=cfg):
    """Build estimated orders for manual execution on the next trading day.

    The returned share count is an estimate based on the signal-day close.  The
    user must review orders and use the real next-session price in a simulator.
    """
    benchmark = data.get(config.benchmark_index)
    stocks = {code: frame for code, frame in data.items() if code != config.benchmark_index}
    holdings = dict(account.get("holdings") or {})
    cash = float(account.get("cash", config.initial_capital))
    orders = []

    surviving = set(holdings)
    for code, position in holdings.items():
        frame = stocks.get(code)
        close = _last_close(frame, signal_date) if frame is not None else None
        if close is None:
            continue
        avg_cost = float(position.get("avg_cost", 0))
        if avg_cost <= 0:
            continue
        if close >= avg_cost * (1 + config.take_profit_pct):
            reason = "take_profit"
        elif close <= avg_cost * (1 - config.stop_loss_pct):
            reason = "stop_loss"
        else:
            continue
        orders.append(
            {
                "signal_date": str(pd.Timestamp(signal_date).date()),
                "action": "SELL",
                "code": code,
                "estimated_shares": int(position.get("shares", 0)),
                "reference_close": close,
                "score": None,
                "reason": reason,
            }
        )
        surviving.discard(code)

    if not next_session_is_rebalance(signal_date, execution_date, config):
        return attach_order_ids(orders, execution_date), {
            "can_buy": False,
            "reason": "not_rebalance_session",
        }

    allowed, timing_reason = market_allows_buy(benchmark, signal_date, config)
    if not allowed:
        return attach_order_ids(orders, execution_date), {
            "can_buy": False,
            "reason": timing_reason,
        }

    candidates = select_candidates(stocks, signal_date, surviving, config)
    slots = max(0, config.max_positions - len(surviving))
    portfolio_value = cash
    for code, position in holdings.items():
        close = _last_close(stocks.get(code), signal_date) if stocks.get(code) is not None else None
        if close is not None:
            portfolio_value += close * int(position.get("shares", 0))

    budget = min(cash, portfolio_value * config.max_position_pct)
    for code, factor_score in candidates[:slots]:
        close = _last_close(stocks[code], signal_date)
        if close is None:
            continue
        shares = int(budget / close / 100) * 100
        if shares <= 0:
            continue
        orders.append(
            {
                "signal_date": str(pd.Timestamp(signal_date).date()),
                "action": "BUY",
                "code": code,
                "estimated_shares": shares,
                "reference_close": close,
                "score": factor_score,
                "reason": "ranked_candidate",
            }
        )
    return attach_order_ids(orders, execution_date), {"can_buy": True, "reason": "ok"}
