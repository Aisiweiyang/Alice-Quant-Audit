"""Buffered-rank portfolio simulation for the cross-sectional factor model."""

from dataclasses import replace

import pandas as pd

from engine.backtest import (
    _build_trade_dates,
    _load_market_cache,
    _rebalance_dates,
    _row_on,
    calculate_metrics,
)
from engine.broker import Broker
from engine.execution import order_is_executable
from engine.portfolio import Portfolio
from strategy.cross_sectional import build_feature_store, rescore_feature_store
from strategy.factor_config import FactorStrategyConfig


def load_market(cache_path):
    return _load_market_cache(cache_path)


def prepare_execution_panels(data):
    """Pre-compute per-date buy/sell gate flags with the SAME rules as the
    main backtest, so factor research and production share one execution
    standard (suspension, zero volume, limit-up/down at the open)."""
    buy_flags = {}
    sell_flags = {}
    for code, frame in data.items():
        if frame is None or frame.empty:
            continue
        dates = pd.DatetimeIndex(frame["date"]).normalize()
        rows = frame.to_dict("records")
        buy_flags[code] = pd.Series(
            [order_is_executable(code, row, "BUY")[0] for row in rows], index=dates
        )
        sell_flags[code] = pd.Series(
            [order_is_executable(code, row, "SELL")[0] for row in rows], index=dates
        )
    return pd.DataFrame(buy_flags), pd.DataFrame(sell_flags)


def signal_dates_for_period(data, start_date, end_date, config):
    trade_dates = _build_trade_dates(data, start_date, end_date)
    rebalances = _rebalance_dates(
        trade_dates,
        config.rebalance_frequency,
        config.rebalance_day,
    )
    positions = {date: index for index, date in enumerate(trade_dates)}
    return [trade_dates[positions[date] - 1] for date in sorted(rebalances) if positions[date] > 0]


def prepare_feature_store(data, start_date, end_date, config=None, verbose=True):
    config = config or FactorStrategyConfig()
    signals = signal_dates_for_period(data, start_date, end_date, config)
    return build_feature_store(data, signals, config, verbose=verbose)


def prepare_price_panels(data):
    """Align all opening and closing prices once for repeated simulations."""
    opening = {}
    closing = {}
    for code, frame in data.items():
        if frame is None or frame.empty:
            continue
        # 基准指数等仅有close的序列不参与交易面板（无法开盘成交）。
        if "open" not in frame.columns or "close" not in frame.columns:
            continue
        dates = pd.DatetimeIndex(frame["date"])
        opening[code] = pd.Series(
            pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            index=dates,
        )
        closing[code] = pd.Series(
            pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            index=dates,
        )
    open_panel = pd.DataFrame(opening).sort_index()
    close_panel = pd.DataFrame(closing).sort_index().ffill()
    return open_panel, close_panel


def _benchmark(data, code, start_date, end_date):
    frame = data.get(code)
    if frame is None or frame.empty:
        raise ValueError(f"缓存缺少基准指数 {code}")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return frame[frame["date"].between(start, end)][["date", "close"]].copy()


def _market_timing_map(data, config):
    benchmark = data.get(config.benchmark_index)
    if benchmark is None or benchmark.empty:
        return {}
    series = benchmark[["date", "close"]].copy().sort_values("date")
    series["market_ma"] = series["close"].rolling(config.market_ma_period).mean()
    return {
        pd.Timestamp(row.date).normalize(): bool(row.close >= row.market_ma)
        for row in series.itertuples()
        if pd.notna(row.market_ma)
    }


def run_factor_backtest(
    data,
    start_date,
    end_date,
    config=None,
    feature_store=None,
    price_panels=None,
    execution_panels=None,
    verbose=True,
):
    """Run a point-in-time factor backtest with next-open execution.

    Orders pass the same suspension/limit checks as the main backtest;
    ``execution_panels`` may be precomputed via ``prepare_execution_panels``.
    """
    config = config or FactorStrategyConfig()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    trade_dates = _build_trade_dates(data, start, end)
    if not trade_dates:
        raise ValueError("指定区间没有交易日")

    rebalance_dates = _rebalance_dates(
        trade_dates,
        config.rebalance_frequency,
        config.rebalance_day,
    )
    if feature_store is None:
        feature_store = prepare_feature_store(data, start, end, config, verbose=verbose)
    else:
        feature_store = rescore_feature_store(feature_store, config.factor_weights)
    if price_panels is None:
        price_panels = prepare_price_panels(data)
    open_panel, close_panel = price_panels
    if execution_panels is None:
        execution_panels = prepare_execution_panels(data)
    buy_gate, sell_gate = execution_panels

    market_ok = _market_timing_map(data, config)
    broker = Broker(config)
    portfolio = Portfolio(
        config.initial_capital,
        broker,
        config.max_positions,
        config.max_position_pct,
    )
    # 挂到portfolio上供商业报告如实披露截面可用性（信号截面数、平均截面宽度）。
    portfolio.feature_store = feature_store
    previous_closes = None
    rejected_orders = {"buy_blocked": 0, "sell_blocked": 0}

    for index, date in enumerate(trade_dates):
        opens = open_panel.loc[date] if date in open_panel.index else pd.Series(dtype=float)
        closes = close_panel.loc[date] if date in close_panel.index else pd.Series(dtype=float)
        if date in buy_gate.index:
            buy_flags = buy_gate.loc[date]
            sell_flags = sell_gate.loc[date]
        else:
            buy_flags = pd.Series(dtype=bool)
            sell_flags = pd.Series(dtype=bool)

        def _sell_allowed(code):
            if code not in sell_flags.index:
                rejected_orders["sell_blocked"] += 1
                return False
            if bool(sell_flags.loc[code]):
                return True
            rejected_orders["sell_blocked"] += 1
            return False

        sold_today = set()
        for code in list(portfolio.holdings):
            if previous_closes is None:
                continue
            reference = previous_closes.get(code, float("nan"))
            open_price = opens.get(code, float("nan"))
            if pd.isna(reference) or pd.isna(open_price) or open_price <= 0:
                continue
            if not _sell_allowed(code):
                continue
            entry = portfolio.holdings[code]["avg_cost"]
            hit_stop = config.stop_loss_pct is not None and reference <= entry * (
                1.0 - config.stop_loss_pct
            )
            hit_target = config.take_profit_pct is not None and reference >= entry * (
                1.0 + config.take_profit_pct
            )
            if hit_stop or hit_target:
                portfolio.sell(code, float(open_price), date.strftime("%Y-%m-%d"))
                sold_today.add(code)

        if date in rebalance_dates and index > 0:
            signal_date = trade_dates[index - 1]
            ranked = feature_store.get(signal_date, pd.DataFrame())
            eligible = (
                ranked[ranked["score"] >= config.minimum_score] if not ranked.empty else ranked
            )
            ranked_codes = eligible["code"].tolist() if not eligible.empty else []
            exit_buffer = set(ranked_codes[: config.exit_rank])

            for code in list(portfolio.holdings):
                open_price = opens.get(code, float("nan"))
                if code in exit_buffer or pd.isna(open_price) or open_price <= 0:
                    continue
                buy_date = pd.Timestamp(portfolio.holdings[code]["buy_date"])
                if (date - buy_date).days < config.minimum_holding_days:
                    continue
                if not _sell_allowed(code):
                    continue
                portfolio.sell(code, float(open_price), date.strftime("%Y-%m-%d"))
                sold_today.add(code)

            allow_buys = not config.use_market_timing or market_ok.get(signal_date, False)
            if allow_buys:
                slots = config.max_positions - len(portfolio.holdings)
                portfolio_value = portfolio.total_value(closes)
                for code in ranked_codes:
                    if slots <= 0:
                        break
                    open_price = opens.get(code, float("nan"))
                    if (
                        code in portfolio.holdings
                        or code in sold_today
                        or pd.isna(open_price)
                        or open_price <= 0
                    ):
                        continue
                    if code not in buy_flags.index or not bool(buy_flags.loc[code]):
                        rejected_orders["buy_blocked"] += 1
                        continue
                    if portfolio.buy(
                        code,
                        float(open_price),
                        date.strftime("%Y-%m-%d"),
                        portfolio_value=portfolio_value,
                    ):
                        slots -= 1

        portfolio.record(date.strftime("%Y-%m-%d"), closes)
        previous_closes = closes

        if verbose and index % 100 == 0:
            value = portfolio.total_value(closes)
            print(f"    {date.date()} | 净值 {value:,.0f} | 持仓 {len(portfolio.holdings)}")

    final_date = trade_dates[-1]
    final_label = final_date.strftime("%Y-%m-%d")
    # 期末清仓使用与主回测相同的真实成交检查：停牌/跌停无法卖出。
    final_prices = (
        close_panel.loc[final_date] if final_date in close_panel.index else pd.Series(dtype=float)
    )
    final_rows = {
        code: row
        for code in list(portfolio.holdings)
        if (row := _row_on(data.get(code), final_date)) is not None
    }
    real_final_closes = {
        code: float(row["close"])
        for code, row in final_rows.items()
        if pd.notna(row.get("close")) and float(row["close"]) > 0
    }
    portfolio.close_all(final_label, real_final_closes, rows=final_rows)
    if portfolio.equity_curve and portfolio.equity_curve[-1]["date"] == final_label:
        portfolio.equity_curve.pop()
    portfolio.record(final_label, final_prices)
    portfolio.execution_audit = {
        **rejected_orders,
        "final_unsold": dict(portfolio.final_unsold),
    }

    benchmark = _benchmark(data, config.benchmark_index, start, end)
    metrics = calculate_metrics(portfolio, benchmark)
    return portfolio, metrics


def config_with(config, **changes):
    """Small convenience wrapper used by the research grid."""
    return replace(config, **changes)
