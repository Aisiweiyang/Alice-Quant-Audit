"""Shared selection and market-timing rules for backtests and paper trading."""

from __future__ import annotations

import pandas as pd

from data.cleaner import filter_tradable, prepare_universe
from strategy.config import StrategyConfig as cfg
from strategy.factors import compute_all, score

# filter_tradable 的默认历史长度要求；快速路径与其保持同一口径。
_FILTER_LOOKBACK_DAYS = 60


def passes_strategy_filters(hist, config=cfg):
    days = config.consecutive_up_days
    stock_ma_period = getattr(config, "stock_ma_period", 50)
    if hist is None or len(hist) < max(stock_ma_period, 20, days):
        return False

    recent = pd.to_numeric(hist["pctChg"], errors="coerce").tail(days)
    if len(recent) < days or recent.isna().any():
        return False
    if not ((recent > 0) & (recent <= config.max_daily_gain_pct)).all():
        return False

    closes = pd.to_numeric(hist["close"], errors="coerce")
    ma50 = closes.rolling(stock_ma_period).mean().iloc[-1]
    if pd.isna(ma50) or closes.iloc[-1] < ma50:
        return False

    if getattr(config, "stock_ma_crossover", False):
        ma20 = closes.rolling(20).mean().iloc[-1]
        if pd.isna(ma20) or ma20 <= ma50:
            return False

    lookback = getattr(config, "anti_falling_lookback", 3)
    if getattr(config, "anti_falling_knife", False) and len(closes) >= lookback + 1:
        drop = closes.iloc[-1] / closes.iloc[-(lookback + 1)] - 1
        if drop < -getattr(config, "anti_falling_threshold", 0.05):
            return False

    latest = hist.iloc[-1]
    pe = pd.to_numeric(pd.Series([latest.get("pe")]), errors="coerce").iloc[0]
    pb = pd.to_numeric(pd.Series([latest.get("pb")]), errors="coerce").iloc[0]
    if config.use_pe_filter and pd.notna(pe) and (pe <= config.pe_min or pe > config.pe_max):
        return False
    if config.use_pb_filter and pd.notna(pb) and (pb <= config.pb_min or pb > config.pb_max):
        return False
    return True


def _rank_liquid(liquid, config):
    """按成交额排序并对前top_volume_n只评分（新旧路径共用的尾部逻辑）。"""
    liquid.sort(key=lambda item: item[2], reverse=True)
    candidates = []
    for code, hist, _ in liquid[: config.top_volume_n]:
        factor_score = score(compute_all(hist, config), config.factor_weights)
        latest_pe = pd.to_numeric(pd.Series([hist.iloc[-1].get("pe")]), errors="coerce").iloc[0]
        if pd.notna(latest_pe) and latest_pe > 0:
            factor_score += max(0, (config.pe_max - latest_pe) / config.pe_max) * 1.5
        factor_score = round(factor_score, 2)
        if factor_score >= config.min_score:
            candidates.append((code, factor_score))
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _tradable_with_index(index, target_value):
    """数组级复刻 data.cleaner.filter_tradable（默认lookback=60）的可交易判定。"""
    if index.n_rows < _FILTER_LOOKBACK_DAYS:
        return False
    n_hist = index.count_leq(target_value)
    if n_hist < _FILTER_LOOKBACK_DAYS:
        return False
    last_date = pd.Timestamp(index.dates[n_hist - 1]).normalize()
    if last_date != pd.Timestamp(target_value, unit="ns").normalize():
        return False
    trade_status = index.coerced_filled("tradestatus", 1)
    if trade_status is not None and trade_status[n_hist - 1] == 0:
        return False
    volume = index.column("volume")
    # 与 filter_tradable 的 .get('volume', 0)==0 一致：缺列视同停牌；NaN!=0 放行。
    if volume is None or volume[n_hist - 1] == 0:
        return False
    is_st = index.coerced_filled("isST", 0)
    if is_st is not None and is_st[n_hist - 1] == 1:
        return False
    return True


def _select_candidates_fast(data, signal_date, held_codes, config, indexes):
    """快速路径：日期定位与前置过滤走数组，评分尾部与原实现共用。"""
    target = pd.Timestamp(signal_date)
    target_value = target.value
    normalized_value = target.normalize().value
    liquid = []
    for code, frame in data.items():
        if code in held_codes:
            continue
        index = indexes.get(code)
        if index is not None:
            if not _tradable_with_index(index, normalized_value):
                continue
            n_hist = index.count_leq(target_value)
            hist = frame.iloc[:n_hist]
            if not index.passes_leading_filters(n_hist, config):
                continue
            if not passes_strategy_filters(hist, config):
                continue
            amount_arr = index.coerced_filled("amount", 0)
            amount = float(amount_arr[n_hist - 1]) if amount_arr is not None else 0.0
        else:
            ok, _ = filter_tradable(frame, target)
            if not ok:
                continue
            hist = frame.loc[frame["date"] <= target]
            if not passes_strategy_filters(hist, config):
                continue
            amount = (
                pd.to_numeric(pd.Series([hist.iloc[-1].get("amount", 0)]), errors="coerce")
                .fillna(0)
                .iloc[0]
            )
            amount = float(amount)
        liquid.append((code, hist, amount))
    return _rank_liquid(liquid, config)


def select_candidates(data, signal_date, held_codes=(), config=cfg, indexes=None):
    """Rank candidates using only data available on or before ``signal_date``.

    ``indexes`` 为 engine.fast_index.build_stock_indexes 的产物；提供时走
    数组化快速路径（结果与原实现完全一致，等价性见 tests/test_performance_round.py）。
    """
    if indexes:
        return _select_candidates_fast(data, signal_date, held_codes, config, indexes)
    liquid = []
    target = pd.Timestamp(signal_date)
    for code in prepare_universe(data, target):
        if code in held_codes:
            continue
        hist = data[code].loc[data[code]["date"] <= target]
        if not passes_strategy_filters(hist, config):
            continue
        amount = (
            pd.to_numeric(pd.Series([hist.iloc[-1].get("amount", 0)]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        liquid.append((code, hist, float(amount)))
    return _rank_liquid(liquid, config)


def market_allows_buy(benchmark, signal_date, config=cfg):
    """Apply the same close-based market timing rule used by the backtest."""
    if not config.use_market_timing:
        return True, "market_timing_disabled"
    if benchmark is None or benchmark.empty:
        return False, "benchmark_missing"

    target = pd.Timestamp(signal_date).normalize()
    frame = benchmark.loc[benchmark["date"] <= target, ["date", "close"]].copy()
    frame = frame.sort_values("date")
    if len(frame) < config.ma_period:
        return False, "benchmark_warmup"
    closes = pd.to_numeric(frame["close"], errors="coerce")
    ma = closes.rolling(config.ma_period).mean()
    if pd.isna(ma.iloc[-1]) or closes.iloc[-1] < ma.iloc[-1]:
        return False, f"benchmark_below_ma{config.ma_period}"

    slope_lookback = getattr(config, "ma_slope_lookback", 0)
    if slope_lookback > 0:
        valid_ma = ma.dropna()
        if len(valid_ma) < slope_lookback:
            return False, "benchmark_slope_warmup"
        recent = valid_ma.tail(slope_lookback)
        if recent.iloc[-1] < recent.iloc[0]:
            return False, "benchmark_ma_falling"
    return True, "ok"
