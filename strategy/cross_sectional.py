"""截面因子计算与排名。"""

import numpy as np
import pandas as pd

from strategy.alpha158_subset import (
    COMPOSITE_DIRECTIONS,
    FEATURE_NAMES,
    compute_alpha158_subset,
    last_feature_values,
    prefixed,
)


RANK_COLUMNS = (
    "momentum_rank",
    "trend_quality_rank",
    "value_rank",
    "low_volatility_rank",
    "volume_confirmation_rank",
    "short_reversal_rank",
    "alpha158_pv_rank",
    "alpha158_ridge_rank",
)


def is_main_board_stock(code):
    """Return True for Shanghai/Shenzhen main-board stock codes."""
    if not isinstance(code, str) or "." not in code:
        return False
    exchange, tail = code.split(".", 1)
    if exchange == "sh":
        return tail.startswith(("600", "601", "603", "605"))
    if exchange == "sz":
        return tail.startswith(("000", "001", "002", "003"))
    return False


def _safe_corr(left, right):
    if len(left) != len(right) or len(left) < 5:
        return np.nan
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _trend_quality(close_window):
    """Annualized log-price slope multiplied by regression R-squared."""
    log_price = np.log(close_window)
    x = np.arange(len(log_price), dtype=float)
    x -= x.mean()
    centered = log_price - log_price.mean()
    denominator = float(np.dot(x, x))
    if denominator == 0:
        return np.nan
    slope = float(np.dot(x, centered) / denominator)
    fitted = slope * x
    total_ss = float(np.dot(centered, centered))
    residual_ss = float(np.dot(centered - fitted, centered - fitted))
    r_squared = 0.0 if total_ss == 0 else max(0.0, 1.0 - residual_ss / total_ss)
    return slope * 252.0 * r_squared


def factor_row(code, frame, signal_date, minimum_history=65):
    """Compute raw factors using rows available at the signal close only."""
    if not is_main_board_stock(code) or frame is None or frame.empty:
        return None

    target = pd.Timestamp(signal_date).normalize()
    dates = (
        frame.index
        if isinstance(frame.index, pd.DatetimeIndex)
        else pd.DatetimeIndex(frame["date"])
    )
    position = int(dates.searchsorted(target, side="right") - 1)
    if position < minimum_history - 1 or position < 63 or dates[position].normalize() != target:
        return None

    start = position - 64
    recent = frame.iloc[start : position + 1]
    close = pd.to_numeric(recent["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(recent["volume"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(recent["amount"], errors="coerce").to_numpy(dtype=float)
    latest = recent.iloc[-1]

    if len(close) < 65 or not np.isfinite(close).all() or np.any(close <= 0):
        return None
    if not np.isfinite(volume[-21:]).all() or np.any(volume[-21:] <= 0):
        return None

    latest_open = float(latest.get("open", np.nan))
    latest_pct = float(latest.get("pctChg", np.nan))
    if not np.isfinite(latest_open) or latest_open <= 0:
        return None
    if not np.isfinite(latest_pct) or abs(latest_pct) >= 9.8:
        return None

    daily_returns = np.diff(np.log(close[-21:]))
    volume_changes = np.diff(np.log(volume[-21:]))
    volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
    amount_20 = float(np.nanmedian(amount[-20:]))
    if not np.isfinite(amount_20) or amount_20 <= 0:
        return None

    pe = float(latest.get("pe", np.nan))
    pb = float(latest.get("pb", np.nan))
    earnings_yield = 1.0 / pe if np.isfinite(pe) and 0 < pe <= 150 else np.nan
    book_to_price = 1.0 / pb if np.isfinite(pb) and 0 < pb <= 30 else np.nan
    if not np.isfinite(earnings_yield) and not np.isfinite(book_to_price):
        return None

    row = {
        "code": code,
        "amount_20": amount_20,
        "above_ma60": bool(close[-1] >= np.mean(close[-60:])),
        "momentum_60_5": float(close[-6] / close[-61] - 1.0),
        "momentum_20": float(close[-1] / close[-21] - 1.0),
        "short_reversal": float(-(close[-1] / close[-6] - 1.0)),
        "volatility_20": volatility,
        "trend_quality": _trend_quality(close[-60:]),
        "volume_confirmation": _safe_corr(daily_returns, volume_changes),
        "earnings_yield": earnings_yield,
        "book_to_price": book_to_price,
        "close": float(close[-1]),
        "pe": pe,
        "pb": pb,
    }
    row.update(last_feature_values(recent))
    return row


def _rolling_trend_quality(close, window=60):
    result = np.full(len(close), np.nan, dtype=float)
    if len(close) < window:
        return result
    with np.errstate(divide="ignore", invalid="ignore"):
        log_price = np.log(close)
    windows = np.lib.stride_tricks.sliding_window_view(log_price, window)
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    slope = windows @ x / denominator
    centered = windows - windows.mean(axis=1, keepdims=True)
    total_ss = np.sum(centered * centered, axis=1)
    explained_ss = slope * slope * denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.clip(explained_ss / total_ss, 0.0, 1.0)
    quality = slope * 252.0 * r_squared
    quality[~np.isfinite(windows).all(axis=1)] = np.nan
    result[window - 1 :] = quality
    return result


def _rolling_correlation(left, right, window=20):
    result = np.full(len(left) + 1, np.nan, dtype=float)
    if len(left) < window:
        return result
    left_windows = np.lib.stride_tricks.sliding_window_view(left, window)
    right_windows = np.lib.stride_tricks.sliding_window_view(right, window)
    left_centered = left_windows - left_windows.mean(axis=1, keepdims=True)
    right_centered = right_windows - right_windows.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = numerator / denominator
    correlation[denominator == 0] = 0.0
    valid = np.isfinite(left_windows).all(axis=1) & np.isfinite(right_windows).all(axis=1)
    correlation[~valid] = np.nan
    result[window:] = correlation
    return result


def _factor_arrays(frame):
    """Vectorized raw factors for every row of one security."""
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype=float)
    open_price = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    pct_change = pd.to_numeric(frame.get("pctChg"), errors="coerce").to_numpy(dtype=float)
    pe = pd.to_numeric(frame.get("pe"), errors="coerce").to_numpy(dtype=float)
    pb = pd.to_numeric(frame.get("pb"), errors="coerce").to_numpy(dtype=float)

    close_series = pd.Series(close)
    log_return = np.diff(np.log(close))
    log_volume_change = np.diff(np.log(np.where(volume > 0, volume, np.nan)))
    result = {
        "open": open_price,
        "close": close,
        "volume": volume,
        "pct_change": pct_change,
        "pe": pe,
        "pb": pb,
        "amount_20": pd.Series(amount).rolling(20).median().to_numpy(),
        "ma60": close_series.rolling(60).mean().to_numpy(),
        "momentum_60_5": close_series.shift(5).div(close_series.shift(60)).sub(1).to_numpy(),
        "momentum_20": close_series.div(close_series.shift(20)).sub(1).to_numpy(),
        "short_reversal": close_series.div(close_series.shift(5)).sub(1).mul(-1).to_numpy(),
        "volatility_20": pd.Series(log_return).rolling(20).std().mul(np.sqrt(252)).to_numpy(),
        "trend_quality": _rolling_trend_quality(close),
        "volume_confirmation": _rolling_correlation(log_return, log_volume_change),
    }
    # 收益序列少一行，补到原行情索引。
    result["volatility_20"] = np.insert(result["volatility_20"], 0, np.nan)
    alpha158 = compute_alpha158_subset(frame)
    for name in FEATURE_NAMES:
        result[prefixed(name)] = alpha158[name].to_numpy(dtype=float)
    return result


def rank_cross_section(rows, weights, universe_size=500, require_above_ma60=True):
    """Apply liquidity eligibility, percentile ranks, and a weighted score."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("amount_20", ascending=False).head(universe_size).copy()
    if require_above_ma60:
        frame = frame[frame["above_ma60"]].copy()
    if frame.empty:
        return frame

    frame["momentum"] = 0.70 * frame["momentum_60_5"] + 0.30 * frame["momentum_20"]
    frame["momentum_rank"] = frame["momentum"].rank(pct=True)
    frame["trend_quality_rank"] = frame["trend_quality"].rank(pct=True)
    frame["low_volatility_rank"] = frame["volatility_20"].rank(pct=True, ascending=False)
    frame["volume_confirmation_rank"] = frame["volume_confirmation"].rank(pct=True)
    frame["short_reversal_rank"] = frame["short_reversal"].rank(pct=True)

    alpha158_parts = []
    for name, direction in COMPOSITE_DIRECTIONS.items():
        column = prefixed(name)
        if column in frame:
            alpha158_parts.append(
                frame[column].rank(pct=True, ascending=direction > 0).rename(name)
            )
    frame["alpha158_pv_rank"] = (
        pd.concat(alpha158_parts, axis=1).mean(axis=1, skipna=True) if alpha158_parts else 0.5
    )
    # Ridge 训练完成前保持中性分。
    frame["alpha158_ridge_rank"] = 0.5

    value_parts = pd.concat(
        [
            frame["earnings_yield"].rank(pct=True),
            frame["book_to_price"].rank(pct=True),
        ],
        axis=1,
    )
    frame["value_rank"] = value_parts.mean(axis=1, skipna=True)

    frame[list(RANK_COLUMNS)] = frame[list(RANK_COLUMNS)].fillna(0.5)
    frame["score"] = (
        sum(frame[column] * float(weights.get(column, 0.0)) for column in RANK_COLUMNS) * 100.0
    )
    return frame.sort_values(["score", "amount_20"], ascending=False).reset_index(drop=True)


def build_feature_store(data, signal_dates, config, verbose=True):
    """Build one point-in-time ranked cross-section for each rebalance signal."""
    store = {}
    stock_items = [(code, frame) for code, frame in data.items() if is_main_board_stock(code)]
    dates = sorted({pd.Timestamp(date).normalize() for date in signal_dates})
    rows_by_date = {date: [] for date in dates}
    signal_array = np.asarray(dates, dtype="datetime64[ns]")
    for index, (code, frame) in enumerate(stock_items, start=1):
        frame_dates = pd.DatetimeIndex(frame["date"]).to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(frame_dates, signal_array, side="right") - 1
        factors = _factor_arrays(frame)
        for signal_date, signal_value, position in zip(dates, signal_array, positions):
            if position < config.minimum_history - 1 or position < 0:
                continue
            if frame_dates[position] != signal_value:
                continue
            latest_open = factors["open"][position]
            latest_close = factors["close"][position]
            latest_volume = factors["volume"][position]
            latest_pct = factors["pct_change"][position]
            amount_20 = factors["amount_20"][position]
            required = (
                latest_open,
                latest_close,
                latest_volume,
                latest_pct,
                amount_20,
                factors["momentum_60_5"][position],
                factors["momentum_20"][position],
                factors["short_reversal"][position],
                factors["volatility_20"][position],
                factors["trend_quality"][position],
            )
            if not np.isfinite(required).all():
                continue
            if latest_open <= 0 or latest_close <= 0 or latest_volume <= 0 or amount_20 <= 0:
                continue
            if abs(latest_pct) >= 9.8:
                continue

            pe = factors["pe"][position]
            pb = factors["pb"][position]
            earnings_yield = 1.0 / pe if np.isfinite(pe) and 0 < pe <= 150 else np.nan
            book_to_price = 1.0 / pb if np.isfinite(pb) and 0 < pb <= 30 else np.nan
            if not np.isfinite(earnings_yield) and not np.isfinite(book_to_price):
                continue

            row = {
                "code": code,
                "amount_20": float(amount_20),
                "above_ma60": bool(latest_close >= factors["ma60"][position]),
                "momentum_60_5": float(factors["momentum_60_5"][position]),
                "momentum_20": float(factors["momentum_20"][position]),
                "short_reversal": float(factors["short_reversal"][position]),
                "volatility_20": float(factors["volatility_20"][position]),
                "trend_quality": float(factors["trend_quality"][position]),
                "volume_confirmation": float(factors["volume_confirmation"][position])
                if np.isfinite(factors["volume_confirmation"][position])
                else 0.0,
                "earnings_yield": earnings_yield,
                "book_to_price": book_to_price,
                "close": float(latest_close),
                "pe": float(pe),
                "pb": float(pb),
            }
            row.update(
                {prefixed(name): float(factors[prefixed(name)][position]) for name in FEATURE_NAMES}
            )
            rows_by_date[signal_date].append(row)
        if verbose and (index == 1 or index % 500 == 0 or index == len(stock_items)):
            print(f"    股票因子 {index}/{len(stock_items)}")

    for index, signal_date in enumerate(dates, start=1):
        store[signal_date] = rank_cross_section(
            rows_by_date[signal_date],
            config.factor_weights,
            config.liquid_universe_size,
            config.require_price_above_ma60,
        )
        if verbose and (index == 1 or index % 20 == 0 or index == len(dates)):
            print(f"    截面排名 {index}/{len(dates)}: {signal_date.date()}")
    return store


def rescore_feature_store(store, weights):
    """Recombine existing rank columns without recalculating raw factors."""
    rescored = {}
    for date, frame in store.items():
        current = frame.copy()
        if not current.empty:
            current["score"] = (
                sum(current[column] * float(weights.get(column, 0.0)) for column in RANK_COLUMNS)
                * 100.0
            )
            current = current.sort_values(["score", "amount_20"], ascending=False).reset_index(
                drop=True
            )
        rescored[date] = current
    return rescored
