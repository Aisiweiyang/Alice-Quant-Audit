"""Qlib Alpha158 的行情量价子集。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_SET_NAME = "alpha158_price_volume_subset_v1"
OFFICIAL_SOURCE = "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py"
WINDOWS = (5, 10, 20, 60)
ROLLING_OPERATORS = ("ROC", "MA", "STD", "BETA", "RSQR", "RSV", "CORR", "VMA")
EPSILON = 1e-12


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    formula: str
    inputs: tuple[str, ...]
    window: int
    family: str
    orientation: str = "raw"


def feature_specs() -> tuple[FeatureSpec, ...]:
    """特征公式和字段依赖。"""
    specs = [
        FeatureSpec("KMID", "(close-open)/open", ("open", "close"), 1, "kbar"),
        FeatureSpec("KLEN", "(high-low)/open", ("open", "high", "low"), 1, "kbar"),
        FeatureSpec(
            "KMID2", "(close-open)/(high-low+1e-12)", ("open", "high", "low", "close"), 1, "kbar"
        ),
        FeatureSpec("KUP", "(high-max(open,close))/open", ("open", "high", "close"), 1, "kbar"),
        FeatureSpec(
            "KUP2",
            "(high-max(open,close))/(high-low+1e-12)",
            ("open", "high", "low", "close"),
            1,
            "kbar",
        ),
        FeatureSpec("KLOW", "(min(open,close)-low)/open", ("open", "low", "close"), 1, "kbar"),
        FeatureSpec(
            "KLOW2",
            "(min(open,close)-low)/(high-low+1e-12)",
            ("open", "high", "low", "close"),
            1,
            "kbar",
        ),
        FeatureSpec("KSFT", "(2*close-high-low)/open", ("open", "high", "low", "close"), 1, "kbar"),
        FeatureSpec(
            "KSFT2",
            "(2*close-high-low)/(high-low+1e-12)",
            ("open", "high", "low", "close"),
            1,
            "kbar",
        ),
        FeatureSpec("OPEN0", "open/close", ("open", "close"), 1, "price"),
        FeatureSpec("HIGH0", "high/close", ("high", "close"), 1, "price"),
        FeatureSpec("LOW0", "low/close", ("low", "close"), 1, "price"),
        FeatureSpec(
            "VWAP0",
            "(amount/volume)/close",
            ("amount", "volume", "close"),
            1,
            "price",
            "vwap_proxy",
        ),
    ]
    formulas = {
        "ROC": ("Ref(close,{w})/close", ("close",)),
        "MA": ("Mean(close,{w})/close", ("close",)),
        "STD": ("Std(close,{w})/close", ("close",)),
        "BETA": ("Slope(close,{w})/close", ("close",)),
        "RSQR": ("Rsquare(close,{w})", ("close",)),
        "RSV": (
            "(close-Min(low,{w}))/(Max(high,{w})-Min(low,{w})+1e-12)",
            ("close", "high", "low"),
        ),
        "CORR": ("Corr(close,Log(volume+1),{w})", ("close", "volume")),
        "VMA": ("Mean(volume,{w})/(volume+1e-12)", ("volume",)),
    }
    for operator in ROLLING_OPERATORS:
        formula, inputs = formulas[operator]
        for window in WINDOWS:
            specs.append(
                FeatureSpec(
                    f"{operator}{window}", formula.format(w=window), inputs, window, "rolling"
                )
            )
    return tuple(specs)


FEATURE_NAMES = tuple(spec.name for spec in feature_specs())

# 仅供手工组合实验使用。1 表示值越大排名越高，-1 相反。
COMPOSITE_DIRECTIONS = {
    "KMID": 1,
    "KSFT2": 1,
    "ROC20": -1,
    "MA20": -1,
    "BETA20": 1,
    "RSQR20": 1,
    "RSV20": 1,
    "VMA20": -1,
}


def prefixed(name: str) -> str:
    return f"alpha158_{name.lower()}"


def feature_manifest() -> dict:
    return {
        "feature_set": FEATURE_SET_NAME,
        "official_source": OFFICIAL_SOURCE,
        "feature_count": len(FEATURE_NAMES),
        "windows": list(WINDOWS),
        "operators": list(ROLLING_OPERATORS),
        "vwap_note": "Baostock缓存没有独立VWAP字段，使用amount/volume作为日VWAP代理。",
        "future_data_policy": "仅使用当前行、shift和向后滚动窗口；不使用负向shift。",
        "composite_directions": COMPOSITE_DIRECTIONS,
        "features": [asdict(spec) for spec in feature_specs()],
    }


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _rolling_regression(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.full(len(values), np.nan, dtype=float)
    r_squared = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return slopes, r_squared
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    centered = windows - windows.mean(axis=1, keepdims=True)
    slope = centered @ x / denominator
    total_ss = np.sum(centered * centered, axis=1)
    explained_ss = slope * slope * denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        rsqr = np.where(total_ss > 0, explained_ss / total_ss, 0.0)
    valid = np.isfinite(windows).all(axis=1)
    slope[~valid] = np.nan
    rsqr[~valid] = np.nan
    slopes[window - 1 :] = slope
    r_squared[window - 1 :] = np.clip(rsqr, 0.0, 1.0)
    return slopes, r_squared


def compute_alpha158_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """计算 45 个量价特征，方向与 Qlib 原公式一致。"""
    open_price = _numeric(frame, "open")
    high = _numeric(frame, "high")
    low = _numeric(frame, "low")
    close = _numeric(frame, "close")
    volume = _numeric(frame, "volume")
    amount = _numeric(frame, "amount")
    result = pd.DataFrame(index=frame.index)

    price_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        result["KMID"] = (close - open_price) / open_price
        result["KLEN"] = price_range / open_price
        result["KMID2"] = (close - open_price) / (price_range + EPSILON)
        result["KUP"] = (high - np.maximum(open_price, close)) / open_price
        result["KUP2"] = (high - np.maximum(open_price, close)) / (price_range + EPSILON)
        result["KLOW"] = (np.minimum(open_price, close) - low) / open_price
        result["KLOW2"] = (np.minimum(open_price, close) - low) / (price_range + EPSILON)
        result["KSFT"] = (2.0 * close - high - low) / open_price
        result["KSFT2"] = (2.0 * close - high - low) / (price_range + EPSILON)
        result["OPEN0"] = open_price / close
        result["HIGH0"] = high / close
        result["LOW0"] = low / close
        vwap = np.where(volume > 0, amount / volume, np.nan)
        result["VWAP0"] = vwap / close

    close_series = pd.Series(close, index=frame.index)
    high_series = pd.Series(high, index=frame.index)
    low_series = pd.Series(low, index=frame.index)
    volume_series = pd.Series(volume, index=frame.index)
    log_volume = pd.Series(np.log(np.where(volume >= 0, volume + 1.0, np.nan)), index=frame.index)
    for window in WINDOWS:
        close_roll = close_series.rolling(window, min_periods=window)
        volume_roll = volume_series.rolling(window, min_periods=window)
        slope, rsqr = _rolling_regression(close, window)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"ROC{window}"] = close_series.shift(window) / close_series
            result[f"MA{window}"] = close_roll.mean() / close_series
            result[f"STD{window}"] = close_roll.std(ddof=1) / close_series
            result[f"BETA{window}"] = slope / close
            result[f"RSQR{window}"] = rsqr
            rolling_low = low_series.rolling(window, min_periods=window).min()
            rolling_high = high_series.rolling(window, min_periods=window).max()
            result[f"RSV{window}"] = (close_series - rolling_low) / (
                rolling_high - rolling_low + EPSILON
            )
            result[f"CORR{window}"] = close_series.rolling(window, min_periods=window).corr(
                log_volume
            )
            result[f"VMA{window}"] = volume_roll.mean() / (volume_series + EPSILON)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.loc[:, FEATURE_NAMES]


def last_feature_values(
    frame: pd.DataFrame, names: Iterable[str] = FEATURE_NAMES
) -> dict[str, float]:
    """取最后一个交易日的特征值。"""
    computed = compute_alpha158_subset(frame)
    if computed.empty:
        return {prefixed(name): np.nan for name in names}
    latest = computed.iloc[-1]
    return {prefixed(name): float(latest.get(name, np.nan)) for name in names}
