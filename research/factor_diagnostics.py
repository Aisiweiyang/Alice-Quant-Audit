"""Alpha158 单因子训练期诊断。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.alpha158_subset import COMPOSITE_DIRECTIONS, FEATURE_NAMES, FEATURE_SET_NAME, prefixed


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 5:
        return np.nan
    left_rank = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _forward_return(
    data: dict,
    code: str,
    signal_date: pd.Timestamp,
    days: int,
    cutoff: pd.Timestamp | None = None,
) -> float:
    frame = data.get(code)
    if frame is None or frame.empty:
        return np.nan
    dates = pd.DatetimeIndex(frame["date"])
    position = int(dates.searchsorted(signal_date, side="right") - 1)
    if position < 0 or position + days >= len(frame) or dates[position].normalize() != signal_date:
        return np.nan
    target_date = dates[position + days].normalize()
    if cutoff is not None and target_date > cutoff:
        return np.nan
    current = float(frame.iloc[position]["close"])
    future = float(frame.iloc[position + days]["close"])
    if not np.isfinite(current) or not np.isfinite(future) or current <= 0:
        return np.nan
    return future / current - 1.0


def _quintile_spread(values: np.ndarray, returns: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(returns)
    count = int(valid.sum())
    if count < 10:
        return np.nan
    ordered = np.argsort(values[valid], kind="mergesort")
    sample = returns[valid][ordered]
    size = max(count // 5, 1)
    return float(np.mean(sample[-size:]) - np.mean(sample[:size]))


def diagnose_feature_store(feature_store: dict, data: dict, train_end, forward_days: int = 5):
    """按信号日计算 RankIC，再汇总。"""
    cutoff = pd.Timestamp(train_end).normalize()
    daily_rows = []
    for signal_date, frame in sorted(feature_store.items()):
        signal_date = pd.Timestamp(signal_date).normalize()
        if signal_date > cutoff or frame is None or frame.empty:
            continue
        current = frame.copy()
        current["forward_return"] = [
            _forward_return(data, code, signal_date, forward_days, cutoff=cutoff)
            for code in current["code"]
        ]
        returns = current["forward_return"].to_numpy(dtype=float)
        return_coverage = float(np.isfinite(returns).mean())
        for name in FEATURE_NAMES:
            column = prefixed(name)
            if column not in current:
                continue
            values = pd.to_numeric(current[column], errors="coerce").to_numpy(dtype=float)
            coverage = float(np.isfinite(values).mean())
            raw_ic = _rank_correlation(values, returns)
            raw_spread = _quintile_spread(values, returns)
            direction = int(COMPOSITE_DIRECTIONS.get(name, 0))
            daily_rows.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "feature": name,
                    "coverage_pct": coverage * 100.0,
                    "forward_return_coverage_pct": return_coverage * 100.0,
                    "rank_ic": raw_ic,
                    "top_bottom_spread_pct": raw_spread * 100.0
                    if np.isfinite(raw_spread)
                    else np.nan,
                    "expected_direction": direction,
                    "direction_adjusted_rank_ic": raw_ic * direction
                    if direction and np.isfinite(raw_ic)
                    else np.nan,
                }
            )

    daily = pd.DataFrame(daily_rows)
    summaries = []
    if not daily.empty:
        for name, group in daily.groupby("feature", sort=False):
            ic = group["rank_ic"].dropna()
            adjusted = group["direction_adjusted_rank_ic"].dropna()
            spread = group["top_bottom_spread_pct"].dropna()
            ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
            summaries.append(
                {
                    "feature": name,
                    "dates": int(len(group)),
                    "coverage_pct": float(group["coverage_pct"].mean()),
                    "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
                    "rank_ic_std": ic_std,
                    "rank_icir": float(ic.mean() / ic_std) if len(ic) and ic_std > 0 else 0.0,
                    "positive_ic_rate_pct": float((ic > 0).mean() * 100.0) if len(ic) else np.nan,
                    "mean_top_bottom_spread_pct": float(spread.mean()) if len(spread) else np.nan,
                    "expected_direction": int(COMPOSITE_DIRECTIONS.get(name, 0)),
                    "mean_direction_adjusted_rank_ic": float(adjusted.mean())
                    if len(adjusted)
                    else np.nan,
                }
            )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(
            "mean_direction_adjusted_rank_ic", ascending=False, na_position="last"
        )
    return daily, summary


def write_factor_diagnostics(
    feature_store: dict, data: dict, train_end, output_dir, forward_days: int = 5
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily, summary = diagnose_feature_store(feature_store, data, train_end, forward_days)
    daily.to_csv(output / "alpha158_daily_rank_ic.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "alpha158_factor_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "feature_set": FEATURE_SET_NAME,
        "selection_boundary": f"仅使用信号日不晚于 {pd.Timestamp(train_end).date()} 的训练数据",
        "forward_return_days": forward_days,
        "dates": int(daily["signal_date"].nunique()) if not daily.empty else 0,
        "features": int(daily["feature"].nunique()) if not daily.empty else 0,
        "rank_ic_definition": "当日截面因子百分位与未来N个交易日收益百分位的Pearson相关",
        "icir_definition": "各信号日RankIC均值除以样本标准差，未年化",
    }
    (output / "alpha158_diagnostics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return daily, summary
