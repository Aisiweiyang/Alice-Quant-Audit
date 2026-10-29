"""Alpha158 量价特征的 Ridge 基线。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.alpha158_subset import FEATURE_NAMES, FEATURE_SET_NAME, prefixed


RIDGE_ALPHA = 100.0
FORWARD_DAYS = 5
MIN_TRAINING_DATES = 12
RANK_SCALE = float(np.sqrt(12.0))


def _feature_matrix(frame: pd.DataFrame, names=FEATURE_NAMES) -> np.ndarray:
    columns = []
    for name in names:
        column = prefixed(name)
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            if column in frame
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        ranks = values.rank(method="average", pct=True).fillna(0.5)
        columns.append((ranks.to_numpy(dtype=float) - 0.5) * RANK_SCALE)
    return np.column_stack(columns)


def _forward_info(data: dict, code: str, signal_date: pd.Timestamp, days: int):
    frame = data.get(code)
    if frame is None or frame.empty:
        return np.nan, None
    dates = pd.DatetimeIndex(frame["date"])
    position = int(dates.searchsorted(signal_date, side="right") - 1)
    if position < 0 or position + days >= len(frame) or dates[position].normalize() != signal_date:
        return np.nan, None
    current = float(frame.iloc[position]["close"])
    future = float(frame.iloc[position + days]["close"])
    if not np.isfinite(current) or not np.isfinite(future) or current <= 0:
        return np.nan, None
    return future / current - 1.0, dates[position + days].normalize()


def _training_samples(feature_store: dict, data: dict, cutoff: pd.Timestamp, forward_days: int):
    samples = []
    for signal_date, frame in sorted(feature_store.items()):
        signal_date = pd.Timestamp(signal_date).normalize()
        if signal_date > cutoff or frame is None or frame.empty:
            continue
        outcomes = [_forward_info(data, code, signal_date, forward_days) for code in frame["code"]]
        returns = np.asarray([item[0] for item in outcomes], dtype=float)
        target_dates = [item[1] for item in outcomes if item[1] is not None]
        if not target_dates or max(target_dates) > cutoff:
            continue
        valid = np.isfinite(returns)
        if int(valid.sum()) < max(50, len(FEATURE_NAMES) + 1):
            continue
        target_rank = (
            pd.Series(returns[valid]).rank(method="average", pct=True).to_numpy(dtype=float)
        )
        target = (target_rank - 0.5) * RANK_SCALE
        matrix = _feature_matrix(frame)[valid]
        samples.append(
            {
                "signal_date": signal_date,
                "target_date": max(target_dates),
                "rows": int(len(target)),
                "sum_x": matrix.sum(axis=0),
                "sum_y": float(target.sum()),
                "xtx": matrix.T @ matrix,
                "xty": matrix.T @ target,
            }
        )
    return samples


def _fit(samples: list[dict], alpha: float) -> tuple[np.ndarray, float, int]:
    if not samples:
        return np.zeros(len(FEATURE_NAMES), dtype=float), 0.0, 0
    rows = sum(sample["rows"] for sample in samples)
    sum_x = sum((sample["sum_x"] for sample in samples), np.zeros(len(FEATURE_NAMES)))
    sum_y = sum(sample["sum_y"] for sample in samples)
    xtx = sum(
        (sample["xtx"] for sample in samples), np.zeros((len(FEATURE_NAMES), len(FEATURE_NAMES)))
    )
    xty = sum((sample["xty"] for sample in samples), np.zeros(len(FEATURE_NAMES)))
    x_mean = sum_x / rows
    y_mean = float(sum_y / rows)
    centered_xtx = xtx - rows * np.outer(x_mean, x_mean)
    centered_xty = xty - rows * x_mean * y_mean
    system = centered_xtx + float(alpha) * np.eye(len(FEATURE_NAMES))
    try:
        coefficients = np.linalg.solve(system, centered_xty)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, centered_xty, rcond=None)[0]
    intercept = y_mean - float(x_mean @ coefficients)
    return coefficients, intercept, rows


def _score_frame(frame: pd.DataFrame, coefficients: np.ndarray, intercept: float) -> pd.DataFrame:
    current = frame.copy()
    scores = _feature_matrix(current) @ coefficients + intercept
    current["alpha158_ridge_score"] = scores
    current["alpha158_ridge_rank"] = pd.Series(scores, index=current.index).rank(
        method="average", pct=True
    )
    return current


def build_walk_forward_ridge_scores(
    feature_store: dict,
    data: dict,
    train_end,
    alpha: float = RIDGE_ALPHA,
    forward_days: int = FORWARD_DAYS,
    minimum_training_dates: int = MIN_TRAINING_DATES,
):
    """训练期滚动拟合，样本外使用冻结模型。"""
    cutoff = pd.Timestamp(train_end).normalize()
    samples = _training_samples(feature_store, data, cutoff, forward_days)
    final_coefficients, final_intercept, final_rows = _fit(samples, alpha)
    enriched = {}
    expanding_models = 0
    for signal_date, frame in sorted(feature_store.items()):
        signal_date = pd.Timestamp(signal_date).normalize()
        if frame is None or frame.empty:
            enriched[signal_date] = frame.copy()
            continue
        if signal_date <= cutoff:
            available = [sample for sample in samples if sample["target_date"] <= signal_date]
            if len(available) < minimum_training_dates:
                current = frame.copy()
                current["alpha158_ridge_score"] = 0.0
                current["alpha158_ridge_rank"] = 0.5
                enriched[signal_date] = current
                continue
            coefficients, intercept, _ = _fit(available, alpha)
            expanding_models += 1
        else:
            coefficients, intercept = final_coefficients, final_intercept
        enriched[signal_date] = _score_frame(frame, coefficients, intercept)

    coefficient_rows = sorted(
        [
            {"feature": name, "coefficient": float(value)}
            for name, value in zip(FEATURE_NAMES, final_coefficients)
        ],
        key=lambda row: abs(row["coefficient"]),
        reverse=True,
    )
    model = {
        "model": "ridge_cross_sectional_rank_v1",
        "feature_set": FEATURE_SET_NAME,
        "train_end": str(cutoff.date()),
        "forward_days": int(forward_days),
        "ridge_alpha": float(alpha),
        "minimum_training_dates": int(minimum_training_dates),
        "training_dates": len(samples),
        "training_rows": int(final_rows),
        "expanding_training_models": int(expanding_models),
        "leakage_policy": "训练期每个信号只使用标签已在当日可知的更早样本；train_end后固定使用完整训练期模型。",
        "feature_names": list(FEATURE_NAMES),
        "intercept": float(final_intercept),
        "coefficients": coefficient_rows,
    }
    return enriched, model


def apply_frozen_ridge_model(feature_store: dict, model: dict) -> dict:
    """用保存的模型给新截面打分。"""
    coefficient_map = {
        item["feature"]: float(item["coefficient"]) for item in model["coefficients"]
    }
    names = tuple(model.get("feature_names", FEATURE_NAMES))
    coefficients = np.asarray([coefficient_map[name] for name in names], dtype=float)
    intercept = float(model.get("intercept", 0.0))
    result = {}
    for signal_date, frame in feature_store.items():
        if frame is None or frame.empty:
            result[signal_date] = frame.copy()
        else:
            current = frame.copy()
            matrix = _feature_matrix(current, names)
            scores = matrix @ coefficients + intercept
            current["alpha158_ridge_score"] = scores
            current["alpha158_ridge_rank"] = pd.Series(scores, index=current.index).rank(
                method="average", pct=True
            )
            result[signal_date] = current
    return result


def write_ridge_model(model: dict, path) -> str:
    target = Path(path)
    target.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)
