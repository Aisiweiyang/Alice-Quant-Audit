"""多窗口滚动样本外验证：把回测区间切成若干连续子窗口分别回测。

用于评估策略收益在不同时段的稳定性；任一窗口数据不足时如实返回
insufficient_data，不伪造稳定性结论。
"""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from engine.backtest import run as run_classic_backtest

DEFAULT_WINDOWS = 3
MAX_WINDOWS = 4
MIN_SESSIONS = 30
MIN_TRADES = 3


def split_windows(start, end, windows=DEFAULT_WINDOWS, anchored=False):
    """把[start, end]切分为连续子窗口；返回[(label, start, end)]。

    缺省（anchored=False）：不重叠等分窗口，window_i只覆盖自己的子区间；
    anchored=True：锚定窗口（walk-forward惯例）——每个窗口起点固定为区间
    起点、终点递进（window_i = [区间起点, 第i/windows处分位]），窗口逐个
    变长，模拟"数据逐期累积、每次用全部历史重新评估"的滚动流程。
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts >= end_ts or windows < 2:
        return []
    total_days = (end_ts - start_ts).days
    size = total_days / windows
    result = []
    for index in range(windows):
        w_start = (
            start_ts
            + pd.Timedelta(days=int(index * size))
            + (pd.Timedelta(days=1) if index else pd.Timedelta(days=0))
        )
        w_end = start_ts + pd.Timedelta(days=int((index + 1) * size))
        if index == windows - 1:
            w_end = end_ts
        if anchored and index:
            w_start = start_ts  # 锚定：起点始终固定为区间起点
        result.append(
            (f"window_{index + 1}", w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"))
        )
    return result


def run_rolling_oos(
    request,
    config,
    cache_path,
    windows=DEFAULT_WINDOWS,
    data=None,
    factor_config=None,
    anchored=False,
):
    """逐窗口回测并给出稳定性结论。

    ``factor_config``非空时走截面Alpha158引擎（因子存储全期构建一次、
    各窗口复用）；每窗口附带Sharpe的95%块自助置信区间（含零=该子区间
    正收益不具统计显著性）。``anchored``为True时使用锚定递进窗口
    （窗口起点固定为区间起点），缺省保持不重叠等分不变。
    """
    factor_strategy = factor_config is not None
    windows = max(2, min(int(windows), MAX_WINDOWS))
    segments = split_windows(request.start, request.end, windows, anchored=anchored)
    if not segments:
        return {"status": "insufficient_data", "detail": "区间过短，无法切分滚动窗口"}
    feature_store = None
    results = []
    for label, w_start, w_end in segments:
        try:
            if factor_strategy:
                from engine.factor_backtest import (
                    prepare_feature_store,
                    run_factor_backtest,
                )

                if feature_store is None:
                    feature_store = prepare_feature_store(
                        data, request.start, request.end, factor_config, verbose=False
                    )
                portfolio, metrics = run_factor_backtest(
                    data,
                    w_start,
                    w_end,
                    config=factor_config,
                    feature_store=feature_store,
                    verbose=False,
                )
            else:
                portfolio, metrics = run_classic_backtest(
                    w_start,
                    w_end,
                    verbose=False,
                    cache_path=None if data is not None else cache_path,
                    allow_incomplete_data=request.allow_incomplete_data,
                    config=config,
                    data=data,
                )
        except Exception as exc:
            results.append(
                {
                    "window": label,
                    "start": w_start,
                    "end": w_end,
                    "status": "insufficient_data",
                    "detail": f"窗口运行失败: {exc}",
                }
            )
            continue
        if portfolio is None or metrics is None:
            results.append(
                {"window": label, "start": w_start, "end": w_end, "status": "insufficient_data"}
            )
            continue
        results.append(
            {
                "window": label,
                "start": w_start,
                "end": w_end,
                "status": "ok",
                "total_return_pct": metrics.get("total_return_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sharpe_reliable": metrics.get("sharpe_reliable"),
                "total_trades": metrics.get("total_trades"),
                "sharpe_ci_low": metrics.get("sharpe_ci_low"),
                "sharpe_ci_high": metrics.get("sharpe_ci_high"),
                "sharpe_ci_includes_zero": metrics.get("sharpe_ci_includes_zero"),
            }
        )

    usable = [r for r in results if r.get("status") == "ok"]
    if len(usable) < len(results) or len(usable) < 2:
        weak = [
            r
            for r in results
            if r.get("status") == "ok"
            and (int(r.get("total_trades") or 0) < MIN_TRADES or not r.get("sharpe_reliable"))
        ]
        detail = (
            f"仅{len(usable)}/{len(results)}个窗口数据充足"
            if len(usable) < len(results)
            else f"{len(weak)}个窗口交易数<跨{MIN_TRADES}或样本过短"
        )
        return {
            "status": "insufficient_data",
            "detail": f"滚动样本外数据不足：{detail}；不输出稳定性结论",
            "windows": results,
        }

    returns = [float(r["total_return_pct"]) for r in usable]
    sharpes = [float(r.get("sharpe_ratio") or 0) for r in usable]
    positive_share = sum(1 for v in returns if v > 0) / len(returns)
    sharpe_std = float(pd.Series(sharpes).std(ddof=1)) if len(sharpes) > 1 else 0.0
    # 窗口显著性：Sharpe自助CI包含零的窗口占比（高占比=收益集中、不可信）。
    ci_zero_share = sum(1 for r in usable if r.get("sharpe_ci_includes_zero")) / len(usable)
    if positive_share >= 2 / 3 and sharpe_std <= 0.75:
        verdict = "stable"
    elif positive_share >= 0.5:
        verdict = "mixed"
    else:
        verdict = "unstable"
    detail_text = {
        "stable": f"{len(usable)}个窗口中{int(round(positive_share * len(usable)))}个为正收益，"
        f"Sharpe离散度低（σ={sharpe_std:.2f}），时段稳定性较好",
        "mixed": "部分窗口为正收益，稳定性一般，收益依赖具体时段",
        "unstable": "多数窗口为负收益，收益集中少数时段，须高度警惕",
    }[verdict]
    if ci_zero_share >= 0.5:
        detail_text += (
            f"；且{int(round(ci_zero_share * len(usable)))}/{len(usable)}个窗口的"
            "Sharpe置信区间包含零，正收益显著性不足"
        )
    return {
        "status": "ok",
        "verdict": verdict,
        "detail": detail_text,
        "anchored": bool(anchored),
        "positive_share": round(positive_share, 3),
        "sharpe_std": round(sharpe_std, 3),
        "ci_zero_share": round(ci_zero_share, 3),
        "windows": results,
        "note": (
            "锚定窗口：各窗口起点固定为区间起点、终点递进，窗口间存在重叠；"
            "任一窗口的结论只描述该子区间。"
            if anchored
            else "滚动窗口彼此不重叠；任一窗口的结论只描述该子区间。"
        )
        + "Sharpe置信区间为块自助估计（块长10个交易日，固定种子）。",
    }


def rolling_frame(rolling) -> pd.DataFrame:
    if not rolling or rolling.get("status") != "ok":
        return pd.DataFrame()
    rows = []
    for window in rolling.get("windows", []):
        if window.get("status") != "ok":
            rows.append(
                {
                    "window": window["window"],
                    "start": window["start"],
                    "end": window["end"],
                    "status": "insufficient_data",
                }
            )
            continue
        rows.append(
            {
                "window": window["window"],
                "start": window["start"],
                "end": window["end"],
                "status": "ok",
                "total_return_pct": window.get("total_return_pct"),
                "max_drawdown_pct": window.get("max_drawdown_pct"),
                "sharpe_ratio": window.get("sharpe_ratio"),
                "sharpe_reliable": window.get("sharpe_reliable"),
                "total_trades": window.get("total_trades"),
                "sharpe_ci_low": window.get("sharpe_ci_low"),
                "sharpe_ci_high": window.get("sharpe_ci_high"),
                "sharpe_ci_includes_zero": window.get("sharpe_ci_includes_zero"),
            }
        )
    return pd.DataFrame(rows)
