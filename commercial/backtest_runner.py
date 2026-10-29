"""商业化回测执行层：缓存准备、基础回测、分段一致性检查。

只复用 engine.backtest 的既有执行规则，不引入新的成交假设。
"""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from commercial.config import build_factor_strategy_config
from analysis.backtest_report import derive_run_statistics
from engine.backtest import run as run_classic_backtest

MIN_SEGMENTS_SESSIONS = 30
MIN_SEGMENTS_TRADES = 3


def prepare_cache(request, data, benchmark: str, working_dir: str) -> tuple[str, bool]:
    """返回可直接传给回测引擎的缓存路径；CSV来源会先落盘成Alice缓存。"""
    from commercial.data_audit import stage_cache_for_backtest

    source = os.path.abspath(os.path.expanduser(request.cache))
    if os.path.isfile(source) and source.lower().endswith(".pkl"):
        return source, False
    staged = stage_cache_for_backtest(data, benchmark, working_dir)
    return staged, True


def run_base_backtest(request, config, cache_path, data=None):
    """运行基准回测，附带换手/摩擦成本统计。``data``非空时复用内存数据。"""
    portfolio, metrics = run_classic_backtest(
        request.start,
        request.end,
        verbose=not request.quiet,
        cache_path=None if data is not None else cache_path,
        allow_incomplete_data=request.allow_incomplete_data,
        config=config,
        data=data,
    )
    if portfolio is None or metrics is None:
        return None
    stats = derive_run_statistics(
        pd.DataFrame(portfolio.equity_curve), pd.DataFrame(portfolio.trades)
    )
    return {
        "portfolio": portfolio,
        "metrics": metrics,
        "statistics": stats,
        "equity": pd.DataFrame(portfolio.equity_curve),
        "trades": pd.DataFrame(portfolio.trades),
    }


FACTOR_STRATEGY = "cross_sectional_alpha158"


def run_factor_strategy_backtest(request, cache_path, data=None):
    """运行截面Alpha158策略回测（engine.factor_backtest），执行门禁与主回测一致。

    返回与run_base_backtest相同结构的dict，另附feature_meta（截面可用性披露）。
    数据不足（主板池/65日历史门槛）时如实在feature_meta中披露，不伪造表现。
    """
    from engine.factor_backtest import run_factor_backtest
    from strategy.cross_sectional import is_main_board_stock

    if data is None:
        raise ValueError("截面策略回测需要内存数据（data=直通）")
    factor_config = build_factor_strategy_config(request)
    portfolio, metrics = run_factor_backtest(
        data,
        request.start,
        request.end,
        config=factor_config,
        verbose=not request.quiet,
    )
    stats = derive_run_statistics(
        pd.DataFrame(portfolio.equity_curve), pd.DataFrame(portfolio.trades)
    )
    store = getattr(portfolio, "feature_store", None)
    if store is None:
        # run_factor_backtest不回传store时按主板池规模粗略披露。
        main_board = sum(1 for code in data if is_main_board_stock(code))
        feature_meta = {
            "main_board_stocks_in_data": main_board,
            "signal_sections": None,
            "note": "主板池或最低65日历史不满足时，截面候选为空属正常披露",
        }
    else:
        non_empty = [frame for frame in store.values() if not frame.empty]
        feature_meta = {
            "main_board_stocks_in_data": sum(1 for code in data if is_main_board_stock(code)),
            "signal_sections": len(non_empty),
            "avg_section_size": (
                round(sum(len(f) for f in non_empty) / len(non_empty), 1) if non_empty else 0.0
            ),
        }
    return {
        "portfolio": portfolio,
        "metrics": metrics,
        "statistics": stats,
        "equity": pd.DataFrame(portfolio.equity_curve),
        "trades": pd.DataFrame(portfolio.trades),
        "strategy": FACTOR_STRATEGY,
        "feature_meta": feature_meta,
    }


def run_segment_consistency(request, config, cache_path, data=None, run_function=None) -> dict:
    """前后两段一致性检查（样本外近似）：区间中点切分，分别回测。

    ``run_function(seg_start, seg_end)`` 可注入其他内置策略的回测
    （如截面Alpha158）；缺省用经典引擎。两段任一数据不足时返回
    insufficient_data，不伪造结论。
    """
    if run_function is None:

        def run_function(seg_start, seg_end):
            return run_classic_backtest(
                seg_start,
                seg_end,
                verbose=False,
                cache_path=None if data is not None else cache_path,
                allow_incomplete_data=request.allow_incomplete_data,
                config=config,
                data=data,
            )

    start = pd.Timestamp(request.start).normalize()
    end = pd.Timestamp(request.end).normalize()
    middle = start + (end - start) / 2
    first_end = (middle - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    second_start = middle.strftime("%Y-%m-%d")
    segments = {}
    segment_errors = {}
    for label, seg_start, seg_end in (
        ("first_half", request.start, first_end),
        ("second_half", second_start, request.end),
    ):
        try:
            portfolio, metrics = run_function(seg_start, seg_end)
        except Exception as exc:
            portfolio = None
            metrics = None
            segment_errors[label] = f"{type(exc).__name__}: {exc}"
        segments[label] = {
            "start": seg_start,
            "end": seg_end,
            "metrics": metrics,
            "portfolio": portfolio,
        }
    first = segments["first_half"]["metrics"] or {}
    second = segments["second_half"]["metrics"] or {}
    first_sessions = int(first.get("sharpe_observations") or 0)
    second_sessions = int(second.get("sharpe_observations") or 0)
    first_trades = int(first.get("total_trades") or 0)
    second_trades = int(second.get("total_trades") or 0)
    if segment_errors:
        return {
            "status": "insufficient_data",
            "detail": f"分段回测失败（{'; '.join(segment_errors.values())}），不输出一致性结论",
            "segments": {k: {"metrics": v["metrics"]} for k, v in segments.items()},
        }
    if (
        first_sessions < MIN_SEGMENTS_SESSIONS
        or second_sessions < MIN_SEGMENTS_SESSIONS
        or first_trades < MIN_SEGMENTS_TRADES
        or second_trades < MIN_SEGMENTS_TRADES
    ):
        return {
            "status": "insufficient_data",
            "detail": (
                f"分段数据不足（前段{first_sessions}个收益观测/{first_trades}笔交易，"
                f"后段{second_sessions}个收益观测/{second_trades}笔交易），"
                "不输出一致性结论"
            ),
            "segments": {k: {"metrics": v["metrics"]} for k, v in segments.items()},
        }

    first_return = float(first["total_return_pct"])
    second_return = float(second["total_return_pct"])
    first_sharpe = float(first.get("sharpe_ratio") or 0)
    second_sharpe = float(second.get("sharpe_ratio") or 0)
    same_sign = (first_return > 0) == (second_return > 0)
    sharpe_gap = abs(first_sharpe - second_sharpe)
    if same_sign and sharpe_gap <= 0.75:
        verdict = "consistent"
    elif same_sign:
        verdict = "weakly_consistent"
    else:
        verdict = "inconsistent"
    return {
        "status": "ok",
        "verdict": verdict,
        "detail": {
            "consistent": "前后两段收益方向一致且Sharpe接近，时间稳定性较好",
            "weakly_consistent": "前后两段收益方向一致，但Sharpe差异明显，稳定性一般",
            "inconsistent": "前后两段收益方向相反，策略收益主要来自特定时段，须警惕",
        }[verdict],
        "segments": {
            "first_half": {"metrics": first},
            "second_half": {"metrics": second},
        },
        "sharpe_gap": round(sharpe_gap, 2),
        "note": "该检查是时间分段一致性，属于样本外验证的近似；分段仍在同一历史区间内。",
    }
