"""预定义压力测试场景：最多8个方案，不做无限参数搜索。

每个场景只改变少量明确假设，输出与基准方案的指标差异和敏感度结论。
"""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from analysis.backtest_report import derive_run_statistics
from engine.backtest import run as run_classic_backtest

MAX_SCENARIOS = 8


def _renormalised_without(weights: dict, removed: str) -> dict:
    rest = {k: v for k, v in weights.items() if k != removed}
    total = sum(rest.values())
    if total <= 0:
        return {k: 0.0 for k in rest}
    return {k: v / total for k, v in rest.items()}


def default_scenarios(request, factor_strategy: bool = False) -> list[dict]:
    """返回预定义场景；执行数量由 request.max_scenarios 限制（≤8）。

    默认前8个含消融场景；"持仓数量下调"排第9，默认被截断（专业版开放）。
    截面Alpha158策略使用适配版场景集：无买入延迟场景（截面引擎无该参数），
    动量消融映射为剔除momentum_rank秩权重。
    """
    base_weights = request.extras.get("base_factor_weights")
    scenarios = [
        {"key": "baseline", "label": "基础交易成本", "overrides": {}},
        {
            "key": "cost_up_50",
            "label": "交易成本提高50%",
            "overrides": {
                "commission_rate": request.commission_rate * 1.5,
                "stamp_tax_rate": request.stamp_tax_rate * 1.5,
                "stamp_tax_rate_before_20230828": request.stamp_tax_rate * 1.5,
                "stamp_tax_rate_after_20230828": request.stamp_tax_rate * 1.5,
            },
        },
        {
            "key": "cost_up_100",
            "label": "交易成本提高100%",
            "overrides": {
                "commission_rate": request.commission_rate * 2.0,
                "stamp_tax_rate": request.stamp_tax_rate * 2.0,
                "stamp_tax_rate_before_20230828": request.stamp_tax_rate * 2.0,
                "stamp_tax_rate_after_20230828": request.stamp_tax_rate * 2.0,
            },
        },
        {
            "key": "slippage_x3",
            "label": "滑点扩大3倍",
            "overrides": {"slippage": request.slippage * 3.0},
        },
        {"key": "buy_delay_1d", "label": "买入延迟一个交易日", "overrides": {"signal_lag_days": 2}},
        {
            "key": "no_market_timing",
            "label": "不启用市场择时",
            "overrides": {"use_market_timing": False},
        },
        {
            "key": "smaller_position",
            "label": "单股仓位降低至60%",
            "overrides": {"max_position_pct": round(request.max_position_pct * 0.6, 4)},
        },
        {
            "key": "ablation_no_momentum",
            "label": "剔除动量因子(消融)",
            "overrides": {
                "factor_weights": (
                    _renormalised_without(base_weights, "momentum") if base_weights else {}
                )
            },
        },
        {
            "key": "fewer_positions",
            "label": f"持仓数量降至{max(3, request.max_positions // 2)}只",
            "overrides": {"max_positions": max(3, request.max_positions // 2)},
        },
    ]
    if not base_weights and not factor_strategy:
        # 经典模式消融需要基准权重快照；截面模式在执行处改用factor_config权重。
        scenarios = [s for s in scenarios if s["key"] != "ablation_no_momentum"]
    if factor_strategy:
        # 截面引擎无signal_lag_days，买入延迟场景不适用；动量消融在执行处
        # 映射为剔除momentum_rank秩权重（见run_stress_tests）。
        scenarios = [s for s in scenarios if s["key"] != "buy_delay_1d"]
    return scenarios[: max(1, min(int(request.max_scenarios), MAX_SCENARIOS))]


def run_stress_tests(
    request, base_config, cache_path, verbose=False, data=None, factor_config=None
) -> list[dict]:
    """依次运行预定义场景，返回每个场景的指标与相对基准的变化。

    ``factor_config``非空时走截面Alpha158场景（dataclasses.replace生成场景配置，
    engine.factor_backtest执行）；缺省走经典引擎。单场景失败不阻断整轮。
    """
    factor_strategy = factor_config is not None
    scenarios = default_scenarios(request, factor_strategy=factor_strategy)
    baseline_result = None
    results = []
    feature_store = None  # 截面场景间复用因子存储（权重变化由rescore处理）
    for index, scenario in enumerate(scenarios, start=1):
        overrides = dict(scenario["overrides"])
        if factor_strategy and scenario["key"] == "ablation_no_momentum":
            # 截面消融：剔除momentum_rank并将其余秩权重归一化。
            overrides["factor_weights"] = _renormalised_without(
                dict(factor_config.factor_weights), "momentum_rank"
            )
        try:
            if factor_strategy:
                from dataclasses import replace

                from engine.factor_backtest import (
                    prepare_feature_store,
                    run_factor_backtest,
                )

                if feature_store is None:
                    feature_store = prepare_feature_store(
                        data, request.start, request.end, factor_config, verbose=False
                    )
                scenario_config = (
                    replace(factor_config, **overrides) if overrides else factor_config
                )
                portfolio, metrics = run_factor_backtest(
                    data,
                    request.start,
                    request.end,
                    config=scenario_config,
                    feature_store=feature_store,
                    verbose=False,
                )
            else:
                config = (
                    base_config
                    if not overrides
                    else type("StressConfigView", (base_config,), overrides)
                )
                portfolio, metrics = run_classic_backtest(
                    request.start,
                    request.end,
                    verbose=False,
                    cache_path=None if data is not None else cache_path,
                    allow_incomplete_data=request.allow_incomplete_data,
                    config=config,
                    data=data,
                )
        except Exception as exc:  # 单场景失败不阻断整轮压力测试
            results.append(
                {
                    "scenario": scenario["key"],
                    "label": scenario["label"],
                    "status": "insufficient_data",
                    "detail": f"场景运行失败: {type(exc).__name__}: {exc}",
                    "overrides": {k: v for k, v in scenario["overrides"].items()},
                }
            )
            continue
        if portfolio is None or metrics is None:
            results.append(
                {
                    "scenario": scenario["key"],
                    "label": scenario["label"],
                    "status": "insufficient_data",
                    "overrides": {k: v for k, v in scenario["overrides"].items()},
                }
            )
            continue
        stats = derive_run_statistics(
            pd.DataFrame(portfolio.equity_curve), pd.DataFrame(portfolio.trades)
        )
        entry = {
            "scenario": scenario["key"],
            "label": scenario["label"],
            "status": "ok",
            "overrides": {k: v for k, v in scenario["overrides"].items() if not callable(v)},
            "metrics": metrics,
            "annualized_turnover": stats["annualized_turnover"],
            "total_friction": stats["total_friction"],
        }
        if scenario["key"] == "baseline":
            baseline_result = entry
            entry["deltas"] = {}
        elif baseline_result is not None:
            entry["deltas"] = compare_to_baseline(
                baseline_result["metrics"], metrics, baseline_result, entry
            )
        else:
            entry["deltas"] = {}
        results.append(entry)
        if verbose:
            print(f"    压力场景 {index}/{len(scenarios)}: {scenario['label']} 完成")
    return results


def compare_to_baseline(base_metrics, metrics, base_entry, entry) -> dict:
    return {
        "total_return_delta_pct": round(
            metrics["total_return_pct"] - base_metrics["total_return_pct"], 2
        ),
        "sharpe_delta": round(
            (metrics.get("sharpe_ratio") or 0) - (base_metrics.get("sharpe_ratio") or 0), 2
        ),
        "max_drawdown_delta_pct": round(
            metrics["max_drawdown_pct"] - base_metrics["max_drawdown_pct"], 2
        ),
        "turnover_delta": round(
            (entry.get("annualized_turnover") or 0) - (base_entry.get("annualized_turnover") or 0),
            2,
        ),
        "friction_delta": round(
            (entry.get("total_friction") or 0) - (base_entry.get("total_friction") or 0), 2
        ),
    }


def sensitivity_summary(results: list[dict]) -> dict:
    """哪些假设最影响结果；策略是否对参数过度敏感。"""
    base = next((r for r in results if r.get("scenario") == "baseline"), None)
    if base is None or base.get("metrics") is None:
        return {"status": "insufficient_data", "detail": "缺少基准方案，无法评估敏感性"}
    base_return = float(base["metrics"]["total_return_pct"])
    ranked = []
    for result in results:
        if result.get("scenario") == "baseline" or result.get("status") != "ok":
            continue
        delta = float(result["metrics"]["total_return_pct"]) - base_return
        ranked.append(
            {
                "scenario": result["scenario"],
                "label": result["label"],
                "return_delta_pct": round(delta, 2),
                "sharpe_delta": result["deltas"].get("sharpe_delta"),
                "sign_flipped": (base_return > 0) != (base_return + delta > 0)
                and abs(base_return) > 1e-9,
            }
        )
    ranked.sort(key=lambda item: abs(item["return_delta_pct"]), reverse=True)
    most_impactful = ranked[0] if ranked else None
    oversensitive = bool(ranked) and (
        any(item["sign_flipped"] for item in ranked)
        or (
            most_impactful
            and abs(most_impactful["return_delta_pct"]) > max(30.0, abs(base_return) * 0.5)
        )
    )
    return {
        "status": "ok",
        "base_return_pct": base_return,
        "ranking": ranked,
        "most_impactful": most_impactful,
        "over_sensitive": oversensitive,
        "interpretation": (
            "策略结果对某些假设过度敏感，回测结论不稳定"
            if oversensitive
            else "策略结果在预定义压力场景下未出现极端翻转，但这一点不保证未来表现"
        ),
    }


def scenario_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        if result.get("status") != "ok":
            rows.append(
                {
                    "scenario": result["scenario"],
                    "label": result["label"],
                    "status": result["status"],
                }
            )
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "scenario": result["scenario"],
                "label": result["label"],
                "status": "ok",
                "total_return_pct": metrics.get("total_return_pct"),
                "cagr_pct": metrics.get("cagr_pct"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sharpe_reliable": metrics.get("sharpe_reliable"),
                "total_trades": metrics.get("total_trades"),
                "annualized_turnover": result.get("annualized_turnover"),
                "total_friction": result.get("total_friction"),
                "return_delta_pct": result.get("deltas", {}).get("total_return_delta_pct"),
                "sharpe_delta": result.get("deltas", {}).get("sharpe_delta"),
                "drawdown_delta_pct": result.get("deltas", {}).get("max_drawdown_delta_pct"),
                "turnover_delta": result.get("deltas", {}).get("turnover_delta"),
                "friction_delta": result.get("deltas", {}).get("friction_delta"),
            }
        )
    return pd.DataFrame(rows)
