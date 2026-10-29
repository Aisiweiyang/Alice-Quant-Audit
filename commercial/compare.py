"""批量策略对比：用户显式给出2-8组白名单参数方案，一次数据审计多方案回测。

不做任何自动参数搜索；方案必须由用户在计划文件中显式列出（合规边界）。
"""

from __future__ import annotations

import json
import os
import sys
from time import perf_counter

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import math

import pandas as pd

from analysis.backtest_report import derive_run_statistics
from commercial.backtest_runner import FACTOR_STRATEGY, run_base_backtest
from commercial.config import (
    BENCHMARK_CODE,
    CommercialRequest,
    ParameterError,
    build_engine_config,
    build_factor_strategy_config,
)
from commercial.data_input import (
    load_analysis_data,
    load_security_master,
    synthesize_benchmark,
)
from commercial.data_audit import audit_data_credibility
from commercial.overfitting import cscv_pbo, deflated_sharpe, equity_daily_returns
from commercial.report import create_report_directory, write_report_bundle
from commercial.reliability_score import compute_reliability_score
from engine.factor_backtest import prepare_feature_store, run_factor_backtest
from run_metadata import config_snapshot

MAX_PLANS = 8
MIN_PLANS = 2

# 方案组分散度诊断阈值：两两日收益相关>该值视为"策略实质相同"；
# 完成交易低于该值的方案视为退化方案（如阈值过严几乎不建仓）。
# PBO（CSCV）只有在方案策略实质彼此不同时才有统计意义。
DISPERSION_MAX_CORRELATION = 0.98
DEGENERATE_MIN_TRADES = 30
DISPERSION_MIN_OBS = 20

_ALLOWED_OVERRIDES = {
    "initial_capital",
    "max_positions",
    "max_position_pct",
    "rebalance_freq",
    "commission_rate",
    "stamp_tax_rate",
    "slippage",
    "stop_loss_pct",
    "take_profit_pct",
    "use_market_timing",
    "factor_weights",
}
_RANGE_RULES = {
    "initial_capital": (10_000, 100_000_000, float),
    "max_positions": (1, 50, int),
    "max_position_pct": (0.01, 1.0, float),
    "commission_rate": (0.0, 0.02, float),
    "stamp_tax_rate": (0.0, 0.02, float),
    "slippage": (0.0, 0.05, float),
    "stop_loss_pct": (0.01, 0.5, float),
    "take_profit_pct": (0.01, 1.0, float),
}


def validate_strategy_params(params, plan_index: int = 0) -> dict:
    """截面专属参数白名单校验：与commercial.config同一份白名单与范围。

    classic_trend方案不得携带strategy_params（策略归属在run_compare里
    结合request.strategy把关，因为本函数不感知请求策略）。
    """
    from commercial.config import CROSS_SECTIONAL_PARAM_WHITELIST, _bounded

    if not isinstance(params, dict):
        raise ParameterError(f"方案{plan_index}: strategy_params须为对象")
    unknown = set(params) - set(CROSS_SECTIONAL_PARAM_WHITELIST)
    if unknown:
        raise ParameterError(
            f"方案{plan_index}: 不允许的截面参数 {sorted(unknown)}；"
            f"可选: {sorted(CROSS_SECTIONAL_PARAM_WHITELIST)}"
        )
    checked: dict = {}
    for key, value in params.items():
        low, high, cast = CROSS_SECTIONAL_PARAM_WHITELIST[key]
        checked[key] = _bounded(key, value, low, high, cast)
    return checked


def parse_plan_payload(payload) -> list[dict]:
    """解析并校验对比计划payload（白名单/数量/范围）；只读校验，不执行任何回测。

    每个方案支持两类可选参数：
    - overrides：共享回测参数（资金/成本/止盈止损等，经典引擎与截面引擎通用）；
    - strategy_params：截面Alpha158专属参数（仅cross_sectional_alpha158可用，
      classic_trend方案携带时在run_compare中拒绝）。
    """
    plans = payload.get("plans") if isinstance(payload, dict) else payload
    if not isinstance(plans, list):
        raise ParameterError('计划需要 {"plans": [...]} 数组')
    if not MIN_PLANS <= len(plans) <= MAX_PLANS:
        raise ParameterError(f"方案数量必须在[{MIN_PLANS}, {MAX_PLANS}]内，当前{len(plans)}个")
    normalized = []
    for index, plan in enumerate(plans, start=1):
        if not isinstance(plan, dict):
            raise ParameterError(f"第{index}个方案不是对象")
        label = str(plan.get("label") or f"方案{index}").strip()
        if len(label) > 40:
            raise ParameterError(f"第{index}个方案label过长(≤40字符)")
        normalized.append(
            {
                "label": label,
                "overrides": validate_overrides(plan.get("overrides") or {}, index),
                "strategy_params": validate_strategy_params(
                    plan.get("strategy_params") or {}, index
                ),
            }
        )
    return normalized


def load_compare_plan(path: str) -> list[dict]:
    """读取并校验对比计划JSON；返回方案列表。"""
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise ParameterError(f"对比计划文件不存在: {resolved}")
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ParameterError(f"对比计划解析失败: {exc}") from None
    return parse_plan_payload(payload)


def validate_overrides(overrides: dict, plan_index: int = 0) -> dict:
    """白名单与范围校验；非法键或越界值直接拒绝。"""
    from commercial.config import _bounded

    unknown = set(overrides) - _ALLOWED_OVERRIDES
    if unknown:
        raise ParameterError(
            f"方案{plan_index}: 不允许的参数 {sorted(unknown)}；"
            f"允许范围: {sorted(_ALLOWED_OVERRIDES)}"
        )
    checked: dict = {}
    for key, rule in _RANGE_RULES.items():
        if key in overrides:
            low, high, cast = rule
            checked[key] = _bounded(key, overrides[key], low, high, cast)
    if "rebalance_freq" in overrides:
        if overrides["rebalance_freq"] not in {"daily", "weekly", "monthly"}:
            raise ParameterError("rebalance_freq 只支持 daily/weekly/monthly")
        checked["rebalance_freq"] = overrides["rebalance_freq"]
    if "use_market_timing" in overrides:
        checked["use_market_timing"] = bool(overrides["use_market_timing"])
    if "factor_weights" in overrides:
        from strategy.cross_sectional import RANK_COLUMNS

        weights = overrides["factor_weights"]
        if not isinstance(weights, dict) or not weights:
            raise ParameterError(f"方案{plan_index}: factor_weights须为非空字典")
        # 权重键必须是排名列（*_rank），与FactorStrategyConfig.factor_weights
        # 同一口径；此前误按原始因子名(momentum等)校验，通过校验的权重在
        # 截面评分中全部落空（score恒0），属潜伏缺陷，2026-09-14修正。
        bad = set(weights) - set(RANK_COLUMNS)
        if bad:
            raise ParameterError(
                f"方案{plan_index}: 不允许的因子权重键 {sorted(bad)}；可选: {sorted(RANK_COLUMNS)}"
            )
        values = [float(v) for v in weights.values()]
        if any(v < 0 for v in values) or sum(values) <= 0:
            raise ParameterError(f"方案{plan_index}: 因子权重必须非负且不全为零")
        checked["factor_weights"] = {k: float(v) for k, v in weights.items()}
    return checked


def _plan_config(request: CommercialRequest, overrides: dict):
    return build_engine_config(request, **overrides)


def _plan_factor_config(request: CommercialRequest, plan: dict, benchmark: str):
    """把方案overrides与strategy_params合成为该方案专属FactorStrategyConfig。

    overrides的键与CommercialRequest字段同名（rebalance_freq/stamp_tax_rate等
    由build_factor_strategy_config完成到FactorStrategyConfig的映射，含印花税
    2023-08-28历史分段规则），因此先叠加到请求副本再构建；factor_weights不是
    请求字段、strategy_params为截面白名单参数，二者直接替换到配置上。
    """
    from dataclasses import replace as dc_replace

    overrides = dict(plan.get("overrides") or {})
    weights = overrides.pop("factor_weights", None)
    effective = dc_replace(request, **overrides) if overrides else request
    config = build_factor_strategy_config(effective)
    if benchmark != BENCHMARK_CODE:
        config = dc_replace(config, benchmark_index=benchmark)
    if plan.get("strategy_params"):
        config = dc_replace(config, **plan["strategy_params"])
    if weights:
        config = dc_replace(config, factor_weights=weights)
    return config


def _feature_store_key(config) -> tuple:
    """特征存储不可共享的字段集合（strategy.cross_sectional/engine实测口径）。

    存储的结构性内容（候选行/排名列/名单过滤）依赖：minimum_history、
    liquid_universe_size、require_price_above_ma60（build_feature_store）与
    rebalance_frequency、rebalance_day（signal_dates_for_period）——任一不同
    则不能共享。factor_weights不入键：run_factor_backtest对传入存储会按
    config.factor_weights调用rescore_feature_store重打分（与压力消融同机制），
    权重不同的方案可安全共享同一存储。minimum_score/exit_rank等选股执行期
    参数同样不入键。
    """
    return (
        config.rebalance_frequency,
        getattr(config, "rebalance_day", None),
        config.minimum_history,
        config.liquid_universe_size,
        config.require_price_above_ma60,
    )


def _run_factor_plan(
    request: CommercialRequest, plan: dict, data: dict, benchmark: str, store_cache: dict
) -> dict:
    """运行单个截面方案；特征存储按键复用，不因minimum_score等差异重建。"""
    from strategy.cross_sectional import is_main_board_stock

    config = _plan_factor_config(request, plan, benchmark)
    key = _feature_store_key(config)
    store = store_cache.get(key)
    if store is None:
        store = prepare_feature_store(data, request.start, request.end, config, verbose=False)
        store_cache[key] = store
    portfolio, metrics = run_factor_backtest(
        data,
        request.start,
        request.end,
        config=config,
        feature_store=store,
        verbose=not request.quiet,
    )
    stats = derive_run_statistics(
        pd.DataFrame(portfolio.equity_curve), pd.DataFrame(portfolio.trades)
    )
    non_empty = [frame for frame in store.values() if not frame.empty]
    feature_meta = {
        "main_board_stocks_in_data": sum(1 for code in data if is_main_board_stock(code)),
        "signal_sections": len(non_empty),
        "avg_section_size": (
            round(sum(len(f) for f in non_empty) / len(non_empty), 1) if non_empty else 0.0
        ),
    }
    return {
        "metrics": metrics,
        "statistics": stats,
        "equity": pd.DataFrame(portfolio.equity_curve),
        "strategy": FACTOR_STRATEGY,
        "feature_meta": feature_meta,
    }


def _dispersion_diagnosis(daily_returns: dict, ok_results: list) -> dict:
    """方案组分散度诊断：PBO有效性的前置闸门（不改PBO数值本身）。

    PBO（CSCV）回答"样本内最优方案在样本外是否仍优"——只有方案在策略
    实质上彼此不同时才有统计意义：仅差选股阈值的方案组日收益两两高度
    相关，或含几乎不建仓的退化方案时，PBO只会输出吓人但无信息量的数字
    （真实案例：ms60/70/80 三方案相关≈1、ms80仅1笔交易，PBO=0.95实为
    低分散度产物）。诊断输出validity标记供报告与网页引用：
    - 任两方案日收益相关>DISPERSION_MAX_CORRELATION，或存在完成交易
      <DEGENERATE_MIN_TRADES的退化方案 → "insufficient_dispersion"；
    - 相关对重叠观测普遍不足 → "insufficient_data"（宁可数据不足不给ok）；
    - 否则 "ok"。相关系数按共同交易日对齐计算，观测<20时如实None。
    """
    labels = list(daily_returns)
    correlations = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair = pd.concat(
                [daily_returns[labels[i]], daily_returns[labels[j]]],
                axis=1,
            ).dropna(how="any")
            corr = None
            if (
                len(pair) >= DISPERSION_MIN_OBS
                and float(pair.iloc[:, 0].std(ddof=1)) > 0
                and float(pair.iloc[:, 1].std(ddof=1)) > 0
            ):
                value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                if math.isfinite(value):
                    corr = round(value, 4)
            correlations.append(
                {
                    "pair": [labels[i], labels[j]],
                    "correlation": corr,
                }
            )
    finite = [c["correlation"] for c in correlations if c["correlation"] is not None]
    max_corr = max(finite) if finite else None
    degenerate = []
    for result in ok_results:
        trades = int((result.get("metrics") or {}).get("total_trades") or 0)
        if trades < DEGENERATE_MIN_TRADES:
            degenerate.append({"label": result.get("label"), "total_trades": trades})
    if max_corr is None and not degenerate:
        # 全对相关系数都不可算（重叠观测不足）：宁可"数据不足"也不给ok。
        return {
            "validity": "insufficient_data",
            "max_pair_correlation": None,
            "correlations": correlations,
            "degenerate_plans": degenerate,
            "detail": "方案两两日收益的重叠观测不足，无法评估分散度；"
            "PBO统计意义存疑，不可作为过拟合证据引用",
            "note": "分散度诊断只评估这组方案能否支撑PBO的统计意义，不评价任何方案优劣。",
        }
    low_dispersion = max_corr is not None and max_corr > DISPERSION_MAX_CORRELATION
    validity = "insufficient_dispersion" if (low_dispersion or degenerate) else "ok"
    if low_dispersion and degenerate:
        detail = (
            f"方案组缺乏分散度：两两日收益相关最高{max_corr}，"
            f"且存在退化方案{[d['label'] for d in degenerate]}"
            f"（完成交易<{DEGENERATE_MIN_TRADES}）；PBO无统计意义，"
            "不可作为过拟合证据引用"
        )
    elif low_dispersion:
        detail = (
            f"方案组缺乏分散度：两两日收益相关最高{max_corr}"
            f"（>{DISPERSION_MAX_CORRELATION}）；PBO无统计意义，"
            "不可作为过拟合证据引用"
        )
    elif degenerate:
        detail = (
            f"存在退化方案{[d['label'] for d in degenerate]}"
            f"（完成交易<{DEGENERATE_MIN_TRADES}）；PBO无统计意义，"
            "不可作为过拟合证据引用"
        )
    else:
        detail = (
            f"方案两两日收益相关最高{max_corr}"
            f"（≤{DISPERSION_MAX_CORRELATION}），无退化方案，"
            "PBO具备参考意义"
        )
    return {
        "validity": validity,
        "max_pair_correlation": max_corr,
        "correlations": correlations,
        "degenerate_plans": degenerate,
        "detail": detail,
        "note": "分散度诊断只评估这组方案能否支撑PBO的统计意义，不评价任何方案优劣。",
    }


def run_compare(
    request: CommercialRequest, plan_path: str, report_id: str | None = None, progress=None
) -> dict:
    """执行批量对比；返回分析结果（含report_dir）。"""
    from commercial.pipeline import _make_emit

    emit = _make_emit(request, progress)
    plans = load_compare_plan(plan_path)
    # 策略归属把关：classic_trend方案不得携带截面参数（在创建报告目录前拒绝）。
    if request.strategy != FACTOR_STRATEGY:
        offenders = [p["label"] for p in plans if p.get("strategy_params")]
        if offenders:
            raise ParameterError(
                f"策略 {request.strategy} 不接受截面参数strategy_params；涉及方案: {offenders}"
            )
    report_dir, report_id = create_report_directory(request.output_root, report_id)
    emit(f"报告目录: {report_dir}")

    timings = {}
    stage = perf_counter()
    emit("[1/3] 读取并审计数据（全部方案共享一次审计）...")
    data, metadata, source_info, benchmark, audit = None, None, None, None, None
    loaded, metadata, source_info = load_analysis_data(request.cache, request.candidates)
    data = loaded
    benchmark = BENCHMARK_CODE
    if benchmark not in data:
        data, benchmark = synthesize_benchmark(data)
        source_info.setdefault("warnings", []).append("缺少真实沪深300行情，已使用等权合成基准")
    # 正式缓存旁若有证券主数据，审计按各股上市存续期评估（与full管线同一口径）。
    security_master = load_security_master(request.cache)
    audit = audit_data_credibility(
        data,
        start=request.start,
        end=request.end,
        benchmark=benchmark,
        metadata=metadata,
        source_info=source_info,
        security_master=security_master,
    )
    if audit["rating"] == "NOT_READY":
        emit(f"[阻断] 数据审计NOT_READY：{'；'.join(audit['blockers'])}")
        analysis = {
            "report_id": report_id,
            "report_dir": report_dir,
            "request": request,
            "data_audit": audit,
            "source_info": source_info,
            "candidate_info": None,
            "benchmark_code": benchmark,
            "mode": "compare",
            "blocked": True,
            "block_message": "数据审计NOT_READY：" + "；".join(audit["blockers"]),
            "plans": plans,
            "plan_results": [],
            "timings": {},
            "score": None,
            "issues": [],
            "recommendations": [],
            "notes": [],
        }
        manifest = write_report_bundle(report_dir, analysis)
        analysis["manifest"] = manifest
        return analysis
    if os.path.isfile(request.cache) and not request.cache.lower().endswith(".pkl"):
        request.allow_incomplete_data = True
    timings["data_stage_seconds"] = round(perf_counter() - stage, 2)

    emit("[2/3] 依次回测{0}个方案（策略：{1}）...".format(len(plans), request.strategy))
    request.extras["base_factor_weights"] = None  # 对比模式不注入消融场景
    plan_results = []
    baseline_metrics = None
    store_cache: dict = {}  # 特征存储按键复用；截面方案间共享，避免重复计算
    for index, plan in enumerate(plans, start=1):
        emit(f"    方案 {index}/{len(plans)}: {plan['label']}")
        stage = perf_counter()
        config = None
        try:
            if request.strategy == FACTOR_STRATEGY:
                config = _plan_factor_config(request, plan, benchmark)
                result = _run_factor_plan(request, plan, data, benchmark, store_cache)
            else:
                config = _plan_config(request, plan["overrides"])
                if benchmark != BENCHMARK_CODE:
                    config = _plan_config(
                        request, {**plan["overrides"], "benchmark_index": benchmark}
                    )
                result = run_base_backtest(request, config, request.cache, data=data)
        except Exception as exc:
            plan_results.append(
                {
                    "label": plan["label"],
                    "status": "insufficient_data",
                    "detail": f"方案回测失败: {type(exc).__name__}: {exc}",
                    "overrides": plan["overrides"],
                    "strategy_params": plan.get("strategy_params") or {},
                    "effective_config": config_snapshot(config) if config is not None else {},
                }
            )
            continue
        timings[f"plan_{index}_seconds"] = round(perf_counter() - stage, 2)
        metrics = result["metrics"]
        if baseline_metrics is None:
            baseline_metrics = metrics
        score = compute_reliability_score(
            data_audit=audit,
            base_result=result,
            segment_consistency={"status": "insufficient_data"},
            stress_summary={"status": "insufficient_data"},
            manifest_complete=True,
            metadata_present=metadata is not None,
        )
        score["mode_note"] = (
            "对比模式评分：未包含压力测试与样本外维度，仅基于数据质量/执行/可复现性"
        )
        plan_results.append(
            {
                "label": plan["label"],
                "status": "ok",
                "overrides": plan["overrides"],
                "strategy_params": plan.get("strategy_params") or {},
                "strategy": request.strategy,
                "metrics": metrics,
                "score": score,
                "equity": result["equity"],
                "annualized_turnover": result["statistics"].get("annualized_turnover"),
                "total_friction": result["statistics"].get("total_friction"),
                "return_delta_vs_baseline": round(
                    metrics["total_return_pct"] - baseline_metrics["total_return_pct"], 2
                ),
                "effective_config": config_snapshot(config),
            }
        )

    # 过拟合审计：方案≥3时做CSCV的PBO；每个完成回测的方案按
    # 论文公式计算Deflated Sharpe（试验数N=完成回测的方案数，各试验Sharpe
    # 的分散度已知）。方案不足或时段不足时如实"数据不足"，不编造数字。
    ok_results = [r for r in plan_results if r.get("status") == "ok"]
    overfitting_audit = {"n_trials": len(ok_results)}
    if ok_results:
        daily_returns = {r["label"]: equity_daily_returns(r["equity"]) for r in ok_results}
        if len(ok_results) >= 3:
            returns_table = pd.concat(daily_returns, axis=1).dropna(how="any")
            pbo_result = cscv_pbo(returns_table)
            # 分散度前置诊断：评估这组方案能否支撑PBO的统计意义。
            overfitting_audit["dispersion"] = _dispersion_diagnosis(daily_returns, ok_results)
        else:
            pbo_result = {
                "status": "insufficient_data",
                "pbo_probability": None,
                "cscv_combinations": None,
                "detail": f"完成回测的方案仅{len(ok_results)}个，PBO（CSCV）需要≥3个方案",
            }
        overfitting_audit["pbo"] = pbo_result
        trial_sharpes = [
            float(series.mean() / series.std(ddof=1))
            for series in daily_returns.values()
            if float(series.std(ddof=1)) > 0
        ]
        for r in ok_results:
            r["deflated_sharpe"] = deflated_sharpe(
                daily_returns[r["label"]],
                n_trials=len(ok_results),
                trial_sharpes=trial_sharpes if len(trial_sharpes) >= 2 else None,
            )
        overfitting_audit["note"] = "PBO衡量的是这组方案整体过拟合概率，不是对未来收益的预测。"
    else:
        overfitting_audit["pbo"] = {
            "status": "insufficient_data",
            "pbo_probability": None,
            "cscv_combinations": None,
            "detail": "没有完成回测的方案，无法做PBO",
        }

    emit("[3/3] 生成对比报告...")
    analysis = {
        "report_id": report_id,
        "report_dir": report_dir,
        "request": request,
        "data_audit": audit,
        "source_info": source_info,
        "candidate_info": None,
        "benchmark_code": benchmark,
        "mode": "compare",
        "blocked": audit["rating"] == "NOT_READY",
        "block_message": None,
        "plans": plans,
        "plan_results": plan_results,
        "timings": timings,
        "score": None,
        "notes": [
            "对比模式：各方案共享一次数据审计；评分未含压力/样本外维度（见各方案说明）。",
        ],
        "overfitting_audit": overfitting_audit,
        "issues": [],
        "recommendations": [],
        "effective_config": {
            "mode": "compare",
            "benchmark_used": benchmark,
        },
    }
    issues = []
    for warning in audit.get("warnings") or []:
        issues.append(f"数据警告：{warning}")
    for result in plan_results:
        if result.get("status") != "ok":
            issues.append(f"方案「{result['label']}」未完成：{result.get('detail')}")
    analysis["issues"] = issues
    manifest = write_report_bundle(report_dir, analysis)
    analysis["manifest"] = manifest
    emit(f"完成：对比报告 {os.path.join(report_dir, 'index.html')}")
    return analysis
