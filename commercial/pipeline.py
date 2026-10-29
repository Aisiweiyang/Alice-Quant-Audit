"""商业分析编排管线：数据审计 → 可信回测 → 压力测试 → 评分 → 报告。

数据不足或质量不达标时阻断回测并如实披露，不生成虚假结论。
"""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from time import perf_counter

import pandas as pd

from commercial.backtest_runner import (
    FACTOR_STRATEGY,
    prepare_cache,
    run_base_backtest,
    run_factor_strategy_backtest,
    run_segment_consistency,
)
from commercial.rolling_oos import run_rolling_oos
from commercial.config import (
    BENCHMARK_CODE,
    build_engine_config,
    build_factor_strategy_config,
)
from commercial.data_audit import audit_data_credibility
from commercial.data_input import (
    DataInputError,
    load_analysis_data,
    load_security_master,
    synthesize_benchmark,
)
from commercial.report import (
    create_report_directory,
    sha256_file,
    write_report_bundle,
)
from commercial.reliability_score import compute_reliability_score
from commercial.stress import run_stress_tests, scenario_frame, sensitivity_summary
from run_metadata import config_snapshot


def _log(request, message, progress=None):
    if not request.quiet:
        print(message)
    if progress:
        try:
            progress(message)
        except Exception:
            pass


def _make_emit(request, progress=None):
    def emit(message):
        _log(request, message, progress)

    return emit


def _collect_issues(audit, base, stress_summary, consistency, score) -> list[str]:
    issues = []
    issues.extend((audit or {}).get("blockers") or [])
    for warning in (audit or {}).get("warnings") or []:
        issues.append(f"数据警告：{warning}")
    metrics = (base or {}).get("metrics") or {}
    for warning in metrics.get("reliability_warnings") or []:
        issues.append(f"统计警告：{warning}")
    if metrics.get("sharpe_ci_includes_zero"):
        issues.append("Sharpe的95%自助置信区间包含零：正收益不具统计显著性，勿据单次回测下结论")
    if (stress_summary or {}).get("over_sensitive"):
        issues.append("压力测试显示策略结果对部分假设过度敏感")
    if (consistency or {}).get("verdict") == "inconsistent":
        issues.append("前后半段收益方向相反，收益高度依赖特定时段")
    for key in (score or {}).get("failed_dimensions") or []:
        issues.append(f"评分维度未通过：{key}")
    return issues


def _build_recommendations(audit, base, stress_summary, consistency, rolling=None) -> list[str]:
    recommendations = []
    rating = (audit or {}).get("rating")
    if rating == "LIMITED":
        recommendations.append(
            "补齐数据后重新审计：需要完整历史停牌/ST字段、真实沪深300基准和历史PE/PB，"
            "才能把评级提升到READY"
        )
    metrics = (base or {}).get("metrics") or {}
    if metrics and not metrics.get("sharpe_reliable"):
        recommendations.append("延长回测区间：当前样本太短，Sharpe与波动指标不具统计意义")
    if metrics.get("sharpe_ci_includes_zero"):
        recommendations.append(
            "Sharpe块自助置信区间包含零：正收益统计显著性不足，"
            "延长区间或扩大样本后重新评估，勿据单次回测下结论"
        )
    if int(metrics.get("total_trades") or 0) < 10:
        recommendations.append("交易笔数过少，延长区间或扩大候选范围后再下结论")
    mc_p95 = metrics.get("mc_max_drawdown_pct_p95")
    realized_dd = metrics.get("max_drawdown_pct")
    if mc_p95 is not None and realized_dd is not None and float(mc_p95) < float(realized_dd) * 0.8:
        # 已实现回撤明显浅于重排分布的95分位：当前平静可能只是收益顺序的运气。
        recommendations.append(
            f"已实现最大回撤{realized_dd}%明显浅于蒙特卡洛重构的95分位{mc_p95}%："
            "回撤变深主要是收益顺序问题，资金规划请按更深的分位数口径留余量"
            "（历史样本统计重构，不是对未来回撤的预测）"
        )
    if (stress_summary or {}).get("over_sensitive"):
        recommendations.append("优先降低换手与成本假设的敏感性，再评估收益本身")
    if (consistency or {}).get("verdict") in {"weakly_consistent", "inconsistent"}:
        recommendations.append("在多个不同市况区间分别验证策略，勿依赖单一时段表现")
    if (rolling or {}).get("status") == "ok" and (rolling or {}).get("verdict") in {
        "mixed",
        "unstable",
    }:
        recommendations.append(
            f"滚动样本外结论为「{(rolling or {}).get('verdict')}」，"
            "优先研究收益的时段来源，再考虑扩展策略"
        )
    if rating == "READY" and not (stress_summary or {}).get("over_sensitive"):
        recommendations.append("当前证据下可进入下一阶段研究：多窗口滚动样本外验证（专业版功能）")
    if not recommendations:
        recommendations.append("维持当前研究流程，积累更多样本外证据后再评估")
    return recommendations


def _prepare(request, report_dir, emit=None):
    """加载数据、合成基准（如缺失）并完成数据审计。"""
    emit = emit or (lambda message: _log(request, message))
    emit("[1/6] 读取并审计数据...")
    loaded_data, metadata, source_info = load_analysis_data(request.cache, request.candidates)
    data = loaded_data
    candidate_info = source_info.pop("candidate_info", None)
    benchmark = BENCHMARK_CODE
    if benchmark not in data:
        data, benchmark = synthesize_benchmark(data)
        source_info.setdefault("warnings", []).append(
            "缺少真实沪深300行情，已生成等权合成基准 synthetic:equal_weight；"
            "报告中所有'基准'均为该合成序列"
        )
    input_sha = None
    try:
        input_sha = sha256_file(request.cache)
    except (OSError, ValueError):
        input_sha = None
    # 正式缓存旁若有证券主数据（含退市日期），审计按各股上市存续期评估覆盖率，
    # 并给出幸存者偏差风险结论；无主数据时审计如实降级为"数据不足"。
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
    audit["input_cache_sha256"] = input_sha
    return data, metadata, source_info, benchmark, audit, candidate_info


def _base_analysis(request, report_id, report_dir, data, metadata, source_info, benchmark, audit):
    analysis = {
        "report_id": report_id,
        "report_dir": report_dir,
        "request": request,
        "data_audit": audit,
        "input_cache_sha256": audit.get("input_cache_sha256"),
        "timings": {},
        "source_info": source_info,
        "data_source": str((metadata or {}).get("source") or "未声明"),
        "candidate_info": None,
        "benchmark_code": benchmark,
        "base": None,
        "stress_frame": None,
        "stress_summary": None,
        "segment_consistency": None,
        "factor_diagnostics": None,
        "score": None,
        "blocked": False,
        "block_message": None,
        "notes": [],
        "issues": [],
        "recommendations": [],
    }
    return analysis


def run_data_audit_only(request, report_id: str | None = None, progress=None) -> dict:
    """只做数据可信度审计并生成报告目录。"""
    emit = _make_emit(request, progress)
    report_dir, report_id = create_report_directory(request.output_root, report_id)
    emit(f"报告目录: {report_dir}")
    data, metadata, source_info, benchmark, audit, candidate_info = _prepare(
        request, report_dir, emit
    )
    analysis = _base_analysis(
        request, report_id, report_dir, data, metadata, source_info, benchmark, audit
    )
    analysis["candidate_info"] = candidate_info
    analysis["audit_only"] = True
    analysis["blocked"] = True
    analysis["block_message"] = "本次为纯数据审计运行，未执行回测。"
    analysis["score"] = compute_reliability_score(
        data_audit=audit,
        base_result=None,
        segment_consistency=None,
        stress_summary=None,
        manifest_complete=True,
        metadata_present=metadata is not None,
    )
    analysis["issues"] = _collect_issues(audit, None, None, None, analysis["score"])
    analysis["recommendations"] = _build_recommendations(audit, None, None, None, None)
    engine_config = build_engine_config(request)
    analysis["effective_config"] = {
        "engine_config": config_snapshot(engine_config),
        "benchmark_used": benchmark,
        "mode": "data_audit_only",
    }
    manifest = write_report_bundle(report_dir, analysis)
    analysis["manifest"] = manifest
    return analysis


def run_full_analysis(request, report_id: str | None = None, progress=None) -> dict:
    """完整商业分析：审计→回测→压力→一致性→滚动样本外→因子诊断→评分→报告。"""
    emit = _make_emit(request, progress)
    report_dir, report_id = create_report_directory(request.output_root, report_id)
    emit(f"报告目录: {report_dir}")

    try:
        data, metadata, source_info, benchmark, audit, candidate_info = _prepare(
            request, report_dir, emit
        )
    except DataInputError as exc:
        raise SystemExit(f"数据输入错误: {exc}") from exc

    analysis = _base_analysis(
        request, report_id, report_dir, data, metadata, source_info, benchmark, audit
    )
    analysis["candidate_info"] = candidate_info

    if audit["rating"] == "NOT_READY":
        message = "；".join(audit["blockers"]) or "数据质量不满足最低要求"
        analysis["blocked"] = True
        analysis["block_message"] = f"数据审计结论为NOT_READY：{message}"
        emit(f"[阻断] {analysis['block_message']}")
        analysis["score"] = compute_reliability_score(
            data_audit=audit,
            base_result=None,
            segment_consistency=None,
            stress_summary=None,
            manifest_complete=True,
            metadata_present=metadata is not None,
        )
        analysis["issues"] = _collect_issues(audit, None, None, None, analysis["score"])
        analysis["recommendations"] = _build_recommendations(audit, None, None, None, None)
        analysis["effective_config"] = {
            "engine_config": config_snapshot(build_engine_config(request)),
            "benchmark_used": benchmark,
            "mode": "blocked",
        }
        manifest = write_report_bundle(report_dir, analysis)
        analysis["manifest"] = manifest
        return analysis

    emit(f"[2/6] 数据评级 {audit['rating']}，准备回测缓存...")
    cache_path, staged = prepare_cache(request, data, benchmark, report_dir)
    if staged:
        analysis["notes"].append("CSV/导入数据已规范化为Alice缓存后回测；字段口径见数据质量附件。")
    # 商业入口用自己的数据审计做门禁（NOT_READY已在上方阻断）；
    # 引擎基于metadata质量标志的历史门禁在此放行，避免重复且口径不一致。
    if staged or not (metadata or {}).get("quality"):
        request.allow_incomplete_data = True
        analysis["notes"].append(
            "引擎metadata质量门禁由商业数据审计替代；审计结论见data_quality.json。"
        )
    config = build_engine_config(request)
    if benchmark != BENCHMARK_CODE:
        # 合成基准：让引擎用合成序列做门禁与对比，报告中明确披露。
        config = build_engine_config(request, benchmark_index=benchmark)
    # 压力测试的因子消融场景需要基准权重快照。
    request.extras["base_factor_weights"] = dict(config.factor_weights)

    allow_incomplete = request.allow_incomplete_data
    factor_strategy = request.strategy == FACTOR_STRATEGY
    if factor_strategy:
        factor_config = build_factor_strategy_config(request)
        emit("[3/6] 运行基准回测（截面Alpha158排名策略）...")
    else:
        emit("[3/6] 运行基准回测...")
    stage = perf_counter()
    base = None
    block_reason = None
    try:
        if factor_strategy:
            base = run_factor_strategy_backtest(request, cache_path, data=data)
        else:
            base = run_base_backtest(request, config, cache_path, data=data)
    except ValueError as exc:
        block_reason = str(exc)
    analysis["timings"]["base_backtest_seconds"] = round(perf_counter() - stage, 2)
    if base is None:
        analysis["blocked"] = True
        analysis["block_message"] = block_reason or (
            "回测无可用数据（区间内无交易日或股票池为空）。"
        )
        analysis["score"] = compute_reliability_score(
            data_audit=audit,
            base_result=None,
            segment_consistency=None,
            stress_summary=None,
            manifest_complete=True,
            metadata_present=metadata is not None,
        )
        analysis["issues"] = _collect_issues(audit, None, None, None, analysis["score"])
        analysis["recommendations"] = _build_recommendations(audit, None, None, None, None)
        analysis["effective_config"] = {
            "engine_config": config_snapshot(config),
            "benchmark_used": benchmark,
            "mode": "no_data",
        }
        manifest = write_report_bundle(report_dir, analysis)
        analysis["manifest"] = manifest
        return analysis
    base["benchmark"] = (
        data.get(benchmark, pd.DataFrame())[["date", "close"]] if benchmark in data else None
    )
    analysis["base"] = base

    # Deflated Sharpe：full模式按用户声明的试验方案数N（--trials，
    # 缺省1=退化为PSR）计算多重检验扣除；单一净值序列下试验Sharpe分散度未知，
    # 按论文用估计量。结果同时注入base.metrics（随metrics.json机读）。
    trials = int(getattr(request, "trials", 1) or 1)
    try:
        from commercial.overfitting import deflated_sharpe, equity_daily_returns

        dsr_result = deflated_sharpe(equity_daily_returns(base["equity"]), n_trials=trials)
        analysis["overfitting_audit"] = {
            "mode": "full",
            "n_trials": trials,
            "deflated_sharpe": dsr_result,
            "note": "DSR为观测Sharpe扣除N次试验多重检验后仍显著的概率，不是对未来收益的预测。",
        }
        base["metrics"]["deflated_sharpe"] = dsr_result.get("deflated_sharpe")
        base["metrics"]["deflated_sharpe_trials"] = (
            trials if dsr_result.get("status") == "ok" else None
        )
    except Exception as exc:  # DSR失败不阻断报告，如实标注
        analysis.setdefault("notes", []).append(f"Deflated Sharpe未能计算：{exc}")

    emit("[4/6] 运行压力测试与分段一致性...")
    stress_results = []
    stress_summary = {"status": "insufficient_data"}
    if request.run_stress:
        stage = perf_counter()
        stress_results = run_stress_tests(
            request,
            config,
            cache_path,
            verbose=not request.quiet,
            data=data,
            factor_config=factor_config if factor_strategy else None,
        )
        analysis["timings"]["stress_seconds"] = round(perf_counter() - stage, 2)
        stress_summary = sensitivity_summary(stress_results)
        analysis["stress_frame"] = scenario_frame(stress_results)
        analysis["stress_summary"] = stress_summary
        if factor_strategy:
            analysis["notes"].append(
                "截面Alpha158使用适配版压力场景：成本/滑点/择时/仓位/持仓数/"
                "动量秩消融；无买入延迟场景（截面引擎无该参数）。"
            )
    else:
        analysis["notes"].append("本次运行按用户选择跳过压力测试。")

    _factor_segment_runner = None
    if factor_strategy:

        def _factor_segment_runner(seg_start, seg_end):
            from engine.factor_backtest import run_factor_backtest

            return run_factor_backtest(
                data, seg_start, seg_end, config=factor_config, verbose=False
            )

    if request.run_segment_consistency:
        stage = perf_counter()
        consistency = run_segment_consistency(
            request,
            config,
            cache_path,
            data=data,
            run_function=_factor_segment_runner,
        )
        analysis["timings"]["segment_seconds"] = round(perf_counter() - stage, 2)
    else:
        consistency = {
            "status": "skipped",
            "detail": "用户选择跳过分段一致性检查（--no-segment），不输出时间稳定性结论",
        }
        analysis["notes"].append("本次运行按用户选择跳过前后分段一致性检查。")
    analysis["segment_consistency"] = consistency

    rolling = {"status": "insufficient_data", "detail": "未启用滚动样本外"}
    if request.run_rolling_oos:
        emit("    运行多窗口滚动样本外...")
        rolling = run_rolling_oos(
            request,
            config,
            cache_path,
            windows=request.rolling_windows,
            data=data,
            factor_config=factor_config if factor_strategy else None,
            anchored=bool(getattr(request, "rolling_anchored", False)),
        )
    analysis["rolling_oos"] = rolling

    feature_meta = (base or {}).get("feature_meta")
    if feature_meta:
        analysis["feature_meta"] = feature_meta
        if not feature_meta.get("signal_sections"):
            analysis["notes"].append(
                "截面策略没有产生任何可用信号截面（主板池/最低65日历史不满足），"
                "回测结果不含真实选股，不构成策略表现证据。"
            )

    if request.run_factor_diagnostics:
        emit("    运行因子诊断...")
        try:
            from commercial.factor_diagnostics import (
                compute_factor_diagnostics,
                render_factor_section_html,
            )

            summary, corr_matrix, factor_meta = compute_factor_diagnostics(
                data,
                request.start,
                request.end,
                benchmark=benchmark,
            )
            analysis["factor_diagnostics"] = summary
            analysis["factor_metadata"] = factor_meta
            analysis["factor_section_html"] = render_factor_section_html(
                summary, corr_matrix, factor_meta
            )
        except Exception as exc:  # 因子诊断失败不阻断报告，如实标注
            analysis["notes"].append(f"因子诊断未能完成：{exc}")
            analysis["factor_section_html"] = (
                f"<p class='muted'>因子诊断数据不足或计算失败（{exc}），不伪造结论。</p>"
            )
    else:
        analysis["notes"].append("本次运行按用户选择跳过因子诊断。")

    emit("[5/6] 计算策略可信度评分...")
    analysis["score"] = compute_reliability_score(
        data_audit=audit,
        base_result=base,
        segment_consistency=consistency,
        stress_summary=stress_summary,
        manifest_complete=True,
        metadata_present=metadata is not None,
        rolling_oos=rolling,
    )
    analysis["issues"] = _collect_issues(
        audit, base, stress_summary, consistency, analysis["score"]
    )
    analysis["recommendations"] = _build_recommendations(
        audit, base, stress_summary, consistency, rolling
    )
    if factor_strategy:
        strategy_config_snapshot = factor_config.to_dict()
        strategy_config_snapshot["strategy_params_user"] = dict(request.strategy_params or {})
    else:
        strategy_config_snapshot = config_snapshot(config)
    analysis["effective_config"] = {
        "engine_config": strategy_config_snapshot,
        "benchmark_used": benchmark,
        "staged_cache": staged,
        "allow_incomplete_data_used": allow_incomplete,
        "mode": "full_analysis",
    }

    emit("[6/6] 生成商业交付报告...")
    manifest = write_report_bundle(report_dir, analysis)
    analysis["manifest"] = manifest
    score = analysis["score"]
    _log(
        request,
        f"完成：可信度评分 {score['score'] if score['score'] is not None else '无法评分'}"
        f"（{score['grade_label']}）\n报告: {os.path.join(report_dir, 'index.html')}",
    )
    return analysis
