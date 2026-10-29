"""黄金基线构建与对比（参考backtesting.py的金文件测试实践）。

固定 demo_cache + 固定区间 + 固定参数跑完整 full 管线，抽取确定性核心结果
（评分、区间收益、Sharpe及其CI端点、交易数、新增收益质量指标、回撤明细、
各图存在性、manifest结构）作为黄金基线。目的：未来任何回测数值口径的
无意改动会立即在 tests/test_golden_baseline.py 红灯。

- 浮点字段按 1e-9 容差比较；其余字段（整数/布尔/字符串/结构）精确相等；
- 有意口径变更时用 scripts/update_golden.py --confirm 重新生成，
  并在提交信息中声明 "GOLDEN UPDATE: <原因>"（一次提交只允许一个口径变更）；
- 黄金文件只收录平台无关内容：图表仅记存在性与数量（PNG字节随渲染环境
  字体/版本变化，不宜跨环境哈希），不收录report_id/时间戳/git哈希/耗时。
"""

import hashlib
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(PROJECT_DIR, "tests", "golden")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "golden_full.json")

GOLDEN_PARAMS = {
    "cache": "examples/commercial_demo/demo_cache.pkl",
    "start": "2025-07-01",
    "end": "2025-12-31",
    "strategy": "classic_trend",
    # 其余参数全部取 CommercialRequest 默认值（压力/滚动/分段/因子诊断全开），
    # 与 scripts/run_commercial_demo.ps1 的策略一段口径一致。
}

# 进入黄金基线的指标字段（metrics.json子集；全部为确定性输出）。
GOLDEN_METRIC_KEYS = (
    "total_return_pct",
    "cagr_pct",
    "max_drawdown_pct",
    "annual_volatility_pct",
    "sharpe_ratio",
    "sharpe_observations",
    "sharpe_reliable",
    "sharpe_ci_low",
    "sharpe_ci_high",
    "sharpe_ci_includes_zero",
    "sortino_ratio",
    "calmar_ratio",
    "max_consecutive_losses",
    "max_consecutive_wins",
    "alpha_annualized_pct",
    "beta",
    "information_ratio",
    "tail_ratio",
    "up_capture_pct",
    "down_capture_pct",
    "recovery_factor",
    "benchmark_return_pct",
    "excess_return_pct",
    "total_trades",
    "win_rate_pct",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "unsold_final_positions",
)


def _resolve_cache() -> str:
    return os.path.join(PROJECT_DIR, GOLDEN_PARAMS["cache"])


def dataset_fingerprint(data) -> str:
    """Hash table values, independent of pandas' pickle representation.

    Raw cache byte hashes remain in each report manifest. This portable golden
    anchor checks the generated data itself across supported dependency versions.
    """
    parts = []
    for code in sorted(data):
        frame = data[code].sort_values("date").reindex(sorted(data[code].columns), axis=1)
        parts.extend(
            [
                code,
                frame.to_csv(
                    index=False, float_format="%.12g", date_format="%Y-%m-%d", lineterminator="\n"
                ),
            ]
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def build_golden(output_root: str) -> dict:
    """跑一次完整full管线并抽取黄金基线内容（不写文件）。"""
    from data.cache_store import load_market_cache
    from commercial.config import CommercialRequest
    from commercial.pipeline import run_full_analysis

    cache = _resolve_cache()
    request = CommercialRequest(
        cache=cache,
        start=GOLDEN_PARAMS["start"],
        end=GOLDEN_PARAMS["end"],
        output_root=output_root,
        quiet=True,
    )
    request.validate()
    # report_id用默认唯一编号：固定id会撞报告目录防覆盖（历轮教训），
    # 黄金基线内容不含report_id，编号变化无影响。
    analysis = run_full_analysis(request)
    base = analysis.get("base") or {}
    metrics = base.get("metrics") or {}
    score = analysis.get("score") or {}
    segment = analysis.get("segment_consistency") or {}
    rolling = analysis.get("rolling_oos") or {}
    diagnostics = analysis.get("factor_diagnostics")
    audit = analysis.get("data_audit") or {}
    manifest = analysis.get("manifest") or {}

    data, metadata = load_market_cache(cache, "sh.000300")
    ok_factors = 0
    if diagnostics is not None and not diagnostics.empty and "status" in diagnostics.columns:
        ok_factors = int((diagnostics["status"] == "ok").sum())

    return {
        "pipeline": {
            "mode": "full",
            "start": GOLDEN_PARAMS["start"],
            "end": GOLDEN_PARAMS["end"],
            "strategy": GOLDEN_PARAMS["strategy"],
            "dataset_sha256": dataset_fingerprint(data),
            "cache_metadata_source": (metadata or {}).get("source"),
        },
        "metrics": {key: metrics.get(key) for key in GOLDEN_METRIC_KEYS},
        "drawdown_periods": metrics.get("drawdown_periods"),
        "score": {
            "score": score.get("score"),
            "grade_label": score.get("grade_label"),
            "failed_dimensions": sorted(score.get("failed_dimensions") or []),
            "insufficient_dimensions": sorted(score.get("insufficient_dimensions") or []),
            "dimensions": [
                {k: d.get(k) for k in ("key", "status", "earned_points", "max_points")}
                for d in score.get("dimensions") or []
            ],
        },
        "data_rating": audit.get("rating"),
        "segment": {
            "status": segment.get("status"),
            "verdict": segment.get("verdict"),
            "sharpe_gap": segment.get("sharpe_gap"),
        },
        "rolling": {
            "status": rolling.get("status"),
            "verdict": rolling.get("verdict"),
            "positive_share": rolling.get("positive_share"),
            "ci_zero_share": rolling.get("ci_zero_share"),
            "ok_windows": len([w for w in rolling.get("windows") or [] if w.get("status") == "ok"]),
        },
        "factor_diagnostics": {
            "ok_factor_count": ok_factors,
            "status": None if diagnostics is None else "present",
        },
        "charts_written": sorted(analysis.get("charts_written") or []),
        "manifest_keys": sorted(manifest.keys()),
        "manifest_output_names": sorted((manifest.get("outputs_sha256") or {}).keys()),
    }


def diff_golden(golden, actual, path="$") -> list:
    """递归比较黄金基线与实际结果；返回差异描述列表（空列表=完全一致）。"""
    differences = []
    if isinstance(golden, dict) and isinstance(actual, dict):
        for key in sorted(set(golden) | set(actual)):
            if key not in golden:
                differences.append(f"{path}.{key}: 黄金基线缺失该键（新增）")
            elif key not in actual:
                differences.append(f"{path}.{key}: 实际结果缺失该键")
            else:
                differences.extend(diff_golden(golden[key], actual[key], f"{path}.{key}"))
    elif isinstance(golden, list) and isinstance(actual, list):
        if len(golden) != len(actual):
            differences.append(f"{path}: 长度不同 golden={len(golden)} actual={len(actual)}")
        for index, (g, a) in enumerate(zip(golden, actual)):
            differences.extend(diff_golden(g, a, f"{path}[{index}]"))
    elif isinstance(golden, bool) or isinstance(actual, bool):
        # bool与int在Python中True==1，必须连类型一起比较，防止布尔口径漂移。
        if golden != actual or type(golden) is not type(actual):
            differences.append(f"{path}: golden={golden!r} actual={actual!r}")
    elif isinstance(golden, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(golden) - float(actual)) > 1e-9:
            differences.append(f"{path}: golden={golden!r} actual={actual!r}")
    elif golden != actual:
        differences.append(f"{path}: golden={golden!r} actual={actual!r}")
    return differences
