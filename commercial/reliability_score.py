"""策略可信度评分（Strategy Reliability Score, 0-100）。

七维评分：数据质量20 + 时间一致性20 + 样本外验证20 + 执行真实性15
+ 成本敏感性10 + 参数稳定性10 + 可复现性5。

原则：
- 收益率不是主要评分依据；
- 区分"未通过"（fail，计0分并给原因）与"缺少数据"（insufficient_data，
  从分母中剔除并在报告注明）；
- 数据质量不达标时整体评分封顶在0-49区间。
"""

from __future__ import annotations

GRADE_BANDS = [
    (85, 100, "research_grade", "研究证据较完整"),
    (70, 84, "continue_research", "可以继续研究"),
    (50, 69, "unstable", "结论不稳定"),
    (0, 49, "do_not_rely", "不应依据该回测做决策"),
]

NOT_READY_SCORE_CAP = 39

DIMENSION_SPECS = [
    ("data_quality", "数据质量", 20),
    ("temporal_integrity", "时间一致性与未来信息", 20),
    ("out_of_sample", "样本外验证", 20),
    ("execution_realism", "交易执行真实性", 15),
    ("cost_sensitivity", "成本与容量敏感性", 10),
    ("parameter_stability", "参数稳定性", 10),
    ("reproducibility", "可复现性", 5),
]

SCORE_DECLARATION = "该评分评价研究流程的可信度，不评价未来收益。"


def _dimension(key, earned, status, reasons, max_points=None):
    spec_max = dict((k, m) for k, _, m in DIMENSION_SPECS)[key]
    return {
        "key": key,
        "max_points": max_points or spec_max,
        "earned_points": round(float(earned), 1),
        "status": status,
        "reasons": list(reasons),
    }


def _score_data_quality(audit) -> dict:
    rating = (audit or {}).get("rating")
    coverage = (audit or {}).get("coverage", {})
    reasons = []
    if rating == "READY":
        reasons.append("数据通过正式回测门槛：覆盖、OHLC、字段完整度达标")
        return _dimension("data_quality", 20, "pass", reasons)
    if rating == "LIMITED":
        base = 10.0
        median = float(coverage.get("median_session_coverage") or 0)
        if median >= 0.95:
            base += 4
            reasons.append(f"中位会话覆盖率{median:.1%}，接近正式门槛")
        else:
            reasons.append(f"中位会话覆盖率{median:.1%}，低于正式门槛")
        fields = (audit or {}).get("fields", {})
        if not fields.get("historical_trade_status"):
            reasons.append("缺少历史停牌/ST字段，无法验证不现实成交")
        else:
            base += 2
        if not fields.get("has_real_benchmark"):
            reasons.append("缺少沪深300基准，使用等权合成基准")
        else:
            base += 2
        return _dimension("data_quality", base, "partial", reasons)
    reasons.extend((audit or {}).get("blockers", []) or ["数据质量未达最低要求"])
    return _dimension("data_quality", 0, "fail", reasons)


def _score_temporal_integrity(audit, base_result) -> dict:
    reasons = []
    points = 0.0
    # 结构性保证：T日信号、T+1开盘成交，引擎统一实现。
    points += 8
    reasons.append("引擎统一采用T日信号、T+1开盘成交，无同Bar成交")
    fields = (audit or {}).get("fields", {})
    if fields.get("historical_trade_status"):
        points += 6
        reasons.append("历史停牌/ST字段完整，可逐日验证可交易性")
    else:
        reasons.append("缺少历史停牌/ST字段，存在未来信息或不现实成交风险")
    metrics = (base_result or {}).get("metrics", {})
    unsold = int(metrics.get("unsold_final_positions") or 0)
    if unsold == 0 and metrics:
        points += 3
        reasons.append("期末清仓全部真实可执行")
    elif metrics:
        reasons.append(f"期末有{unsold}个持仓无法真实卖出")
    if (audit or {}).get("rating") == "READY":
        points += 3
        reasons.append("数据审计未发现边界缺失或异常OHLC")
    else:
        reasons.append("数据审计存在限制，未来信息检查只能部分验证")
    return _dimension("temporal_integrity", points, "pass" if points >= 17 else "partial", reasons)


def _score_out_of_sample(consistency, rolling=None) -> dict:
    rolling_status = (rolling or {}).get("status")
    if rolling_status == "ok":
        verdict = rolling.get("verdict")
        window_kind = "锚定递进窗口" if rolling.get("anchored") else "不重叠窗口"
        reasons = [
            rolling.get("detail", ""),
            f"共{len(rolling.get('windows') or [])}个{window_kind}，"
            f"正收益占比{rolling.get('positive_share', 0):.0%}，"
            f"Sharpe离散度σ={rolling.get('sharpe_std', 0):.2f}",
            rolling.get("note", ""),
        ]
        if verdict == "stable":
            return _dimension("out_of_sample", 20, "pass", reasons)
        if verdict == "mixed":
            return _dimension("out_of_sample", 12, "partial", reasons)
        return _dimension("out_of_sample", 4, "fail", reasons)
    status = (consistency or {}).get("status")
    if status in {"insufficient_data", "skipped"}:
        return _dimension(
            "out_of_sample",
            0,
            "insufficient_data",
            [consistency.get("detail", "分段样本不足，无法评估")],
        )
    verdict = consistency.get("verdict")
    reasons = [consistency.get("detail", ""), consistency.get("note", "")]
    if verdict == "consistent":
        return _dimension("out_of_sample", 20, "pass", reasons)
    if verdict == "weakly_consistent":
        return _dimension("out_of_sample", 12, "partial", reasons)
    return _dimension("out_of_sample", 4, "fail", reasons)


def _score_execution_realism(audit, base_result) -> dict:
    reasons = []
    fields = (audit or {}).get("fields", {})
    if not fields.get("historical_trade_status"):
        return _dimension(
            "execution_realism",
            5,
            "partial",
            ["缺少停牌/涨跌停历史字段，只能按保守规则近似，执行真实性无法完全验证"],
        )
    metrics = (base_result or {}).get("metrics", {})
    if not metrics:
        return _dimension("execution_realism", 0, "insufficient_data", ["回测未运行"])
    points = 10.0
    reasons.append("挂单经停牌/涨跌停门禁校验，按开盘价成交")
    unsold = int(metrics.get("unsold_final_positions") or 0)
    if unsold == 0:
        points += 3
        reasons.append("期末无滞留持仓")
    else:
        reasons.append(f"期末{unsold}个持仓无法卖出，已按最后已知价估值并披露")
    stats = (base_result or {}).get("statistics", {})
    turnover = float(stats.get("annualized_turnover") or 0)
    if 0 < turnover <= 60:
        points += 2
        reasons.append(f"年化换手{turnover:.1f}倍，处于可交易范围")
    else:
        reasons.append(f"年化换手{turnover:.1f}倍偏高，实际成交可能低于回测假设")
    return _dimension("execution_realism", points, "pass" if points >= 13 else "partial", reasons)


def _score_cost_sensitivity(stress_summary) -> dict:
    status = (stress_summary or {}).get("status")
    if status != "ok":
        return _dimension(
            "cost_sensitivity",
            0,
            "insufficient_data",
            ["未运行压力测试，无法评估成本敏感性"],
        )
    ranking = stress_summary.get("ranking") or []
    cost_related = [
        item
        for item in ranking
        if item.get("scenario") in {"cost_up_50", "cost_up_100", "slippage_x3"}
    ]
    if not cost_related:
        return _dimension(
            "cost_sensitivity",
            0,
            "insufficient_data",
            ["压力场景未覆盖成本变化"],
        )
    worst = max(abs(float(item["return_delta_pct"])) for item in cost_related)
    any_flip = any(item.get("sign_flipped") for item in cost_related)
    reasons = [
        f"{item['label']}: 收益变化{item['return_delta_pct']:+.2f}个百分点" for item in cost_related
    ]
    if any_flip:
        reasons.append("存在成本上升后收益方向翻转的场景")
        return _dimension("cost_sensitivity", 2, "fail", reasons)
    if worst <= 10:
        return _dimension("cost_sensitivity", 10, "pass", reasons)
    if worst <= 25:
        return _dimension("cost_sensitivity", 6, "partial", reasons)
    return _dimension("cost_sensitivity", 3, "partial", reasons + ["成本假设变化影响显著"])


def _score_parameter_stability(stress_summary, consistency) -> dict:
    """以预定义场景(仓位/择时/延迟)与分段一致性近似参数稳定性。

    真正的多窗口滚动样本外属于专业版功能；此处不做参数搜索。
    """
    parts = []
    reasons = []
    status = (stress_summary or {}).get("status")
    if status == "ok":
        structural = [
            item
            for item in (stress_summary.get("ranking") or [])
            if item.get("scenario")
            in {"no_market_timing", "smaller_position", "fewer_positions", "buy_delay_1d"}
        ]
        if structural:
            worst = max(abs(float(item["return_delta_pct"])) for item in structural)
            points = 10.0 if worst <= 15 else 6.0 if worst <= 35 else 2.0
            reasons.extend(
                f"{item['label']}: 收益变化{item['return_delta_pct']:+.2f}个百分点"
                for item in structural
            )
            parts.append(points)
        else:
            reasons.append("缺少结构性压力场景")
            parts.append(5.0)
    else:
        return _dimension(
            "parameter_stability",
            0,
            "insufficient_data",
            ["未运行压力场景，无法评估参数稳定性"],
        )
    if (consistency or {}).get("status") == "ok":
        if consistency.get("verdict") == "consistent":
            parts.append(0.0)
        elif consistency.get("verdict") == "weakly_consistent":
            parts.append(-2.0)
            reasons.append("前后半段Sharpe差异明显")
        else:
            parts.append(-4.0)
            reasons.append("前后半段收益方向相反")
    return _dimension(
        "parameter_stability",
        max(0.0, sum(parts)),
        "pass" if sum(parts) >= 8 else "partial",
        reasons,
    )


def _score_reproducibility(manifest_complete: bool, metadata_present: bool) -> dict:
    reasons = []
    points = 0.0
    if manifest_complete:
        points += 3
        reasons.append("报告包含输入/参数/产物SHA-256哈希与唯一report_id")
    else:
        reasons.append("清单不完整，哈希记录缺失")
    if metadata_present:
        points += 2
        reasons.append("数据来源元数据完整")
    else:
        reasons.append("数据缺少来源元数据，只能部分追溯")
    return _dimension("reproducibility", points, "pass" if points >= 4 else "partial", reasons)


def compute_reliability_score(
    *,
    data_audit: dict,
    base_result: dict | None,
    segment_consistency: dict | None,
    stress_summary: dict | None,
    manifest_complete: bool = False,
    metadata_present: bool = False,
    rolling_oos: dict | None = None,
) -> dict:
    """汇总七维评分；返回机器可读结构。"""
    dimensions = [
        _score_data_quality(data_audit),
        _score_temporal_integrity(data_audit, base_result or {}),
        _score_out_of_sample(segment_consistency or {}, rolling_oos),
        _score_execution_realism(data_audit, base_result or {}),
        _score_cost_sensitivity(stress_summary or {}),
        _score_parameter_stability(stress_summary or {}, segment_consistency or {}),
        _score_reproducibility(manifest_complete, metadata_present),
    ]
    applicable_max = sum(d["max_points"] for d in dimensions if d["status"] != "insufficient_data")
    earned = sum(d["earned_points"] for d in dimensions if d["status"] != "insufficient_data")
    insufficient = [d["key"] for d in dimensions if d["status"] == "insufficient_data"]
    failed = [d["key"] for d in dimensions if d["status"] == "fail"]

    rating = (data_audit or {}).get("rating")
    score = None
    if applicable_max >= 40:
        raw = earned / applicable_max * 100.0
        score = int(round(raw))
        if rating == "NOT_READY":
            score = min(score, NOT_READY_SCORE_CAP)
    grade_key, grade_label = "insufficient", "数据不足，无法评分"
    if score is not None:
        for low, high, key, label in GRADE_BANDS:
            if low <= score <= high:
                grade_key, grade_label = key, label
                break
    return {
        "score": score,
        "grade": grade_key,
        "grade_label": grade_label,
        "applicable_max_points": applicable_max,
        "earned_points": round(earned, 1),
        "insufficient_dimensions": insufficient,
        "failed_dimensions": failed,
        "dimensions": dimensions,
        "data_rating": rating,
        "declaration": SCORE_DECLARATION,
        "usage_note": "评分用于决定'是否值得继续研究该策略'，不是收益预测，不构成投资建议。",
    }
