"""qlib式实验索引（简化版）：报告根目录的追加式摘要清单。

每次报告落盘后追加一行摘要（report_id/时间/策略/区间/评分/数据评级/关键指标），
便于跨报告检索证据。只追加不删除；索引损坏或写失败时如实跳过，绝不影响
报告本体，也不改写已有索引历史。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

# 进入索引摘要的关键指标键（full模式；compare模式指标在各方案下，摘要只记计数）。
INDEX_METRIC_KEYS = (
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sharpe_ci_includes_zero",
    "total_trades",
    "deflated_sharpe",
    "omega_ratio",
)


def experiment_entry(analysis: dict) -> dict:
    """从分析结果抽取一行摘要；缺字段如实None，不编造。"""
    base = analysis.get("base") or {}
    metrics = base.get("metrics") or {}
    score = analysis.get("score") or {}
    audit = analysis.get("data_audit") or {}
    request = analysis.get("request")
    manifest = analysis.get("manifest") or {}
    mode = analysis.get("mode") or "full"
    plans = analysis.get("plan_results") or []
    entry = {
        "report_id": analysis.get("report_id"),
        "created_at": manifest.get("created_at")
        or datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "strategy": getattr(request, "strategy", None),
        "start": getattr(request, "start", None),
        "end": getattr(request, "end", None),
        "score": score.get("score"),
        "data_rating": audit.get("rating"),
        "blocked": bool(analysis.get("blocked")),
        "key_metrics": {key: metrics.get(key) for key in INDEX_METRIC_KEYS},
    }
    if mode == "compare":
        ok = sum(1 for r in plans if r.get("status") == "ok")
        overfit = analysis.get("overfitting_audit") or {}
        pbo = overfit.get("pbo") or {}
        entry["plans_ok"] = f"{ok}/{len(plans)}"
        entry["n_trials"] = overfit.get("n_trials")
        entry["pbo_probability"] = pbo.get("pbo_probability")
    return entry


def append_experiment_index(report_dir: str, analysis: dict) -> str | None:
    """把本次报告摘要追加到报告根目录experiments_index.json；返回索引路径。

    - 只追加不删除：既有记录原样保留；
    - 原子写（临时文件+os.replace），避免并发/中断产生半截文件；
    - 索引不存在则创建；已损坏（非JSON/非列表）时如实跳过并返回None，
      绝不改写历史——宁可缺失本次摘要，不可破坏既往证据；
    - 任何失败只返回None，调用方无需也无法因此中断报告流程。
    """
    try:
        root = os.path.dirname(os.path.abspath(report_dir))
        if not root or not os.path.isdir(root):
            return None
        path = os.path.join(root, "experiments_index.json")
        records: list = []
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                return None  # 损坏：不覆盖历史
            if not isinstance(loaded, list):
                return None
            records = loaded
        records.append(experiment_entry(analysis))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return path
    except OSError:
        return None
