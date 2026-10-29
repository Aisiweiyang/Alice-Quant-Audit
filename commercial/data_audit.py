"""数据可信度审计：覆盖范围、缺失、异常OHLC、字段完整度与幸存者偏差风险。

复用 data/quality.py 的逐股审计，并输出 READY / LIMITED / NOT_READY 评级。
缺少数据时明确标注"数据不足"，不伪造结论。
"""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data.cache_store import build_metadata, has_historical_trade_status
from data.quality import audit_market_data

from commercial.data_input import BENCHMARK_CODE

MIN_BACKTEST_SESSIONS = 30
MIN_BACKTEST_STOCKS = 3


def _valuation_coverage(data: dict, benchmark: str) -> float:
    cells = 0
    values = 0
    for code, frame in data.items():
        if code == benchmark:
            continue
        for column in ("pe", "pb"):
            if column in frame.columns:
                cells += len(frame)
                values += pd.to_numeric(frame[column], errors="coerce").notna().sum()
    return float(values / cells) if cells else 0.0


def _delisted_coverage(data: dict, benchmark: str, security_master, start=None, end=None) -> dict:
    """退市股票覆盖情况：缺少证券主数据时如实报告"数据不足"。

    覆盖率的分母只统计审计窗口内退市（outDate落在[start,end]）的股票：
    窗口开始前已退市的股票本就不可能出现在窗口股票池中，全部历史退市股
    计入分母会把覆盖率错判为过低；指数等非股票行（type≠1，若提供type列）
    同样不计入。start/end为None时退回全期口径（兼容旧调用）。
    """
    if security_master is None:
        return {
            "status": "insufficient_data",
            "detail": "缺少证券主数据（含退市日期），无法评估退市股票覆盖",
        }
    frame = security_master
    if not isinstance(frame, pd.DataFrame) or "outDate" not in frame.columns:
        return {"status": "insufficient_data", "detail": "证券主数据缺少outDate字段"}
    master = frame.copy()
    master["outDate"] = pd.to_datetime(master["outDate"], errors="coerce")
    if "type" in master.columns:
        master = master[master["type"].astype(str).str.strip() == "1"]
    delisted = master[master["outDate"].notna()]
    if start is not None:
        delisted = delisted[delisted["outDate"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        delisted = delisted[delisted["outDate"] <= pd.Timestamp(end).normalize()]
    delisted_codes = set(str(code) for code in delisted["code"])
    in_universe = [code for code in data if code != benchmark and code in delisted_codes]
    if not delisted_codes:
        return {
            "status": "warning",
            "detail": "证券主数据中没有退市股票，股票池可能只含当前存续股票",
        }
    ratio = len(in_universe) / max(len(delisted_codes), 1)
    detail = (
        f"数据期内已退市股票{len(delisted_codes)}只，其中{len(in_universe)}只包含在数据中"
        f"（覆盖比例{ratio:.1%}）"
    )
    if ratio < 0.3:
        return {"status": "fail", "detail": detail + "；退市覆盖过低，存在幸存者偏差风险"}
    if ratio < 0.7:
        return {"status": "warning", "detail": detail + "；部分退市股票缺失"}
    return {"status": "pass", "detail": detail}


DELISTING_TAIL_GRACE_DAYS = 30


def _mark_suspected_delisting_tail(stocks: list, security_master, warnings: list) -> None:
    """退市尾段缺失细分（透明度增强，不改pass/fail判定与评级口径）。

    对"头部完整覆盖窗口起点、却在窗口内提前终止"的fail股票，若主数据
    outDate缺失或与实际终止日接近（≤30自然日），在条目上标记
    suspected_delisting_tail=True并汇总一条警告——这类fail大概率是退市
    导致的尾段缺失，而非数据质量缺陷；outDate远离数据终止日的仍视为
    普通覆盖缺口（不标记），宁可保守不误标。
    """
    if security_master is None or not isinstance(security_master, pd.DataFrame):
        return
    if not {"code", "outDate"}.issubset(security_master.columns):
        return
    master = security_master.copy()
    master["outDate"] = pd.to_datetime(master["outDate"], errors="coerce")
    out_dates = {str(code): out for code, out in zip(master["code"], master["outDate"])}
    suspected: list[str] = []
    for row in stocks:
        if row.get("status") != "fail" or not row.get("actual_end"):
            continue
        expected_start = pd.Timestamp(row["expected_start"])
        expected_end = pd.Timestamp(row["expected_end"])
        actual_start = pd.Timestamp(row["actual_start"]) if row.get("actual_start") else None
        actual_end = pd.Timestamp(row["actual_end"])
        if actual_start is None or actual_start > expected_start + pd.Timedelta(days=10):
            continue  # 头部就没覆盖：属普通数据缺口，不是退市尾段
        if actual_end >= expected_end - pd.Timedelta(days=10):
            continue  # 尾部覆盖到窗口末：不缺尾段
        out = out_dates.get(str(row["code"]))
        if out is None or pd.isna(out):
            near_out = True  # 主数据缺退市日：无法排除退市可能
        else:
            near_out = abs((out - actual_end).days) <= DELISTING_TAIL_GRACE_DAYS
        if near_out:
            row["suspected_delisting_tail"] = True
            suspected.append(str(row["code"]))
    if suspected:
        preview = "、".join(suspected[:10]) + ("等" if len(suspected) > 10 else "")
        warnings.append(
            f"疑似退市尾段缺失{len(suspected)}只（{preview}）：数据在窗口内提前终止，"
            "该时段收益口径存疑；不影响评级判定"
        )


def audit_data_credibility(
    data: dict,
    *,
    start: str,
    end: str,
    benchmark: str = BENCHMARK_CODE,
    security_master=None,
    metadata=None,
    source_info=None,
    minimum_coverage: float = 0.90,
) -> dict:
    """输出机器可读的数据可信度报告与质量评级。"""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    has_real_benchmark = benchmark in data
    stock_codes = [code for code in data if code != benchmark]

    audit_input = data
    if not has_real_benchmark:
        # CSV等来源可能没有基准：用全市场日期并集构造审计日历，保证覆盖率可计算。
        union_dates = sorted(
            {
                pd.Timestamp(d).normalize()
                for frame in data.values()
                for d in pd.to_datetime(frame["date"], errors="coerce").dropna()
            }
        )
        calendar_frame = pd.DataFrame({"date": union_dates, "close": 1.0})
        audit_input = dict(data)
        audit_input[benchmark] = calendar_frame

    audit = audit_market_data(
        audit_input,
        benchmark=benchmark,
        start=start_ts,
        end=end_ts,
        requested_codes=stock_codes,
        security_master=security_master,
        minimum_coverage=minimum_coverage,
    )
    stocks = audit["stocks"]
    coverage_values = [row["session_coverage"] for row in stocks]
    invalid_total = sum(int(row.get("invalid_ohlc_rows") or 0) for row in stocks)
    missing_sessions_total = sum(
        int(row["missing_sessions"] or 0)
        for row in stocks
        if row.get("missing_sessions") is not None
    )

    # 导入数据若用占位字段补齐停牌/ST，则视为"无历史字段"，不得宣称可验证成交。
    provenance = (source_info or {}).get("field_provenance") or {}
    fabricated_status_codes = [
        code for code, fields in provenance.items() if "tradestatus" in fields or "isST" in fields
    ]
    trade_status_complete = (
        has_historical_trade_status(data, benchmark) and not fabricated_status_codes
    )
    valuation = _valuation_coverage(data, benchmark)
    delisted = _delisted_coverage(data, benchmark, security_master, start=start_ts, end=end_ts)

    # 区间内交易日数（用基准日历或全市场并集近似）。
    all_dates = set()
    for frame in data.values():
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        all_dates.update(dates[(dates >= start_ts) & (dates <= end_ts)].dt.normalize())
    sessions = len(all_dates)

    blockers: list[str] = []
    warnings: list[str] = []
    if not stocks:
        blockers.append("区间内没有可审计的股票数据")
    if sessions < MIN_BACKTEST_SESSIONS:
        blockers.append(f"区间内交易日仅{sessions}天（<{MIN_BACKTEST_SESSIONS}），无法支撑回测")
    if len(stock_codes) < MIN_BACKTEST_STOCKS:
        blockers.append(f"股票数量仅{len(stock_codes)}只（<{MIN_BACKTEST_STOCKS}）")
    median_coverage = float(pd.Series(coverage_values).median()) if coverage_values else 0.0
    if median_coverage < 0.5:
        blockers.append(f"中位会话覆盖率仅{median_coverage:.1%}，数据缺口过大")

    failed = audit["summary"].get("failed_codes", 0)
    if failed:
        warnings.append(f"{failed}只股票未通过覆盖率/OHLC检查")
    if fabricated_status_codes:
        warnings.append(
            f"{len(fabricated_status_codes)}只股票的停牌/ST字段为导入占位值，不构成历史真实数据"
        )
    if not trade_status_complete:
        warnings.append("缺少完整历史停牌/ST/preclose字段，无法验证不现实成交")
    if valuation < 0.25:
        warnings.append(f"PE/PB覆盖率仅{valuation:.1%}，估值过滤不可靠")
    if not has_real_benchmark:
        warnings.append("缺少沪深300基准，仅能使用等权合成基准")
    if delisted["status"] == "insufficient_data":
        warnings.append(delisted["detail"])
    elif delisted["status"] == "fail":
        warnings.append(delisted["detail"])
        blockers.append("退市股票覆盖过低，存在幸存者偏差")
    elif delisted["status"] == "warning":
        warnings.append(delisted["detail"])
    for warning in (source_info or {}).get("warnings", []):
        warnings.append(warning)
    # 退市尾段缺失细分（透明度增强）：放最后，警告顺序=构建顺序的补充说明。
    _mark_suspected_delisting_tail(stocks, security_master, warnings)

    formal_ready = bool(audit["summary"].get("formal_ready"))
    if blockers:
        rating = "NOT_READY"
    elif formal_ready and trade_status_complete and has_real_benchmark and valuation >= 0.25:
        rating = "READY"
    else:
        rating = "LIMITED"

    result = {
        "rating": rating,
        "rating_definition": {
            "READY": "适合正式回测",
            "LIMITED": "可做研究但结论需保留限制说明",
            "NOT_READY": "数据缺陷会扭曲结论，正式回测被阻断",
        },
        "coverage": {
            "start": start_ts.date().isoformat(),
            "end": end_ts.date().isoformat(),
            "trade_sessions_in_range": sessions,
            "stock_count": len(stock_codes),
            "median_session_coverage": round(median_coverage, 6),
            "missing_sessions_total": missing_sessions_total,
            "invalid_ohlc_rows_total": invalid_total,
            "duplicate_rows_total": _count_duplicates(data, benchmark),
        },
        "fields": {
            "historical_trade_status": bool(trade_status_complete),
            "valuation_coverage": round(valuation, 6),
            "has_real_benchmark": bool(has_real_benchmark),
            "delisted_coverage": delisted,
        },
        "survivorship_bias_risk": {
            "level": "high"
            if delisted["status"] == "fail"
            else "unknown"
            if delisted["status"] == "insufficient_data"
            else "medium"
            if delisted["status"] == "warning"
            else "low",
            "detail": delisted["detail"],
        },
        "blockers": blockers,
        "warnings": warnings,
        "summary": audit["summary"],
        "stocks": stocks,
        "disclaimer": "数据审计只描述输入数据质量，不预测任何收益。",
    }
    return result


def _count_duplicates(data: dict, benchmark: str) -> int:
    """统计(已规范化的)缓存内同日重复记录；Alice缓存加载时已去重，通常为0。"""
    total = 0
    for code, frame in data.items():
        if code == benchmark:
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        total += int(len(dates) - dates.nunique())
    return total


def build_cache_metadata_snapshot(data: dict, benchmark: str) -> dict:
    """为CSV等无元数据来源生成 Alice 兼容的元数据快照。"""
    return build_metadata(
        data,
        benchmark=benchmark,
        adjust="unknown",
        universe_mode="file",
        source="commercial_csv_import",
    )


def stage_cache_for_backtest(data: dict, benchmark: str, target_dir: str) -> str:
    """把已加载数据落成Alice缓存，供回测引擎直接使用；返回缓存路径。"""
    from data.cache_store import save_market_cache

    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "staged_input_cache.pkl")
    if os.path.exists(target):
        # 同一报告目录内重用已生成的缓存，不覆盖其他报告。
        return target
    metadata = build_metadata(
        data,
        benchmark=benchmark,
        adjust="unknown",
        universe_mode="file",
        source="commercial_import",
        warnings=["由商业化MVP导入的数据，字段口径以数据审计结论为准。"],
    )
    save_market_cache(target, data, metadata)
    return target
