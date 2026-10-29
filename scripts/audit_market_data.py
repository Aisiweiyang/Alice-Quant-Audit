#!/usr/bin/env python3
"""Audit an existing trusted Alice cache without downloading market data."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from data.cache_store import (
    has_historical_trade_status,
    load_market_cache,
    metadata_path,
)
from data.quality import audit_market_data, save_quality_report
from data.universe import load_security_master
from strategy.config import StrategyConfig as cfg


def _atomic_metadata_update(cache_path, metadata):
    path = metadata_path(cache_path)
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description="离线审计 Alice 行情缓存质量")
    parser.add_argument("--cache", required=True, help="受信任的本地pickle缓存")
    parser.add_argument("--start", help="覆盖元数据中的开始日期")
    parser.add_argument("--end", help="覆盖元数据中的结束日期")
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 < args.minimum_coverage <= 1:
        raise SystemExit("minimum-coverage必须在0到1之间")

    data, metadata = load_market_cache(args.cache, cfg.benchmark_index)
    if metadata is None:
        raise SystemExit("缓存缺少metadata.json，无法确认来源和请求区间")
    period = metadata.get("requested_period") or metadata.get("period") or {}
    start = args.start or period.get("start")
    end = args.end or period.get("end")
    if not start or not end:
        raise SystemExit("无法确定审计区间，请传入 --start 和 --end")
    universe = metadata.get("universe") or {}
    codes = universe.get("requested_codes") or [
        code for code in data if code != cfg.benchmark_index
    ]
    security_master = load_security_master(args.cache)
    report = audit_market_data(
        data,
        benchmark=cfg.benchmark_index,
        start=start,
        end=end,
        requested_codes=codes,
        security_master=security_master,
        minimum_coverage=args.minimum_coverage,
    )
    json_path, csv_path = save_quality_report(args.cache, report)
    metadata["quality"] = report["summary"]
    metadata.setdefault("capabilities", {})["historical_trade_status"] = (
        has_historical_trade_status(data, cfg.benchmark_index)
    )
    _atomic_metadata_update(args.cache, metadata)

    summary = report["summary"]
    print(f"审计股票: {summary['audited_codes']}")
    print(f"通过/失败: {summary['passed_codes']}/{summary['failed_codes']}")
    print(f"中位交易日覆盖率: {summary['median_session_coverage']:.1%}")
    print(f"正式回测就绪: {'是' if summary['formal_ready'] else '否'}")
    print(f"质量报告: {json_path}")
    print(f"逐股明细: {csv_path}")
    return 0 if summary["formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
