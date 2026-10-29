#!/usr/bin/env python3
"""Seed the long-history cache from a compatible, audited recent cache."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data.cache_store import load_market_cache, merge_market_data
from data.universe import load_security_master
from scripts.update_market_data import _write_checkpoint
from strategy.config import StrategyConfig as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="把兼容的近期正式缓存安全合并到长期缓存",
    )
    parser.add_argument("--target", required=True, help="长期缓存路径")
    parser.add_argument("--recent", required=True, help="近期正式缓存路径")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


def validate_compatibility(target_metadata, recent_metadata):
    for key in ("source", "adjust"):
        target_value = str((target_metadata or {}).get(key) or "")
        recent_value = str((recent_metadata or {}).get(key) or "")
        if not target_value or target_value != recent_value:
            raise ValueError(f"缓存{key}不兼容: target={target_value!r}, recent={recent_value!r}")
    recent_quality = (recent_metadata or {}).get("quality") or {}
    if recent_quality.get("formal_ready") is not True:
        raise ValueError("近期缓存没有通过formal_ready质量验收，拒绝合并")


def main():
    args = parse_args()
    target_data, target_metadata = load_market_cache(args.target)
    recent_data, recent_metadata = load_market_cache(args.recent)
    validate_compatibility(target_metadata, recent_metadata)

    requested_codes = list(
        ((target_metadata or {}).get("universe") or {}).get("requested_codes") or []
    )
    if not requested_codes:
        raise ValueError("长期缓存缺少requested_codes，拒绝猜测股票池")
    allowed = set(requested_codes) | {cfg.benchmark_index}
    compatible_recent = {code: frame for code, frame in recent_data.items() if code in allowed}
    merged = merge_market_data(
        target_data,
        compatible_recent,
        cfg.benchmark_index,
    )
    merged = {code: frame for code, frame in merged.items() if code in allowed}

    master = load_security_master(args.target)
    if master is None or master.empty:
        raise ValueError("长期缓存缺少证券主数据，拒绝合并")
    warnings = list(
        dict.fromkeys(
            list((target_metadata or {}).get("warnings") or [])
            + ["复用已审计近期缓存，减少重复网络下载"]
        )
    )
    metadata = _write_checkpoint(
        args.target,
        merged,
        adjust=str(target_metadata["adjust"]),
        universe_mode=((target_metadata.get("universe") or {}).get("mode") or "historical-main"),
        source=str(target_metadata["source"]),
        warnings=warnings,
        requested_codes=requested_codes,
        status="partial",
        security_master_info=target_metadata.get("security_master"),
        requested_start=pd.Timestamp(args.start),
        requested_end=pd.Timestamp(args.end),
        security_master=master,
    )
    print(
        f"合并近期缓存股票: {len(compatible_recent) - int(cfg.benchmark_index in compatible_recent)}"
    )
    print(f"长期缓存当前股票: {metadata['universe']['stock_count']}")
    print(f"完整区间已完成: {metadata['download']['completed_codes']}/{len(requested_codes)}")
    print(f"目标缓存: {os.path.abspath(args.target)}")


if __name__ == "__main__":
    main()
