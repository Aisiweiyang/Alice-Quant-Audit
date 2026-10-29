#!/usr/bin/env python3
"""规模性能基准：生成N只股票合成数据，实测审计与完整分析耗时。

用法：python scripts/benchmark_scale.py --stocks 1000 --days 250
输出：各阶段耗时JSON（stdout）。仅使用SYNTHETIC合成数据。
"""

import argparse
import json
import os
import sys
import time
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_DIR, "output", ".matplotlib"))

from data.cache_store import build_metadata, save_market_cache


def generate(stocks, days, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    data = {}
    for index in range(stocks):
        code = f"sh.6{index:05d}" if index % 2 == 0 else f"sz.0{index:05d}"
        steps = rng.normal(0.0003, 0.015, size=days)
        closes = 15 * np.exp(np.cumsum(steps))
        preclose = np.concatenate([[closes[0]], closes[:-1]])
        volume = rng.lognormal(13, 0.4, size=days).round(0)
        data[code] = pd.DataFrame(
            {
                "date": dates,
                "open": (closes * 1.001).round(3),
                "high": (closes * 1.012).round(3),
                "low": (closes * 0.99).round(3),
                "close": closes.round(3),
                "volume": volume,
                "amount": (volume * closes).round(0),
                "preclose": preclose.round(3),
                "pctChg": ((closes / preclose - 1) * 100).round(4),
                "tradestatus": 1,
                "isST": 0,
                "pe": 20.0,
                "pb": 2.0,
            }
        )
    data["sh.000300"] = pd.DataFrame(
        {
            "date": dates,
            "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=days)))).round(2),
            "open": 1000.0,
        }
    )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=1000)
    parser.add_argument("--days", type=int, default=250)
    parser.add_argument("--start", default="2025-01-15")
    parser.add_argument("--end", default="2025-11-28")
    parser.add_argument("--keep-cache", default=None, help="保留生成的缓存到指定路径")
    args = parser.parse_args()

    result = {"stocks": args.stocks, "days": args.days, "stages_seconds": {}}
    timings = result["stages_seconds"]

    stage = perf_counter()
    data = generate(args.stocks, args.days)
    timings["generate"] = round(perf_counter() - stage, 1)

    stage = perf_counter()
    metadata = build_metadata(
        data,
        benchmark="sh.000300",
        adjust="2",
        universe_mode="file",
        source="benchmark_synthetic",
    )
    cache_path = args.keep_cache or os.path.join(
        os.path.join(PROJECT_DIR, "output", "benchmark"),
        f"bench_{args.stocks}x{args.days}.pkl",
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_market_cache(cache_path, data, metadata)
    timings["save_cache"] = round(perf_counter() - stage, 1)
    result["cache_mb"] = round(os.path.getsize(cache_path) / 1048576, 1)

    from commercial.config import CommercialRequest
    from commercial.pipeline import run_data_audit_only, run_full_analysis

    output_root = os.path.join(PROJECT_DIR, "output", "benchmark_reports")
    # run_id带时间戳：历史基准目录不覆盖、可重跑、可并列对比。
    stamp = time.strftime("%m%d_%H%M%S")
    request = CommercialRequest(
        cache=cache_path,
        start=args.start,
        end=args.end,
        output_root=output_root,
        quiet=True,
        run_stress=False,
        run_factor_diagnostics=True,
        run_rolling_oos=False,
        max_scenarios=8,
    )
    request.validate()

    stage = perf_counter()
    run_data_audit_only(request, report_id=f"bench_audit_{args.stocks}_{stamp}")
    timings["audit_only"] = round(perf_counter() - stage, 1)

    stage = perf_counter()
    analysis = run_full_analysis(request, report_id=f"bench_full_{args.stocks}_{stamp}")
    timings["full_no_stress"] = round(perf_counter() - stage, 1)
    result["full_timings"] = analysis.get("timings")
    result["rating"] = analysis["data_audit"]["rating"]
    result["note"] = "SYNTHETIC合成数据基准；耗时含图表与manifest哈希。"

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
