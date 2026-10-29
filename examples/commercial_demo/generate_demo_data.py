#!/usr/bin/env python3
"""生成小型合成行情演示数据（SYNTHETIC，非真实行情）。

用途：离线演示 Alice Quant Audit 的完整分析流程。
数据特征为固定随机种子的几何随机游走，不代表任何真实股票或真实收益。
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from data.cache_store import build_metadata, save_market_cache

SYNTHETIC_TAG = "SYNTHETIC-DEMO"


def build_stock_frame(dates, rng, base_price):
    steps = rng.normal(loc=0.0004, scale=0.015, size=len(dates))
    closes = base_price * np.exp(np.cumsum(steps))
    opens = closes * (1 + rng.normal(0, 0.004, size=len(dates)))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.006, size=len(dates))))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.006, size=len(dates))))
    volume = rng.lognormal(mean=13.5, sigma=0.4, size=len(dates)).round(0)
    preclose = np.concatenate([[closes[0]], closes[:-1]])
    pct_chg = (closes / preclose - 1) * 100
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": opens.round(3),
            "high": highs.round(3),
            "low": lows.round(3),
            "close": closes.round(3),
            "volume": volume,
            "amount": (volume * closes).round(0),
            "preclose": preclose.round(3),
            "pctChg": pct_chg.round(4),
            "tradestatus": 1,
            "isST": 0,
            "pe": rng.uniform(8, 45),
            "pb": rng.uniform(0.8, 8),
        }
    )
    return frame


def generate(target_dir, stocks=40, days=260, seed=20260907):
    os.makedirs(target_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    data = {}
    for index in range(stocks):
        if index % 2 == 0:
            code = f"sh.60{index:04d}"
        else:
            code = f"sz.00{index:04d}"
        data[code] = build_stock_frame(dates, rng, base_price=float(rng.uniform(6, 40)))

    # 基准：全部股票等权收益指数（演示数据内的"沪深300"替身，同样为合成数据）。
    returns = {}
    for code, frame in data.items():
        series = frame.set_index("date")["close"].pct_change()
        for date, value in series.dropna().items():
            returns[date] = returns.get(date, 0.0) + value / stocks
    level = 1000.0
    bench_rows = []
    for date in dates:
        if date in returns:
            level *= 1 + returns[date]
        bench_rows.append({"date": date, "close": round(level, 3)})
    data["sh.000300"] = pd.DataFrame(bench_rows)

    metadata = build_metadata(
        data,
        benchmark="sh.000300",
        adjust="2",
        universe_mode="file",
        source=f"{SYNTHETIC_TAG}:random_walk",
        warnings=[
            "SYNTHETIC DEMO DATA / 合成演示数据：固定种子随机游走，非真实行情，不代表任何真实收益。",
            f"seed={seed}, stocks={stocks}, trading_days={days}",
        ],
    )
    cache_path = os.path.join(target_dir, "demo_cache.pkl")
    save_market_cache(cache_path, data, metadata)

    candidates = {
        "schema_version": 1,
        "source_system": SYNTHETIC_TAG,
        "strategy_version": "synthetic-demo-1",
        "market": "CN_A_SHARE",
        "trade_date": dates[-1].strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.now().isoformat(),
        "quality": {"passed": True, "universe_coverage": 1.0, "history_coverage": 1.0},
        "candidates": [
            {
                "rank": rank + 1,
                "code": code,
                "name": f"合成演示股票{rank + 1}",
                "score": 80.0 - rank,
                "risk": "演示",
                "components": {},
                "reasons": ["SYNTHETIC-DEMO"],
            }
            for rank, code in enumerate(sorted([c for c in data if c != "sh.000300"][:10]))
        ],
    }
    candidates_path = os.path.join(target_dir, "demo_candidates.json")
    with open(candidates_path, "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, ensure_ascii=False, indent=2)

    print(f"合成演示缓存: {cache_path}")
    print(f"合成演示候选: {candidates_path}")
    print(f"股票数: {stocks}  交易日: {days}  数据起始: {dates[0].date()}")
    return cache_path


def main():
    parser = argparse.ArgumentParser(description="生成SYNTHETIC合成演示行情")
    parser.add_argument("--target-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--stocks", type=int, default=40)
    parser.add_argument("--days", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    generate(args.target_dir, args.stocks, args.days, args.seed)


if __name__ == "__main__":
    main()
