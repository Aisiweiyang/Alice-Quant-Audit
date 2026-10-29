#!/usr/bin/env python3
"""Recalculate period metrics from saved equity and trade records."""

import argparse
import json
import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="从已保存净值重算子区间指标")
    parser.add_argument("--equity", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", help="可选JSON输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    equity = pd.read_csv(args.equity, parse_dates=["date"])
    equity = equity[equity["date"].between(start, end)].copy()
    if len(equity) < 2:
        raise SystemExit("子区间净值点不足")

    returns = equity["total"].pct_change().dropna()
    total_return = equity["total"].iloc[-1] / equity["total"].iloc[0] - 1
    days = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days, 1)
    peak = equity["total"].cummax()
    drawdown = equity["total"].div(peak).sub(1)
    volatility = returns.std(ddof=1) * np.sqrt(252)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252) if returns.std(ddof=1) > 0 else 0

    trades = pd.read_csv(args.trades, parse_dates=["date"])
    sells = trades[(trades["action"] == "SELL") & trades["date"].between(start, end)].copy()
    wins = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] < 0]

    with open(args.cache, "rb") as handle:
        data = pickle.load(handle)
    benchmark = data["sh.000300"].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark[benchmark["date"].between(start, end)].sort_values("date")
    benchmark_return = benchmark["close"].iloc[-1] / benchmark["close"].iloc[0] - 1

    result = {
        "period_note": "子区间指标由连续全区间净值切片计算，期初持仓延续自此前日期",
        "total_return_pct": round(float(total_return * 100), 2),
        "cagr_pct": round(float(((1 + total_return) ** (365 / days) - 1) * 100), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "annual_volatility_pct": round(float(volatility * 100), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "benchmark_return_pct": round(float(benchmark_return * 100), 2),
        "excess_return_pct": round(float((total_return - benchmark_return) * 100), 2),
        "total_trades": int(len(sells)),
        "win_rate_pct": round(float(len(wins) / len(sells) * 100), 1) if len(sells) else 0.0,
        "profit_factor": round(float(wins["pnl"].sum() / abs(losses["pnl"].sum())), 2)
        if len(losses) and losses["pnl"].sum() != 0
        else None,
    }
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(content)
    print(content)


if __name__ == "__main__":
    main()
