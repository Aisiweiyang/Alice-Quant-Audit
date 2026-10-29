#!/usr/bin/env python3
"""Generate a CSV order plan for manual paper-trading execution."""

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data.cache_store import load_market_cache
from paper_trade.order_ledger import OrderLedger
from paper_trade.planner import build_order_plan
from paper_trade.risk import RiskLimits, evaluate_orders
from strategy.config import StrategyConfig as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="生成下一交易日模拟盘订单草案")
    parser.add_argument("--cache", required=True, help="可信的本地行情缓存")
    parser.add_argument("--signal-date", required=True, help="信号日期 YYYY-MM-DD")
    parser.add_argument("--execution-date", required=True, help="计划执行日期 YYYY-MM-DD")
    parser.add_argument("--account", help="账户JSON；不传则使用空账户")
    parser.add_argument(
        "--reconciliation-report",
        help="最近一次账户对账JSON；未通过时禁止生成买入订单",
    )
    parser.add_argument(
        "--emergency-stop",
        action="store_true",
        help="紧急停止新增买入，风险降低卖出仍可保留",
    )
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_DIR, "output", "paper_trade"))
    return parser.parse_args()


def main():
    args = parse_args()
    data, metadata = load_market_cache(args.cache, cfg.benchmark_index)
    if args.account:
        with open(args.account, "r", encoding="utf-8") as handle:
            account = json.load(handle)
    else:
        account = {"cash": cfg.initial_capital, "holdings": {}}

    orders, timing = build_order_plan(data, args.signal_date, args.execution_date, account, cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    ledger_path = os.path.join(args.output_dir, "order_ledger.json")
    ledger = OrderLedger(ledger_path)
    reconciliation = {
        "provided": bool(args.reconciliation_report),
        "passed": True,
        "path": args.reconciliation_report,
    }
    if args.reconciliation_report:
        with open(args.reconciliation_report, "r", encoding="utf-8") as handle:
            reconciliation_report = json.load(handle)
        reconciliation["passed"] = bool(reconciliation_report.get("passed"))
        reconciliation["blocking_issue_count"] = int(
            reconciliation_report.get("blocking_issue_count", 0)
        )

    limits = RiskLimits.from_config(
        cfg,
        emergency_stop=(args.emergency_stop or cfg.emergency_stop),
    )
    risk = evaluate_orders(
        orders,
        account,
        ledger.snapshot(),
        limits,
        reconciliation_passed=reconciliation["passed"],
    )
    orders = list(risk.allowed_orders)
    if any("account_reconciliation_failed" in item["reasons"] for item in risk.rejected_orders):
        timing = {"can_buy": False, "reason": "account_reconciliation_failed"}
    reservation = ledger.reserve(orders)
    stem = f"orders_{args.signal_date.replace('-', '')}"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    pd.DataFrame(orders).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "signal_date": args.signal_date,
                "execution_date": args.execution_date,
                "execution": "next_session_open_manual_review",
                "market_timing": timing,
                "data_metadata": metadata,
                "reconciliation": reconciliation,
                "risk": risk.as_dict(),
                "idempotency": {
                    "ledger": ledger_path,
                    "new_order_ids": list(reservation.new_order_ids),
                    "existing_order_ids": list(reservation.existing_order_ids),
                },
                "orders": orders,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    print(f"订单草案: {csv_path}")
    print(
        f"幂等检查: 新订单 {len(reservation.new_order_ids)} 笔，"
        f"已存在 {len(reservation.existing_order_ids)} 笔"
    )
    print(f"风控检查: 允许 {len(risk.allowed_orders)} 笔，拦截 {len(risk.rejected_orders)} 笔")
    print("注意：这是人工复核文件，不会连接券商或自动下单。")


if __name__ == "__main__":
    main()
