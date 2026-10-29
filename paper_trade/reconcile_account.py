#!/usr/bin/env python3
"""Build a blocking local-vs-broker account reconciliation report."""

import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from paper_trade.adapters import ManualCsvAdapter
from paper_trade.order_ledger import OrderLedger
from paper_trade.reconciliation import reconcile_account, write_report


def parse_args():
    parser = argparse.ArgumentParser(description="核对Alice本地状态与模拟盘导出文件")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--local-account", required=True)
    parser.add_argument("--broker-account", required=True)
    parser.add_argument("--broker-positions", required=True)
    parser.add_argument("--broker-orders", required=True)
    parser.add_argument("--broker-fills", required=True)
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_DIR, "output", "paper_trade", "reconciliation.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local = ManualCsvAdapter(args.local_account)
    broker = ManualCsvAdapter(
        args.broker_account,
        positions_path=args.broker_positions,
        orders_path=args.broker_orders,
        fills_path=args.broker_fills,
    )
    report = reconcile_account(
        local.load_account(),
        broker.load_account(),
        OrderLedger(args.ledger).snapshot(),
        broker.load_orders(),
        broker.load_fills(),
    )
    write_report(args.output, report)
    print(f"对账报告: {os.path.abspath(args.output)}")
    print(f"阻断差异: {report['blocking_issue_count']} 项")
    if not report["passed"]:
        print("对账未通过：在差异解决前禁止生成新的买入订单。")
        raise SystemExit(2)
    print("对账通过：本地状态与模拟盘导出快照一致。")


if __name__ == "__main__":
    main()
