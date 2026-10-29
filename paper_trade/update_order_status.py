#!/usr/bin/env python3
"""Apply one manual, validated order-state transition to the local ledger."""

import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from paper_trade.order_ledger import OrderLedger


def parse_args():
    parser = argparse.ArgumentParser(description="人工确认并更新本地订单状态")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--order-id", required=True)
    parser.add_argument(
        "--status",
        required=True,
        choices=("REVIEWED", "SUBMITTED", "REJECTED", "CANCELLED"),
    )
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    record = OrderLedger(args.ledger).transition(
        args.order_id,
        args.status,
        {"type": "MANUAL_CONFIRMATION", "note": args.note},
    )
    print(f"订单 {args.order_id}: {record['status']}")
    print("只更新本地台账，未向券商提交或撤销订单。")


if __name__ == "__main__":
    main()
