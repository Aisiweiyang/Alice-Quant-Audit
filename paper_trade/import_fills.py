#!/usr/bin/env python3
"""Import an offline CSV or JSON fill report into the local order ledger."""

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from paper_trade.order_ledger import OrderLedger


def parse_args():
    parser = argparse.ArgumentParser(description="离线导入并核对模拟盘成交回报")
    parser.add_argument("--ledger", required=True, help="本地order_ledger.json")
    parser.add_argument("--fills", required=True, help="成交CSV或JSON文件")
    return parser.parse_args()


def load_fills(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data["fills"] if isinstance(data, dict) else data
    raise ValueError("成交文件必须是CSV或JSON")


def main():
    args = parse_args()
    fills = load_fills(args.fills)
    if not isinstance(fills, list):
        raise ValueError("成交文件内容必须是记录列表")
    result = OrderLedger(args.ledger).import_fills(fills)
    print(f"新增成交: {len(result.new_fill_ids)} 笔")
    print(f"重复成交: {len(result.existing_fill_ids)} 笔")
    print(f"更新订单: {len(result.updated_order_ids)} 笔")
    print("只更新本地台账，未连接券商或交易账户。")


if __name__ == "__main__":
    main()
