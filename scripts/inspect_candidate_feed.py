#!/usr/bin/env python3
"""Validate an external candidate feed without creating orders or files."""

from __future__ import annotations

import argparse
import json
import os
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from integration.candidate_feed import load_candidate_feed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读检查外部候选清单")
    parser.add_argument("feed", help="候选清单JSON路径")
    parser.add_argument("--expected-date", help="预期交易日期 YYYY-MM-DD")
    parser.add_argument("--minimum-universe-coverage", type=float, default=0.80)
    parser.add_argument("--minimum-history-coverage", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        feed = load_candidate_feed(
            args.feed,
            expected_trade_date=args.expected_date,
            minimum_universe_coverage=args.minimum_universe_coverage,
            minimum_history_coverage=args.minimum_history_coverage,
        )
    except (OSError, ValueError) as exc:
        print(f"候选清单拒绝导入: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(feed.summary(), ensure_ascii=False, indent=2, allow_nan=False))
    print("候选清单验证通过；本命令不会生成订单或修改任何账户。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
