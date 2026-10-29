#!/usr/bin/env python3
"""黄金基线重生成工具（显式重生成 + diff打印，禁止静默覆盖）。

用法（在 alice_framework_v4 目录下）：
  python scripts/update_golden.py             # 重跑full管线并与现有黄金基线打印差异，不写文件
  python scripts/update_golden.py --confirm   # 确认差异可接受后，覆盖 tests/golden/golden_full.json

规则：
- 无既有黄金文件时首次生成可直接写入；
- 已有黄金文件时，不带 --confirm 只打印差异（保护现场）；
- 带 --confirm 且差异非空时覆盖并提示按 "GOLDEN UPDATE: <原因>" 写进提交信息，
  一次提交只允许一个口径变更。
"""

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from tests.golden_builder import (  # noqa: E402
    GOLDEN_PATH,
    build_golden,
    diff_golden,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="黄金基线重生成（显式+diff，禁止静默覆盖）")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="实际覆盖 tests/golden/golden_full.json；缺省只打印差异",
    )
    args = parser.parse_args()

    output_root = os.path.join(PROJECT_DIR, "output", "golden_regen")
    os.makedirs(output_root, exist_ok=True)
    # 该目录是本脚本的专属暂存区：每次重生成前清掉历史报告，避免堆积。
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        import shutil

        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    print("重跑full管线生成黄金基线（demo_cache固定种子，无随机性）...")
    actual = build_golden(output_root)

    if not os.path.isfile(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w", encoding="utf-8") as handle:
            json.dump(actual, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"黄金基线不存在，已首次生成: {GOLDEN_PATH}")
        return 0

    with open(GOLDEN_PATH, "r", encoding="utf-8") as handle:
        golden = json.load(handle)
    differences = diff_golden(golden, actual)
    if not differences:
        print("与现有黄金基线完全一致，无需更新。")
        return 0
    print(f"共{len(differences)}处差异：")
    for line in differences[:80]:
        print(f"  {line}")
    if len(differences) > 80:
        print(f"  ...（其余{len(differences) - 80}处省略）")
    if not args.confirm:
        print(
            "\n未写入（缺省只读）。确认差异为有意的口径变更后，"
            "加 --confirm 重新执行以覆盖黄金文件，"
        )
        print("并在提交信息中声明 GOLDEN UPDATE: <原因>（一次提交只允许一个口径变更）。")
        return 1
    with open(GOLDEN_PATH, "w", encoding="utf-8") as handle:
        json.dump(actual, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\n已覆盖 {GOLDEN_PATH}。请以 GOLDEN UPDATE: <原因> 说明本次口径变更。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
