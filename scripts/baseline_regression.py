"""双基线回归对照：一键重跑真实数据基线并与锚点逐位比对。

用途：任何口径改动后，验证近端/长窗口两条真实数据基线没有被无意破坏。
锚点文件：tests/baselines/real_baselines.json（含命令参数与确定性指标锚点；
tmc_*/mc_*/Sharpe CI 在同代码+同数据+固定种子下可复现，必须逐位一致）。

用法：
  python scripts/baseline_regression.py            # 全量（近端+长窗口，约25分钟）
  python scripts/baseline_regression.py --quick    # 仅近端（约10分钟）
  python scripts/baseline_regression.py --list     # 只打印锚点摘要，不跑回测

结果：output/baseline_regression/report_<时间戳>.json + 控制台摘要；
任何漂移以退出码1结束（供CI/人工判别），禁止静默通过。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DEFAULT_BASELINE = os.path.join(PROJECT_DIR, "tests", "baselines", "real_baselines.json")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "output", "baseline_regression")


def load_baseline(path: str = DEFAULT_BASELINE) -> dict:
    """读取并校验锚点文件结构（缺字段/类型错误直接抛错，不静默）。"""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("baselines"), list) or not data["baselines"]:
        raise ValueError(f"锚点文件缺少 baselines 数组: {path}")
    if not data.get("data_cache"):
        raise ValueError("锚点文件缺少 data_cache")
    for entry in data["baselines"]:
        for field in ("key", "args", "anchors", "quick"):
            if field not in entry:
                raise ValueError(f"基线[{entry.get('key')}]缺少字段: {field}")
        for key, value in entry["anchors"].items():
            if key != "data_rating" and not isinstance(value, (int, float, bool)):
                raise ValueError(f"锚点 {key} 必须是数字/布尔，当前: {value!r}")
    return data


def compare_metrics(anchors: dict, metrics: dict, data_rating=None) -> list[dict]:
    """逐位比对锚点；返回差异列表（空=通过）。确定性指标禁止容差。"""
    diffs = []
    for key, expected in anchors.items():
        if key == "data_rating":
            actual = data_rating
        else:
            actual = metrics.get(key)
        if actual is None:
            diffs.append({"key": key, "expected": expected, "actual": None, "kind": "missing"})
        elif actual != expected:
            diffs.append({"key": key, "expected": expected, "actual": actual, "kind": "drift"})
    return diffs


def run_baseline(entry: dict, data_cache: str, python: str = sys.executable) -> str:
    """跑单条基线的full回测，返回报告目录（report_id带时间戳防覆盖）。

    报告统一落在默认 output/commercial_reports/（与历史基线同目录，便于追溯）。
    """
    report_id = f"baseline_{entry['key']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        python,
        os.path.join(PROJECT_DIR, "commercial_mvp.py"),
        "full",
        "--cache",
        data_cache,
        "--start",
        entry["args"]["start"],
        "--end",
        entry["args"]["end"],
        "--report-id",
        report_id,
        "--quiet",
    ]
    cmd += list(entry["args"].get("extra") or [])
    subprocess.run(cmd, check=True, cwd=PROJECT_DIR)
    return os.path.join(PROJECT_DIR, "output", "commercial_reports", report_id)


def run_regression(
    baseline_path: str = DEFAULT_BASELINE,
    output_root: str = DEFAULT_OUTPUT,
    quick: bool = False,
    runner=None,
) -> dict:
    """执行回归：跑批→比对→写报告JSON。runner可注入（测试用，免真实回测）。"""
    baseline = load_baseline(baseline_path)
    runner = runner or run_baseline
    results = []
    for entry in baseline["baselines"]:
        if quick and not entry.get("quick"):
            continue
        report_dir = runner(entry, baseline["data_cache"])
        metrics_path = os.path.join(report_dir, "metrics.json")
        quality_path = os.path.join(report_dir, "data_quality.json")
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rating = None
        if os.path.isfile(quality_path):
            with open(quality_path, "r", encoding="utf-8") as handle:
                rating = json.load(handle).get("rating")
        diffs = compare_metrics(entry["anchors"], metrics, rating)
        results.append(
            {
                "key": entry["key"],
                "report_dir": report_dir,
                "passed": not diffs,
                "diffs": diffs,
            }
        )
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "quick": quick,
        "passed": all(r["passed"] for r in results),
        "results": results,
    }
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(
        output_root,
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    summary["report_path"] = out_path
    return summary


def print_summary(summary: dict) -> None:
    for result in summary["results"]:
        status = "通过" if result["passed"] else f"漂移{len(result['diffs'])}项"
        print(f"[{result['key']}] {result['report_dir']}  →  {status}")
        for diff in result["diffs"]:
            print(
                f"    {diff['kind']}: {diff['key']}  锚点={diff['expected']!r} "
                f"实际={diff['actual']!r}"
            )
    print("双基线回归：" + ("全部一致 ✅" if summary["passed"] else "存在漂移 ❌"))
    print(f"报告: {summary.get('report_path')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="双基线回归对照")
    parser.add_argument("--quick", action="store_true", help="仅近端基线")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="只打印锚点摘要")
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    if args.list:
        for entry in baseline["baselines"]:
            print(f"[{entry['key']}] {entry['title']}  锚点{len(entry['anchors'])}项")
        return 0
    summary = run_regression(
        args.baseline,
        args.output_root,
        quick=args.quick,
    )
    print_summary(summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
