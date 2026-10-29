#!/usr/bin/env python3
"""Run a small, explicit parameter plan and build a comparison report."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from analysis.parameter_comparison import write_parameter_comparison
from strategy.parameters import ParameterError, validate_overrides


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_PLAN = PROJECT_DIR / "experiments" / "risk_balance_example.json"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "parameter_sweeps"


def _slug(value: str) -> str:
    slug = SAFE_NAME.sub("-", value).strip("-._")
    return slug[:50] or "variant"


def load_plan(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterError(f"无法读取实验计划 {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParameterError("实验计划最外层必须是 JSON 对象")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise ParameterError("实验计划至少需要两个 variants")
    if len(variants) > 8:
        raise ParameterError("一次最多运行8个方案，避免误触发耗时寻优")
    base_overrides = payload.get("base_overrides", {})
    if not isinstance(base_overrides, dict):
        raise ParameterError("base_overrides 必须是 JSON 对象")

    names = set()
    normalized_variants = []
    for index, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            raise ParameterError(f"第{index}个方案必须是 JSON 对象")
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ParameterError(f"第{index}个方案缺少 name")
        if name in names:
            raise ParameterError(f"方案名称重复: {name}")
        names.add(name)
        overrides = variant.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ParameterError(f"{name}.overrides 必须是 JSON 对象")
        merged = dict(base_overrides)
        merged.update(overrides)
        normalized_variants.append(
            {
                "name": name,
                "description": str(variant.get("description", "")).strip(),
                "overrides": validate_overrides(merged),
            }
        )
    return {
        "experiment_name": str(payload.get("experiment_name", resolved.stem)).strip()
        or resolved.stem,
        "notes": str(payload.get("notes", "")).strip(),
        "source_plan": str(resolved),
        "variants": normalized_variants,
    }


def _command(
    start: str,
    end: str,
    cache: Path,
    run_root: Path,
    run_id: str,
    overrides: dict[str, Any],
    allow_incomplete_data: bool,
    with_run_charts: bool,
    quiet: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_DIR / "run_backtest.py"),
        start,
        end,
        "--cache",
        str(cache),
        "--output-root",
        str(run_root),
        "--run-id",
        run_id,
    ]
    if not with_run_charts:
        command.append("--no-charts")
    if allow_incomplete_data:
        command.append("--allow-incomplete-data")
    if quiet:
        command.append("--quiet")
    for name, value in overrides.items():
        command.extend(
            ["--set", f"{name}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"]
        )
    return command


def run_experiment(args: argparse.Namespace) -> Path:
    plan = load_plan(args.plan)
    cache = Path(args.cache).expanduser().resolve()
    if not cache.is_file():
        raise ParameterError(f"行情缓存不存在: {cache}")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{_slug(plan['experiment_name'])}_{stamp}"
    output_root = Path(args.output_root).expanduser().resolve()
    experiment_dir = output_root / experiment_id
    if experiment_dir.exists():
        raise ParameterError(f"实验目录已存在，拒绝覆盖: {experiment_dir}")
    run_root = experiment_dir / "runs"

    if args.dry_run:
        print(f"实验名称: {plan['experiment_name']}")
        print(f"区间: {args.start} -> {args.end}")
        print(f"缓存: {cache}")
        for variant in plan["variants"]:
            print(f"- {variant['name']}: {json.dumps(variant['overrides'], ensure_ascii=False)}")
        return experiment_dir

    run_root.mkdir(parents=True)
    recorded_plan = {
        **plan,
        "period": {"start": args.start, "end": args.end},
        "cache": str(cache),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (experiment_dir / "experiment_plan.json").write_text(
        json.dumps(recorded_plan, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    entries = []
    for index, variant in enumerate(plan["variants"], start=1):
        run_id = f"{index:02d}-{_slug(variant['name'])}"
        command = _command(
            args.start,
            args.end,
            cache,
            run_root,
            run_id,
            variant["overrides"],
            args.allow_incomplete_data,
            args.with_run_charts,
            args.quiet,
        )
        print(f"\n[{index}/{len(plan['variants'])}] {variant['name']}")
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"方案 {variant['name']} 回测失败，退出码 {completed.returncode}")
        entries.append(
            {
                "name": variant["name"],
                "description": variant["description"],
                "overrides": variant["overrides"],
                "run_dir": run_root / run_id,
            }
        )

    report = write_parameter_comparison(
        experiment_dir,
        entries,
        plan["experiment_name"],
        args.start,
        args.end,
    )
    print(f"\n参数对比报告: {report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行最多8个明确方案，并生成参数敏感性报告")
    parser.add_argument("start", help="回测开始日期")
    parser.add_argument("end", help="回测结束日期")
    parser.add_argument("--cache", required=True, help="已经审计通过的本地行情缓存")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="JSON实验计划")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT), help="实验输出根目录")
    parser.add_argument(
        "--allow-incomplete-data", action="store_true", help="仅兼容研究：允许不完整缓存"
    )
    parser.add_argument(
        "--with-run-charts", action="store_true", help="同时生成每个方案的PNG图；默认只生成网页"
    )
    parser.add_argument("--quiet", action="store_true", help="减少每个回测的进度输出")
    parser.add_argument("--dry-run", action="store_true", help="只检查和显示计划，不运行回测")
    return parser.parse_args()


def main() -> None:
    try:
        run_experiment(parse_args())
    except (ParameterError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
