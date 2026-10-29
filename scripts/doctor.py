#!/usr/bin/env python3
"""Offline project and market-cache readiness check for Alice v4."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_DIR, "output", ".matplotlib"))

from data.cache_store import load_market_cache, metadata_path
from run_metadata import project_version
from strategy.config import StrategyConfig as cfg


REQUIRED_PACKAGES = ["numpy", "pandas", "matplotlib", "PIL"]


def _item(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def inspect_project(cache_path=None):
    checks = [
        _item("version", "ok", project_version()),
        _item("python", "ok", f"{platform.python_version()} · {sys.executable}"),
    ]
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            checks.append(_item(f"dependency:{package}", "ok", str(version)))
        except ImportError:
            checks.append(_item(f"dependency:{package}", "error", "未安装"))

    output_root = os.path.join(PROJECT_DIR, "output", "runs")
    checks.append(_item("output", "ok", output_root))
    if not cache_path:
        checks.append(
            _item("market_cache", "waiting", "尚未指定缓存；当前可继续开发和测试，不能运行正式回测")
        )
        return checks

    resolved = os.path.abspath(os.path.expanduser(cache_path))
    if not os.path.isfile(resolved):
        checks.append(_item("market_cache", "error", f"文件不存在: {resolved}"))
        return checks

    try:
        data, metadata = load_market_cache(resolved, cfg.benchmark_index)
    except Exception as exc:
        checks.append(_item("market_cache", "error", str(exc)))
        return checks

    stocks = [code for code in data if code != cfg.benchmark_index]
    checks.append(_item("market_cache", "ok", f"{len(stocks)}只股票 · {resolved}"))
    if metadata is None:
        checks.append(_item("metadata", "error", f"缺少 {metadata_path(resolved)}"))
        return checks

    checks.append(_item("metadata", "ok", f"source={metadata.get('source', 'unknown')}"))
    capabilities = metadata.get("capabilities") or {}
    if capabilities.get("historical_trade_status"):
        checks.append(_item("historical_trade_status", "ok", "历史停牌/ST/preclose字段完整"))
    else:
        checks.append(
            _item(
                "historical_trade_status",
                "warning",
                "历史停牌或ST字段不完整，只能用于兼容研究，不能作为正式历史回测结论",
            )
        )
    coverage = float(capabilities.get("valuation_coverage", 0.0))
    if capabilities.get("historical_valuation_fields"):
        checks.append(_item("historical_valuation_fields", "ok", f"PE/PB覆盖率 {coverage:.1%}"))
    else:
        checks.append(
            _item(
                "historical_valuation_fields",
                "warning",
                f"PE/PB覆盖率 {coverage:.1%}；估值过滤可能被跳过",
            )
        )
    quality = metadata.get("quality") or {}
    if not quality:
        checks.append(
            _item(
                "quality_audit",
                "warning",
                "尚未运行离线质量审计；执行 scripts\\audit_market_data.py --cache ...",
            )
        )
    elif quality.get("formal_ready"):
        checks.append(
            _item(
                "quality_audit",
                "ok",
                f"{quality.get('passed_codes', 0)}只股票通过覆盖率和OHLC检查",
            )
        )
    else:
        checks.append(
            _item(
                "quality_audit",
                "warning",
                f"失败{quality.get('failed_codes', '未知')}只，不满足正式回测门槛",
            )
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description="离线检查 Alice v4 运行环境和行情缓存")
    parser.add_argument("--cache", help="可选：要检查的本地行情缓存")
    parser.add_argument("--json", action="store_true", help="输出机器可读JSON")
    args = parser.parse_args()
    checks = inspect_project(args.cache)
    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        labels = {"ok": "通过", "waiting": "等待", "warning": "警告", "error": "失败"}
        print(f"Alice 项目体检 · {project_version()}")
        print("=" * 60)
        for check in checks:
            print(f"[{labels[check['status']]}] {check['name']}: {check['detail']}")
    return 1 if any(item["status"] == "error" for item in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
