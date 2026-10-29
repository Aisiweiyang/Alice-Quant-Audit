#!/usr/bin/env python3
"""Run a reproducible backtest from live data or a trusted local cache."""

import argparse
import json
import os
import pickle
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data.fetcher import get_index_data
from engine.backtest import run
from run_metadata import (
    build_manifest,
    build_run_id,
    config_snapshot,
    prepare_run_directory,
    project_version,
    write_manifest,
)
from strategy.config import StrategyConfig as cfg
from strategy.parameters import (
    ParameterError,
    apply_overrides,
    combine_overrides,
    load_override_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="运行 Alice 股票选择策略回测")
    parser.add_argument("start", nargs="?", default="2025-01-01", help="开始日期")
    parser.add_argument("end", nargs="?", default="2026-08-06", help="结束日期")
    parser.add_argument(
        "--cache",
        help="受信任的本地pickle行情缓存；设置后不访问网络",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        default=os.path.join(PROJECT_DIR, "output", "runs"),
        help="运行根目录；每次回测会在其中创建独立子目录",
    )
    parser.add_argument("--run-id", help="可选的唯一运行名称；默认按版本和时间生成")
    parser.add_argument(
        "--config",
        dest="config_file",
        help="JSON参数文件；只覆盖列出的参数，不修改 strategy/config.py",
    )
    parser.add_argument(
        "--set",
        dest="assignments",
        action="append",
        default=[],
        metavar="参数=值",
        help="临时覆盖一个参数，可重复使用，例如 --set max_positions=12",
    )
    parser.add_argument("--version", action="version", version=project_version())
    parser.add_argument("--no-charts", action="store_true", help="跳过图表生成")
    parser.add_argument("--no-html-report", action="store_true", help="跳过单次回测网页报告")
    parser.add_argument("--quiet", action="store_true", help="减少进度输出")
    parser.add_argument(
        "--allow-incomplete-data",
        action="store_true",
        help="允许缺少历史ST/停牌或PE/PB字段的缓存；仅用于兼容研究",
    )
    return parser.parse_args()


def _benchmark(args, config=cfg):
    if not args.cache:
        return get_index_data(config.benchmark_index, args.start, args.end)
    with open(os.path.abspath(os.path.expanduser(args.cache)), "rb") as handle:
        data = pickle.load(handle)
    benchmark = data.get(config.benchmark_index)
    if benchmark is None:
        return pd.DataFrame()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    return benchmark[benchmark["date"].between(start, end)][["date", "close"]].copy()


def main():
    args = parse_args()
    try:
        file_overrides = load_override_file(args.config_file) if args.config_file else {}
        requested_overrides = combine_overrides(file_overrides, args.assignments)
        effective_overrides = apply_overrides(requested_overrides, cfg)
    except ParameterError as exc:
        raise SystemExit(f"参数错误: {exc}") from exc

    run_id = args.run_id or build_run_id()
    try:
        run_dir = prepare_run_directory(args.output_root, run_id)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc

    portfolio, metrics = run(
        args.start,
        args.end,
        verbose=not args.quiet,
        cache_path=args.cache,
        allow_incomplete_data=args.allow_incomplete_data,
    )
    if portfolio is None or metrics is None:
        raise SystemExit("回测无数据")

    report_dir = os.path.join(run_dir, "reports")
    chart_dir = os.path.join(run_dir, "charts")
    os.makedirs(report_dir, exist_ok=True)

    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)

    equity = pd.DataFrame(portfolio.equity_curve)
    trades = pd.DataFrame(portfolio.trades)
    decisions = list(getattr(portfolio, "rebalance_decisions", []))
    equity.to_csv(os.path.join(report_dir, "equity.csv"), index=False, encoding="utf-8-sig")
    trades.to_csv(os.path.join(report_dir, "trades.csv"), index=False, encoding="utf-8-sig")
    with open(
        os.path.join(report_dir, "rebalance_decisions.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(decisions, handle, ensure_ascii=False, indent=2, allow_nan=False)
    decision_rows = []
    for decision in decisions:
        selected = decision.get("selected", [])
        decision_rows.append(
            {
                **{key: value for key, value in decision.items() if key != "selected"},
                "selected_codes": ",".join(str(item.get("code", "")) for item in selected),
                "selected_scores": ",".join(str(item.get("score", "")) for item in selected),
            }
        )
    pd.DataFrame(decision_rows).to_csv(
        os.path.join(report_dir, "rebalance_decisions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(report_dir, "effective_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_snapshot(cfg), handle, ensure_ascii=False, indent=2, allow_nan=False)
    with open(
        os.path.join(report_dir, "parameter_overrides.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(effective_overrides, handle, ensure_ascii=False, indent=2, allow_nan=False)

    benchmark = _benchmark(args, cfg)

    if not args.no_charts:
        from analysis.visualize import (
            plot_equity_curve,
            plot_monthly_returns,
            plot_trade_distribution,
        )

        os.makedirs(chart_dir, exist_ok=True)
        plot_equity_curve(equity, benchmark, os.path.join(chart_dir, "equity_curve.png"))
        plot_monthly_returns(equity, os.path.join(chart_dir, "monthly_returns.png"))
        if not trades.empty:
            plot_trade_distribution(trades, os.path.join(chart_dir, "trade_dist.png"))

    html_report = None
    if not args.no_html_report:
        from analysis.backtest_report import write_backtest_report

        html_report = write_backtest_report(
            run_dir,
            metrics,
            equity,
            trades,
            benchmark,
            cfg,
            args.start,
            args.end,
            decisions=decisions,
        )

    manifest = build_manifest(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        config=cfg,
    )
    manifest_path = write_manifest(run_dir, manifest)

    print(f"\n运行编号: {run_id}")
    print(f"结果目录: {run_dir}")
    print(f"指标文件: {metrics_path}")
    if html_report:
        print(f"可视化报告: {html_report}")
    print(f"运行清单: {manifest_path}")


if __name__ == "__main__":
    main()
