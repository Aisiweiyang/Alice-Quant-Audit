"""多窗口compare扫描：同一计划文件在多个窗口串行回测并汇总成一张表。

用法示例（在项目根目录）：
    python scripts/multi_window_scan.py \
        --cache local_data/alice_historical_main_2015_2026.pkl \
        --plan output/realdata_runs/plan_csx_diversified.json \
        --strategy cross_sectional_alpha158 \
        --windows 2022-07-01:2023-06-30,2023-07-01:2024-06-30 \
        --report-prefix real_compare_csx_div

每个窗口调一次 commercial_mvp.py compare（串行，报告编号
<report-prefix>_w<起YYYYMM>_<止YYYYMM>），全部跑完后从各报告目录读取
metrics.json / scenario_comparison.csv / data_quality.json，汇总输出
markdown表（窗口×方案的收益/回撤/Sharpe/DSR/交易数 + 每窗口PBO与分散度），
打印到终端并写入 output/realdata_runs/multi_window_scan_<时间戳>.md。

PBO是窗口内统计量，同一方案组在不同窗口可能大幅波动（2026-09实测同一
计划在三个窗口得到0.25/0.01/0.57），本表就是用来观察这种波动的。
仅供研究，不构成投资建议。
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_DIR / "output" / "commercial_reports"

METRIC_COLUMNS = [
    ("total_return_pct", "收益%"),
    ("max_drawdown_pct", "回撤%"),
    ("sharpe_ratio", "Sharpe"),
    ("deflated_sharpe", "DSR"),
    ("total_trades", "交易"),
]


def parse_windows(spec: str) -> list:
    """把"起:止,起:止"解析成[(start,end)]；起止倒置或格式错误直接报错。"""
    windows = []
    for part in spec.split(","):
        fields = part.strip().split(":")
        if len(fields) != 2:
            raise ValueError(f"窗口格式应为 起:止，收到：{part!r}")
        start, end = fields[0].strip(), fields[1].strip()
        try:
            s = datetime.strptime(start, "%Y-%m-%d")
            e = datetime.strptime(end, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"日期无法解析：{part!r}（{exc}）") from exc
        if s >= e:
            raise ValueError(f"窗口起点不早于终点：{part!r}")
        windows.append((start, end))
    if not windows:
        raise ValueError("至少需要一个窗口")
    return windows


def window_label(start: str, end: str) -> str:
    return f"w{start.replace('-', '')[:6]}_{end.replace('-', '')[:6]}"


def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def aggregate_report(report_dir: Path) -> dict:
    """读取单个compare报告目录，提取汇总所需的最小字段集。"""
    metrics = _load_json(report_dir / "metrics.json") or {}
    quality = _load_json(report_dir / "data_quality.json") or {}
    audit = metrics.get("overfitting_audit") or {}
    pbo = audit.get("pbo") or {}
    dispersion = audit.get("dispersion") or {}
    plans = {}
    csv_path = report_dir / "scenario_comparison.csv"
    if csv_path.is_file():
        with open(csv_path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                plans[row.get("label")] = {key: row.get(key) for key, _ in METRIC_COLUMNS}
    return {
        "report_id": report_dir.name,
        "rating": quality.get("rating"),
        "pbo": pbo.get("pbo_probability"),
        "n_trials": audit.get("n_trials"),
        "validity": dispersion.get("validity"),
        "max_correlation": dispersion.get("max_pair_correlation"),
        "plans": plans,
    }


def render_markdown(snapshots: list, plan_order: list) -> str:
    """把各窗口快照渲染成markdown：窗口×方案主表 + PBO/分散度行。"""
    header = "| 方案 | " + " | ".join(f"{s['report_id']}" for s in snapshots) + " |"
    align = "|---|" + "|".join("---:" for _ in snapshots) + "|"
    lines = [header, align]
    for label in plan_order:
        cells = []
        for snap in snapshots:
            row = snap["plans"].get(label)
            if not row:
                cells.append("—")
                continue
            text = " / ".join(
                "—" if row.get(key) in (None, "") else str(row.get(key))
                for key, _ in METRIC_COLUMNS
            )
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, title in [
        ("pbo", "PBO"),
        ("n_trials", "n_trials"),
        ("validity", "分散度validity"),
        ("max_correlation", "最高两两相关"),
        ("rating", "数据评级"),
    ]:
        lines.append(
            f"| {title} | "
            + " | ".join("—" if s.get(key) is None else str(s.get(key)) for s in snapshots)
            + " |"
        )
    lines.append("")
    lines.append(
        "指标列顺序：收益% / 回撤% / Sharpe / DSR / 交易数。"
        "PBO只反映本窗口内的排名延续性，随窗口可能大幅波动，"
        "单窗口数字不可外推；分散度validity不是ok时PBO不具备参考"
        "意义。仅供研究，不构成投资建议。"
    )
    return "\n".join(lines)


def should_run(report_dir: Path) -> bool:
    """报告目录已存在时默认复用（compare防覆盖，同编号重跑会被拒绝）。"""
    return not report_dir.is_dir()


def run_compare(start: str, end: str, args) -> str:
    report_id = f"{args.report_prefix}_{window_label(start, end)}"
    command = [
        sys.executable,
        str(PROJECT_DIR / "commercial_mvp.py"),
        "compare",
        "--cache",
        args.cache,
        "--start",
        start,
        "--end",
        end,
        "--strategy",
        args.strategy,
        "--plan",
        args.plan,
        "--report-id",
        report_id,
    ]
    print(f"[multi_window_scan] {start} ~ {end} -> {report_id}", flush=True)
    subprocess.run(command, cwd=str(PROJECT_DIR), check=True)
    return report_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--windows", required=True, help="起:止,起:止（YYYY-MM-DD），串行逐窗执行")
    parser.add_argument("--strategy", default="classic_trend")
    parser.add_argument(
        "--report-prefix", required=True, help="报告编号前缀，最终编号为 <前缀>_w起止"
    )
    args = parser.parse_args()

    windows = parse_windows(args.windows)
    for start, end in windows:
        report_id = f"{args.report_prefix}_{window_label(start, end)}"
        if not should_run(REPORTS_ROOT / report_id):
            print(
                f"[multi_window_scan] {report_id} 已存在，复用现有报告"
                f"（要重跑请先把该报告目录改名）",
                flush=True,
            )
            continue
        run_compare(start, end, args)

    snapshots = []
    for start, end in windows:
        report_id = f"{args.report_prefix}_{window_label(start, end)}"
        report_dir = REPORTS_ROOT / report_id
        if not report_dir.is_dir():
            print(f"[multi_window_scan] 缺少报告目录，跳过汇总：{report_dir}", flush=True)
            continue
        snapshots.append(aggregate_report(report_dir))

    plan_order = []
    for snap in snapshots:
        for label in snap["plans"]:
            if label not in plan_order:
                plan_order.append(label)

    markdown = render_markdown(snapshots, plan_order)
    print(markdown, flush=True)
    out_dir = PROJECT_DIR / "output" / "realdata_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"multi_window_scan_{stamp}.md"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")
    print(f"[multi_window_scan] 汇总已写入 {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
