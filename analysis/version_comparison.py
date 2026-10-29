"""生成历史回测版本的指标演进图和独立 HTML 报告。"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ANALYSIS_DIR.parent
DEFAULT_HISTORY = ANALYSIS_DIR / "version_history.json"
DEFAULT_COMPARISON = (
    PROJECT_DIR.parent / "quant_v10" / "output" / "factor_research" / "comparison.json"
)
DEFAULT_ALICE_COMPARISON = PROJECT_DIR / "output" / "alice_alpha158_ridge_v2" / "comparison.json"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "version_comparison"
REQUIRED_METRICS = (
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "excess_return_pct",
    "win_rate_pct",
    "profit_factor",
)
COLORS = {
    "V3": "#94a3b8",
    "V6": "#38bdf8",
    "V7": "#f59e0b",
    "V9": "#22c55e",
    "V10": "#8b5cf6",
    "Alice": "#ec4899",
}


def _portable_source(path: Path, section: str) -> str:
    try:
        display = path.resolve().relative_to(PROJECT_DIR.parent.resolve()).as_posix()
    except ValueError:
        display = path.name
    return f"{display}#{section}"


def load_version_history(
    history_path: Path,
    comparison_path: Path | None = None,
    alice_comparison_path: Path | None = None,
) -> list[dict[str, Any]]:
    """读取版本历史，并用可复现的 V9/V10 结果覆盖静态快照。"""
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    versions = payload.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError("version_history.json 中缺少非空 versions 列表")
    result = [dict(item) for item in versions]
    labels = [item.get("version") for item in result]
    if len(labels) != len(set(labels)):
        raise ValueError("版本编号不能重复")
    for item in result:
        missing = [
            metric for metric in REQUIRED_METRICS if not isinstance(item.get(metric), (int, float))
        ]
        if missing:
            raise ValueError(
                f"{item.get('version', '未知版本')} 缺少数值指标: {', '.join(missing)}"
            )

    if comparison_path and comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        sections = {"V9": "legacy_full_period", "V10": "full_period"}
        for item in result:
            section_name = sections.get(item["version"])
            if not section_name:
                continue
            section = comparison.get(section_name)
            if not isinstance(section, dict):
                raise ValueError(f"comparison.json 中缺少 {section_name}")
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    item[key] = value
            item["source"] = _portable_source(comparison_path, section_name)
            item["source_quality"] = "reproduced_full_period"
    if alice_comparison_path and alice_comparison_path.exists():
        comparison = json.loads(alice_comparison_path.read_text(encoding="utf-8"))
        alice = next((item for item in result if item["version"] == "Alice"), None)
        section = comparison.get("full_period")
        if alice is not None and isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    alice[key] = value
            alice["source"] = _portable_source(alice_comparison_path, "full_period")
            experiment = comparison.get("alpha158_ridge_experiment", {})
            if isinstance(experiment.get("out_of_sample"), dict):
                alice["ridge_out_of_sample"] = experiment["out_of_sample"]
    return result


WIDTH, HEIGHT = 1100, 620
LEFT, RIGHT, TOP, BOTTOM = 90, 60, 82, 76
PLOT_W, PLOT_H = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM


def _x_positions(count: int) -> list[float]:
    if count == 1:
        return [LEFT + PLOT_W / 2]
    return [LEFT + index * PLOT_W / (count - 1) for index in range(count)]


def _scale(
    value: float, low: float, high: float, start: float, length: float, invert: bool = False
) -> float:
    ratio = (value - low) / (high - low) if high != low else 0.5
    return start + length * (1 - ratio if invert else ratio)


def _svg_document(title: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}">
<style>text{{font-family:"Microsoft YaHei","SimHei",sans-serif;fill:#334155}}.title{{font-size:25px;font-weight:700;fill:#172033}}.tick{{font-size:14px}}.value{{font-size:14px;font-weight:700}}.legend{{font-size:15px;font-weight:600}}.grid{{stroke:#e2e8f0;stroke-width:1}}.axis{{stroke:#94a3b8;stroke-width:1.2}}</style>
<rect width="100%" height="100%" rx="12" fill="#fff"/><text x="{WIDTH / 2}" y="40" text-anchor="middle" class="title">{html.escape(title)}</text>{body}</svg>'''


def _base_axes(
    labels: list[str],
    y_low: float,
    y_high: float,
    suffix: str = "",
    right_suffix: str | None = None,
) -> str:
    parts = []
    xs = _x_positions(len(labels))
    for step in range(6):
        value = y_low + (y_high - y_low) * step / 5
        y = _scale(value, y_low, y_high, TOP, PLOT_H, invert=True)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{LEFT - 12}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.1f}{suffix}</text>'
        )
        if right_suffix is not None:
            parts.append(
                f'<text x="{WIDTH - RIGHT + 12}" y="{y + 5:.1f}" class="tick">{value:.1f}{right_suffix}</text>'
            )
    parts.append(
        f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" y2="{HEIGHT - BOTTOM}" class="axis"/>'
    )
    for x, label in zip(xs, labels):
        parts.append(
            f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 30}" text-anchor="middle" class="tick">{html.escape(label)}</text>'
        )
    return "".join(parts)


def _polyline(
    xs: list[float],
    values: list[float],
    low: float,
    high: float,
    color: str,
    marker: str = "circle",
    suffix: str = "",
) -> str:
    ys = [_scale(value, low, high, TOP, PLOT_H, invert=True) for value in values]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    parts = [
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>'
    ]
    for x, y, value in zip(xs, ys, values):
        if marker == "square":
            parts.append(
                f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" fill="{color}"/>'
            )
        else:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 13:.1f}" text-anchor="middle" class="value">{value:.2f}{suffix}</text>'
        )
    return "".join(parts)


def plot_return_evolution(versions: list[dict[str, Any]], path: Path) -> None:
    labels = [item["version"] for item in versions]
    xs = _x_positions(len(labels))
    total = [float(item["total_return_pct"]) for item in versions]
    excess = [float(item["excess_return_pct"]) for item in versions]
    low, high = -15.0, 65.0
    body = _base_axes(labels, low, high, "%")
    body += _polyline(xs, total, low, high, "#2563eb", suffix="%")
    body += _polyline(xs, excess, low, high, "#f97316", suffix="%")
    body += '<circle cx="785" cy="60" r="6" fill="#2563eb"/><text x="798" y="65" class="legend">总收益</text><circle cx="900" cy="60" r="6" fill="#f97316"/><text x="913" y="65" class="legend">超额收益</text>'
    path.write_text(_svg_document("版本演进：总收益与超额收益", body), encoding="utf-8")


def plot_risk_quality(versions: list[dict[str, Any]], path: Path) -> None:
    labels = [item["version"] for item in versions]
    xs = _x_positions(len(labels))
    sharpe = [float(item["sharpe_ratio"]) for item in versions]
    drawdown = [abs(float(item["max_drawdown_pct"])) for item in versions]
    body = _base_axes(labels, 0.0, 2.5)
    body += _polyline(xs, sharpe, 0.0, 2.5, "#2563eb")
    ys = [_scale(value, 0.0, 15.0, TOP, PLOT_H, invert=True) for value in drawdown]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    body += f'<polyline points="{points}" fill="none" stroke="#dc2626" stroke-width="4" stroke-linejoin="round"/>'
    for x, y, value in zip(xs, ys, drawdown):
        body += f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2" fill="#dc2626"/><text x="{x:.1f}" y="{y + 24:.1f}" text-anchor="middle" class="value">{value:.2f}%</text>'
    for step in range(6):
        value = 15.0 * step / 5
        y = _scale(value, 0.0, 15.0, TOP, PLOT_H, invert=True)
        body += f'<text x="{WIDTH - RIGHT + 12}" y="{y + 5:.1f}" class="tick">{value:.0f}%</text>'
    body += '<circle cx="690" cy="60" r="6" fill="#2563eb"/><text x="703" y="65" class="legend">Sharpe（左轴）</text><rect x="855" y="54" width="12" height="12" rx="2" fill="#dc2626"/><text x="875" y="65" class="legend">回撤绝对值（右轴）</text>'
    path.write_text(_svg_document("版本演进：风险调整收益与回撤", body), encoding="utf-8")


def plot_trade_quality(versions: list[dict[str, Any]], path: Path) -> None:
    labels = [item["version"] for item in versions]
    xs = _x_positions(len(labels))
    win_rate = [float(item["win_rate_pct"]) for item in versions]
    profit_factor = [float(item["profit_factor"]) for item in versions]
    body = _base_axes(labels, 0.0, 60.0, "%")
    for x, value in zip(xs, win_rate):
        y = _scale(value, 0.0, 60.0, TOP, PLOT_H, invert=True)
        body += f'<rect x="{x - 30:.1f}" y="{y:.1f}" width="60" height="{HEIGHT - BOTTOM - y:.1f}" rx="5" fill="#0ea5e9" opacity=".82"/><text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value:.1f}%</text>'
    body += _polyline(xs, profit_factor, 0.0, 2.0, "#7c3aed")
    for step in range(6):
        value = 2.0 * step / 5
        y = _scale(value, 0.0, 2.0, TOP, PLOT_H, invert=True)
        body += f'<text x="{WIDTH - RIGHT + 12}" y="{y + 5:.1f}" class="tick">{value:.1f}</text>'
    body += '<rect x="750" y="54" width="14" height="14" rx="2" fill="#0ea5e9"/><text x="772" y="66" class="legend">胜率（左轴）</text><circle cx="900" cy="61" r="6" fill="#7c3aed"/><text x="913" y="66" class="legend">盈利因子（右轴）</text>'
    path.write_text(_svg_document("版本演进：交易质量", body), encoding="utf-8")


def plot_risk_return_map(versions: list[dict[str, Any]], path: Path) -> None:
    x_low, x_high, y_low, y_high = 5.0, 15.0, 0.0, 65.0
    body = _base_axes([], y_low, y_high, "%")
    for step in range(6):
        value = x_low + (x_high - x_low) * step / 5
        x = _scale(value, x_low, x_high, LEFT, PLOT_W)
        body += f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM}" class="grid"/><text x="{x:.1f}" y="{HEIGHT - BOTTOM + 30}" text-anchor="middle" class="tick">{value:.0f}%</text>'
    seen = {}
    for item in versions:
        version = item["version"]
        drawdown = abs(float(item["max_drawdown_pct"]))
        total_return = float(item["total_return_pct"])
        x = _scale(drawdown, x_low, x_high, LEFT, PLOT_W)
        y = _scale(total_return, y_low, y_high, TOP, PLOT_H, invert=True)
        key = (round(drawdown, 4), round(total_return, 4))
        overlap = seen.get(key, 0)
        seen[key] = overlap + 1
        x += overlap * 9
        y -= overlap * 13
        color = COLORS.get(version, "#64748b")
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" stroke="white" stroke-width="3"/><text x="{x + 15:.1f}" y="{y - 12:.1f}" class="value">{version}  {total_return:.1f}% / -{drawdown:.1f}%</text>'
    body += f'<text x="{WIDTH / 2}" y="{HEIGHT - 20}" text-anchor="middle" class="legend">最大回撤绝对值（%）← 越低越好</text><text x="23" y="{HEIGHT / 2}" transform="rotate(-90 23 {HEIGHT / 2})" text-anchor="middle" class="legend">总收益（%）↑ 越高越好</text>'
    path.write_text(_svg_document("风险—收益地图", body), encoding="utf-8")


def write_csv(versions: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "version",
        "name",
        "status",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "annual_volatility_pct",
        "sharpe_ratio",
        "benchmark_return_pct",
        "excess_return_pct",
        "total_trades",
        "win_rate_pct",
        "profit_factor",
        "strategy_note",
        "source_quality",
        "source",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(versions)


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "—"


def write_html(versions: list[dict[str, Any]], output_dir: Path, comparison_used: bool) -> None:
    rows = []
    for item in versions:
        quality = (
            "可复现全区间"
            if item.get("source_quality") == "reproduced_full_period"
            else "旧报告汇总"
        )
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(item['version'])}</strong><br><small>{html.escape(item['name'])}</small></td>"
            f"<td>{_fmt(item['total_return_pct'], suffix='%')}</td><td>{_fmt(item['excess_return_pct'], suffix='%')}</td>"
            f"<td>{_fmt(item['max_drawdown_pct'], suffix='%')}</td><td>{_fmt(item['sharpe_ratio'])}</td>"
            f"<td>{_fmt(item['win_rate_pct'], 1, '%')}</td><td>{_fmt(item['profit_factor'])}</td>"
            f"<td>{html.escape(item['strategy_note'])}</td><td><span class='badge'>{quality}</span></td></tr>"
        )
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    data_note = (
        "V9、V10、Alice 已从回测结果动态刷新。"
        if comparison_used
        else "可复现版本使用历史数据文件中的快照。"
    )
    alice = next((item for item in versions if item["version"] == "Alice"), {})
    ridge = alice.get("ridge_out_of_sample", {})
    ridge_note = ""
    if ridge:
        ridge_note = (
            f"Alice 的 Ridge 实验样本外收益 {ridge.get('total_return_pct', 0):.2f}%、"
            f"最大回撤 {ridge.get('max_drawdown_pct', 0):.2f}%、Sharpe {ridge.get('sharpe_ratio', 0):.2f}，未通过验收。"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>回测版本依次对比报告</title><style>
body{{margin:0;background:#f1f5f9;color:#172033;font-family:"Microsoft YaHei",system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:38px 24px 64px}}
h1{{margin:0 0 8px;font-size:32px}}.lead{{color:#526079;margin:0 0 26px}}.notice{{padding:15px 18px;background:#fff7ed;border-left:4px solid #f59e0b;border-radius:8px;margin:20px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:white;border-radius:14px;padding:16px;box-shadow:0 7px 22px #0f172a12}}.card img{{display:block;width:100%;height:auto}}
.table-wrap{{overflow:auto;background:white;border-radius:14px;margin-top:20px;box-shadow:0 7px 22px #0f172a12}}table{{width:100%;border-collapse:collapse;min-width:1040px}}th,td{{padding:13px 12px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}}th{{background:#172033;color:white}}th:first-child,td:first-child,th:nth-child(8),td:nth-child(8),th:last-child,td:last-child{{text-align:left}}small,.foot{{color:#64748b}}.badge{{background:#e2e8f0;padding:4px 7px;border-radius:999px;font-size:12px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>回测版本依次对比报告</h1><p class="lead">V3 → V6 → V7 → V9 → V10 → Alice｜生成时间：{html.escape(generated)}</p>
<div class="notice"><strong>口径说明：</strong>V3、V6、V7 只有旧报告汇总指标，未保存逐日净值，因此本报告比较指标演进，不拼接历史资金曲线。V9、V10、Alice 为相同全区间的可复现结果。{html.escape(data_note)} {html.escape(ridge_note)}</div>
<section class="grid"><div class="card"><img src="01_return_evolution.svg" alt="收益演进图"></div><div class="card"><img src="02_risk_quality_evolution.svg" alt="风险质量演进图"></div><div class="card"><img src="03_trade_quality_evolution.svg" alt="交易质量演进图"></div><div class="card"><img src="04_risk_return_map.svg" alt="风险收益地图"></div></section>
<div class="table-wrap"><table><thead><tr><th>版本</th><th>总收益</th><th>超额收益</th><th>最大回撤</th><th>Sharpe</th><th>胜率</th><th>盈利因子</th><th>策略变化</th><th>数据质量</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="foot">结论：V6 的汇总收益最高，但无法完整复现。可复现版本中 V9 的 Sharpe 更高、最大回撤更低；Alice 新增的 Alpha158 与 Ridge 没有通过验收，V9 继续作为生产基线。数据明细见 version_metrics.csv。本报告仅供研究学习，不构成投资建议。</p>
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def generate(
    history_path: Path,
    comparison_path: Path | None,
    output_dir: Path,
    alice_comparison_path: Path | None = DEFAULT_ALICE_COMPARISON,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = load_version_history(history_path, comparison_path, alice_comparison_path)
    plot_return_evolution(versions, output_dir / "01_return_evolution.svg")
    plot_risk_quality(versions, output_dir / "02_risk_quality_evolution.svg")
    plot_trade_quality(versions, output_dir / "03_trade_quality_evolution.svg")
    plot_risk_return_map(versions, output_dir / "04_risk_return_map.svg")
    write_csv(versions, output_dir / "version_metrics.csv")
    write_html(versions, output_dir, bool(comparison_path and comparison_path.exists()))
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 V3 至 Alice 的回测版本对比图")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_COMPARISON,
        help="V10 comparison.json；不存在时使用历史快照",
    )
    parser.add_argument("--alice-comparison", type=Path, default=DEFAULT_ALICE_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    versions = generate(
        args.history.resolve(),
        args.comparison.resolve(),
        args.output_dir.resolve(),
        args.alice_comparison.resolve(),
    )
    print("版本顺序:", " -> ".join(item["version"] for item in versions))
    print("报告:", args.output_dir.resolve() / "index.html")


if __name__ == "__main__":
    main()
