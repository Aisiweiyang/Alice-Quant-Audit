"""Comparison report for a bounded set of Alice parameter experiments."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


PALETTE = ["#2563eb", "#e11d48", "#059669", "#7c3aed", "#ea580c", "#0891b2"]


def _path(
    values: list[float],
    left: float,
    top: float,
    width: float,
    height: float,
    low: float,
    high: float,
) -> str:
    scale = max(high - low, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = left + index / max(len(values) - 1, 1) * width
        y = top + height - (value - low) / scale * height
        points.append(f"{x:.2f},{y:.2f}")
    return "M " + " L ".join(points)


def _load_entry(entry: dict[str, Any], index: int) -> dict[str, Any]:
    run_dir = Path(entry["run_dir"])
    metrics = json.loads((run_dir / "reports" / "metrics.json").read_text(encoding="utf-8"))
    statistics_path = run_dir / "reports" / "run_statistics.json"
    statistics = (
        json.loads(statistics_path.read_text(encoding="utf-8")) if statistics_path.exists() else {}
    )
    equity = pd.read_csv(run_dir / "reports" / "equity.csv")
    equity["date"] = pd.to_datetime(equity["date"])
    equity["normalized"] = equity["total"] / float(equity["total"].iloc[0]) * 100.0
    return {
        **entry,
        "run_dir": run_dir,
        "metrics": metrics,
        "statistics": statistics,
        "equity": equity,
        "color": PALETTE[index % len(PALETTE)],
    }


def equity_comparison_svg(entries: list[dict[str, Any]]) -> str:
    width, height = 980, 390
    left, right, top, bottom = 62, 20, 38, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    all_values = [float(value) for entry in entries for value in entry["equity"]["normalized"]]
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * 0.08, 2.0)
    low -= padding
    high += padding
    grid = []
    for step in range(5):
        value = low + (high - low) * step / 4
        y = top + plot_height - step * plot_height / 4
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
        )
        grid.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>'
        )
    lines, legend = [], []
    for index, entry in enumerate(entries):
        values = entry["equity"]["normalized"].astype(float).tolist()
        lines.append(
            f'<path id="line-{index}" d="{_path(values, left, top, plot_width, plot_height, low, high)}" '
            f'fill="none" stroke="{entry["color"]}" stroke-width="2.6"/>'
        )
        legend.append(
            f'<button type="button" class="legend" data-target="line-{index}" aria-pressed="true">'
            f'<span style="background:{entry["color"]}"></span>{html.escape(entry["name"])}</button>'
        )
    dates = entries[0]["equity"]["date"]
    date_labels = [dates.iloc[0], dates.iloc[len(dates) // 2], dates.iloc[-1]]
    date_text = "".join(
        f'<text x="{left + index * plot_width / 2:.1f}" y="{height - 17}" text-anchor="middle" class="axis">{value:%Y-%m-%d}</text>'
        for index, value in enumerate(date_labels)
    )
    return f"""<div class="legend-row">{"".join(legend)}</div><svg viewBox="0 0 {width} {height}" role="img" aria-label="不同参数方案的净值曲线">
<style>.grid{{stroke:#e2e8f0;stroke-width:1}}.axis{{font:12px "Microsoft YaHei",sans-serif;fill:#64748b}}</style>{"".join(grid)}{"".join(lines)}{date_text}</svg>"""


def risk_return_svg(entries: list[dict[str, Any]]) -> str:
    width, height = 620, 360
    left, right, top, bottom = 64, 24, 30, 52
    returns = [float(entry["metrics"]["total_return_pct"]) for entry in entries]
    drawdowns = [abs(float(entry["metrics"]["max_drawdown_pct"])) for entry in entries]
    x_low, x_high = min(drawdowns), max(drawdowns)
    y_low, y_high = min(returns), max(returns)
    x_padding = max((x_high - x_low) * 0.20, 1.0)
    y_padding = max((y_high - y_low) * 0.20, 2.0)
    x_low, x_high = max(0, x_low - x_padding), x_high + x_padding
    y_low, y_high = y_low - y_padding, y_high + y_padding
    plot_width, plot_height = width - left - right, height - top - bottom
    parts = []
    for step in range(5):
        x = left + step * plot_width / 4
        y = top + plot_height - step * plot_height / 4
        xv = x_low + (x_high - x_low) * step / 4
        yv = y_low + (y_high - y_low) * step / 4
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="grid"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" class="axis">{xv:.1f}%</text>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{yv:.1f}%</text>'
        )
    for entry, drawdown, total_return in zip(entries, drawdowns, returns):
        x = left + (drawdown - x_low) / max(x_high - x_low, 1e-9) * plot_width
        y = top + plot_height - (total_return - y_low) / max(y_high - y_low, 1e-9) * plot_height
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{entry["color"]}"/>')
        parts.append(
            f'<text x="{x + 11:.1f}" y="{y - 9:.1f}" class="label">{html.escape(entry["name"])}</text>'
        )
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="参数方案风险收益图"><style>.grid{{stroke:#e2e8f0;stroke-width:1}}.axis{{font:11px "Microsoft YaHei",sans-serif;fill:#64748b}}.label{{font:12px "Microsoft YaHei",sans-serif;fill:#334155}}</style>{"".join(parts)}<text x="{width / 2}" y="{height - 4}" text-anchor="middle" class="axis">最大回撤绝对值（越左越好）</text><text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" text-anchor="middle" class="axis">总收益（越高越好）</text></svg>'''


def write_parameter_comparison(
    output_dir: str | Path,
    raw_entries: list[dict[str, Any]],
    experiment_name: str,
    start: str,
    end: str,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries = [_load_entry(entry, index) for index, entry in enumerate(raw_entries)]
    best_sharpe = max(float(entry["metrics"]["sharpe_ratio"]) for entry in entries)

    fields = [
        "name",
        "description",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "excess_return_pct",
        "total_trades",
        "win_rate_pct",
        "profit_factor",
        "annualized_turnover",
        "total_friction",
        "run_dir",
    ]
    csv_rows = []
    table_rows = []
    for entry in entries:
        metrics, statistics = entry["metrics"], entry["statistics"]
        row = {
            "name": entry["name"],
            "description": entry.get("description", ""),
            **{key: metrics.get(key) for key in fields if key in metrics},
            "annualized_turnover": statistics.get("annualized_turnover"),
            "total_friction": statistics.get("total_friction"),
            "run_dir": entry["run_dir"].name,
        }
        csv_rows.append(row)
        highlight = ' class="best"' if float(metrics["sharpe_ratio"]) == best_sharpe else ""
        table_rows.append(
            f"<tr{highlight}><td><strong>{html.escape(entry['name'])}</strong><br><small>{html.escape(entry.get('description', ''))}</small></td>"
            f"<td>{metrics['total_return_pct']:+.2f}%</td><td>{metrics['cagr_pct']:+.2f}%</td><td>{metrics['sharpe_ratio']:.2f}</td>"
            f"<td>{metrics['max_drawdown_pct']:.2f}%</td><td>{metrics['excess_return_pct']:+.2f}%</td><td>{metrics['total_trades']}</td>"
            f"<td>{statistics.get('annualized_turnover', 0):.2f}倍</td><td>¥{statistics.get('total_friction', 0):,.0f}</td>"
            f'<td><a href="runs/{html.escape(entry["run_dir"].name)}/index.html">查看单次报告</a></td></tr>'
        )

    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    (root / "comparison.json").write_text(
        json.dumps(csv_rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alice 参数实验对比</title><style>
:root{{--bg:#f4f7fb;--ink:#172033;--muted:#64748b;--card:#fff;--line:#e2e8f0;--blue:#2563eb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei",system-ui,sans-serif}}main{{max-width:1220px;margin:auto;padding:34px 22px 60px}}h1{{margin:0 0 6px}}h2{{font-size:20px;margin:0 0 14px}}.muted,small{{color:var(--muted)}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-top:16px;box-shadow:0 8px 24px #0f172a0a}}.panel svg{{display:block;width:100%;height:auto}}.legend-row{{display:flex;flex-wrap:wrap;gap:12px}}.legend{{border:0;background:transparent;padding:5px;cursor:pointer;color:var(--ink)}}.legend span{{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:6px}}.legend[aria-pressed="false"]{{opacity:.35;text-decoration:line-through}}.table-wrap{{overflow:auto}}table{{width:100%;min-width:1050px;border-collapse:collapse}}th,td{{padding:11px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{color:var(--muted);font-weight:500}}th:first-child,td:first-child{{text-align:left;white-space:normal;min-width:190px}}tr.best{{background:#eff6ff}}a{{color:var(--blue)}}.notice{{border-left:4px solid #f59e0b;background:#fff7ed;padding:13px 15px;border-radius:8px;margin-top:16px}}
</style></head><body><main><h1>{html.escape(experiment_name)}</h1><p class="muted">{html.escape(start)} 至 {html.escape(end)} · 蓝色行只是本区间夏普最高，不会自动成为生产参数</p>
<section class="panel"><h2>净值曲线对比</h2>{equity_comparison_svg(entries)}</section><section class="panel"><h2>风险—收益位置</h2>{risk_return_svg(entries)}</section>
<section class="panel table-wrap"><h2>指标明细</h2><table><thead><tr><th>方案</th><th>总收益</th><th>年化</th><th>夏普</th><th>最大回撤</th><th>超额</th><th>交易</th><th>年化换手</th><th>摩擦成本</th><th>报告</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></section>
<div class="notice"><strong>防止过拟合：</strong>本页只做参数敏感性比较。真正候选参数必须在未参与选择的时间区间再次验证，并且同时满足回撤、换手率和交易成本要求。</div>
<script>document.querySelectorAll('.legend').forEach(button=>button.addEventListener('click',()=>{{const pressed=button.getAttribute('aria-pressed')==='true';button.setAttribute('aria-pressed',String(!pressed));document.getElementById(button.dataset.target).style.display=pressed?'none':'';}}));</script>
</main></body></html>"""
    path = root / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
