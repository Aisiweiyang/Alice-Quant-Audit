"""Create a self-contained, readable HTML report for one Alice backtest run."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_metadata import config_snapshot
from strategy.config import StrategyConfig


COLORS = {
    "strategy": "#2563eb",
    "benchmark": "#64748b",
    "drawdown": "#dc2626",
    "positive": "#15803d",
    "negative": "#b91c1c",
}


def derive_run_statistics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    avg_equity = float(pd.to_numeric(eq["total"], errors="coerce").mean())
    days = max(int((eq["date"].iloc[-1] - eq["date"].iloc[0]).days), 1)
    years = days / 365.25

    traded_value = 0.0
    total_friction = 0.0
    if trades is not None and not trades.empty:

        def numeric_column(name: str) -> pd.Series:
            if name not in trades.columns:
                return pd.Series(0.0, index=trades.index, dtype=float)
            return pd.to_numeric(trades[name], errors="coerce").fillna(0.0)

        buys = numeric_column("cost")
        sells = numeric_column("revenue")
        friction = numeric_column("total_friction")
        traded_value = float(buys.sum() + sells.sum())
        total_friction = float(friction.sum())

    exposure = pd.to_numeric(eq.get("holdings_value"), errors="coerce") / pd.to_numeric(
        eq["total"], errors="coerce"
    ).replace(0, np.nan)
    return {
        "equity_rows": int(len(eq)),
        "average_positions": round(
            float(pd.to_numeric(eq["positions"], errors="coerce").mean()), 2
        ),
        "maximum_positions": int(pd.to_numeric(eq["positions"], errors="coerce").max()),
        "average_exposure_pct": round(float(exposure.fillna(0).mean() * 100), 2),
        "traded_value": round(traded_value, 2),
        "total_friction": round(total_friction, 2),
        "friction_pct_of_initial": round(total_friction / float(eq["total"].iloc[0]) * 100, 3),
        "annualized_turnover": round(traded_value / max(avg_equity, 1.0) / max(years, 1 / 252), 2),
    }


def derive_decision_statistics(decisions: list[dict[str, Any]] | None) -> dict[str, Any]:
    rows = list(decisions or [])
    selected_scores = [
        float(item["score"])
        for row in rows
        for item in row.get("selected", [])
        if item.get("score") is not None
    ]
    allowed = [row for row in rows if row.get("market_allowed")]
    candidate_counts = [int(row.get("candidate_count", 0)) for row in allowed]
    return {
        "rebalance_count": len(rows),
        "market_allowed_count": len(allowed),
        "market_blocked_count": len(rows) - len(allowed),
        "rebalance_with_buys": sum(bool(row.get("selected")) for row in rows),
        "average_candidates_when_allowed": round(float(np.mean(candidate_counts)), 2)
        if candidate_counts
        else 0.0,
        "selected_count": len(selected_scores),
        "selected_score_min": round(min(selected_scores), 4) if selected_scores else None,
        "selected_score_median": round(float(np.median(selected_scores)), 4)
        if selected_scores
        else None,
        "selected_score_max": round(max(selected_scores), 4) if selected_scores else None,
    }


def decision_score_svg(decisions: list[dict[str, Any]] | None) -> str:
    scores = [
        float(item["score"])
        for row in decisions or []
        for item in row.get("selected", [])
        if item.get("score") is not None
    ]
    if not scores:
        return '<p class="muted">本区间没有实际买入评分。</p>'
    counts, edges = np.histogram(scores, bins=min(8, max(3, int(np.sqrt(len(scores))))))
    width, height = 760, 250
    left, right, top, bottom = 48, 18, 20, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_width = plot_width / max(len(counts), 1)
    peak = max(int(counts.max()), 1)
    bars = []
    for index, count in enumerate(counts):
        bar_height = float(count) / peak * plot_height
        x = left + index * bar_width + 3
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 6, 1):.1f}" '
            f'height="{bar_height:.1f}" rx="4" fill="{COLORS["strategy"]}" opacity=".82"/>'
            f'<text x="{x + (bar_width - 6) / 2:.1f}" y="{max(y - 5, 12):.1f}" '
            f'text-anchor="middle" class="axis">{int(count)}</text>'
        )
    labels = "".join(
        f'<text x="{left + index * bar_width:.1f}" y="{height - 15}" '
        f'text-anchor="middle" class="axis">{edge:.2f}</text>'
        for index, edge in enumerate(edges)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="实际买入评分分布">'
        '<style>.axis{font:12px "Microsoft YaHei",sans-serif;fill:#64748b}</style>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#cbd5e1"/>'
        f"{''.join(bars)}{labels}</svg>"
    )


def _normalize(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    result = frame[["date", value_column]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if result.empty or float(result[value_column].iloc[0]) == 0:
        return pd.DataFrame(columns=["date", "value"])
    result["value"] = result[value_column] / float(result[value_column].iloc[0]) * 100.0
    return result[["date", "value"]]


def _series_path(
    values: list[float], x0: float, y0: float, width: float, height: float, low: float, high: float
) -> str:
    if not values:
        return ""
    scale = max(high - low, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = x0 + (index / max(len(values) - 1, 1)) * width
        y = y0 + height - ((value - low) / scale) * height
        points.append(f"{x:.2f},{y:.2f}")
    return "M " + " L ".join(points)


def equity_svg(equity: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> str:
    eq = _normalize(equity, "total")
    bm = (
        _normalize(benchmark, "close")
        if benchmark is not None and not benchmark.empty
        else pd.DataFrame()
    )
    if not bm.empty:
        bm = bm[bm["date"].isin(eq["date"])]
        if not bm.empty:
            bm["value"] = bm["value"] / float(bm["value"].iloc[0]) * 100.0

    strategy_values = eq["value"].tolist()
    benchmark_values = bm["value"].tolist() if not bm.empty else []
    all_values = strategy_values + benchmark_values
    low = min(all_values) if all_values else 95.0
    high = max(all_values) if all_values else 105.0
    padding = max((high - low) * 0.08, 2.0)
    low -= padding
    high += padding

    width, height = 960, 410
    left, right, top = 62, 20, 28
    plot_width, plot_height = width - left - right, 245
    dd_top, dd_height = 315, 70
    strategy_path = _series_path(strategy_values, left, top, plot_width, plot_height, low, high)
    benchmark_path = _series_path(benchmark_values, left, top, plot_width, plot_height, low, high)

    peak = eq["value"].cummax()
    drawdown = ((eq["value"] - peak) / peak * 100.0).tolist()
    dd_low = min(drawdown) if drawdown else -1.0
    drawdown_path = _series_path(drawdown, left, dd_top, plot_width, dd_height, dd_low, 0.0)
    date_labels = []
    if not eq.empty:
        positions = [0, len(eq) // 2, len(eq) - 1]
        date_labels = [eq.iloc[index]["date"].strftime("%Y-%m-%d") for index in positions]

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
    dates = "".join(
        f'<text x="{left + index * plot_width / 2:.1f}" y="404" text-anchor="middle" class="axis">{html.escape(label)}</text>'
        for index, label in enumerate(date_labels)
    )
    bm_path = (
        f'<path d="{benchmark_path}" fill="none" stroke="{COLORS["benchmark"]}" stroke-width="2" opacity=".8"/>'
        if benchmark_path
        else ""
    )
    bm_legend = (
        f'<line x1="760" y1="16" x2="790" y2="16" stroke="{COLORS["benchmark"]}" stroke-width="3"/>'
        '<text x="798" y="20" class="legend">沪深300</text>'
        if benchmark_path
        else ""
    )
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="策略净值、沪深300和回撤曲线">
<style>.grid{{stroke:#e2e8f0;stroke-width:1}}.axis{{font:12px "Microsoft YaHei",sans-serif;fill:#64748b}}.legend{{font:13px "Microsoft YaHei",sans-serif;fill:#334155}}</style>
{"".join(grid)}
<line x1="{left}" y1="{dd_top}" x2="{width - right}" y2="{dd_top}" class="grid"/>
<path d="{strategy_path}" fill="none" stroke="{COLORS["strategy"]}" stroke-width="3"/>{bm_path}
<path d="{drawdown_path}" fill="none" stroke="{COLORS["drawdown"]}" stroke-width="2"/>
<line x1="610" y1="16" x2="640" y2="16" stroke="{COLORS["strategy"]}" stroke-width="3"/><text x="648" y="20" class="legend">Alice策略</text>
{bm_legend}<text x="10" y="350" class="axis">回撤</text>{dates}</svg>'''


def _monthly_html(equity: pd.DataFrame) -> str:
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["month"] = eq["date"].dt.to_period("M")
    monthly = eq.groupby("month", observed=False)["total"].last().pct_change() * 100
    cells = []
    for period, value in monthly.dropna().items():
        tone = "positive" if value >= 0 else "negative"
        strength = min(abs(float(value)) / 12.0, 1.0)
        cells.append(
            f'<div class="month {tone}" style="--strength:{strength:.3f}">'
            f"<span>{html.escape(str(period))}</span><strong>{float(value):+.2f}%</strong></div>"
        )
    return "".join(cells) or '<p class="muted">区间不足一个完整月。</p>'


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "开启" if value else "关闭"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _changed_parameters(
    config: type, baseline: type = StrategyConfig
) -> list[tuple[str, Any, Any]]:
    current = config_snapshot(config)
    original = config_snapshot(baseline)
    return [
        (name, original.get(name), current.get(name))
        for name in sorted(current)
        if current.get(name) != original.get(name)
    ]


def write_backtest_report(
    run_dir: str | Path,
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    config: type,
    start: str,
    end: str,
    decisions: list[dict[str, Any]] | None = None,
) -> Path:
    root = Path(run_dir)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    statistics = derive_run_statistics(equity, trades)
    (report_dir / "run_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    decision_statistics = derive_decision_statistics(decisions)
    (report_dir / "decision_statistics.json").write_text(
        json.dumps(decision_statistics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    cards = [
        ("总收益", f"{metrics['total_return_pct']:+.2f}%"),
        ("年化收益", f"{metrics['cagr_pct']:+.2f}%"),
        ("夏普比率", f"{metrics['sharpe_ratio']:.2f}"),
        ("最大回撤", f"{metrics['max_drawdown_pct']:.2f}%"),
        ("超额收益", f"{metrics['excess_return_pct']:+.2f}%"),
        ("胜率", f"{metrics['win_rate_pct']:.1f}%"),
        (
            "盈利因子",
            "∞" if metrics["profit_factor"] is None else f"{metrics['profit_factor']:.2f}",
        ),
        ("年化换手", f"{statistics['annualized_turnover']:.2f}倍"),
    ]
    card_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    changes = _changed_parameters(config)
    change_rows = (
        "".join(
            f"<tr><td><code>{html.escape(name)}</code></td><td>{html.escape(_format_value(before))}</td>"
            f"<td>{html.escape(_format_value(after))}</td></tr>"
            for name, before, after in changes
        )
        or '<tr><td colspan="3">使用 Alice v4 默认参数</td></tr>'
    )

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alice 回测报告</title><style>
:root{{--bg:#f4f7fb;--ink:#172033;--muted:#64748b;--card:#fff;--line:#e2e8f0;--blue:#2563eb;--green:#15803d;--red:#b91c1c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei",system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:34px 22px 60px}}
h1{{margin:0 0 6px;font-size:30px}}h2{{margin:0 0 15px;font-size:20px}}.subtitle,.muted{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.metric,.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 24px #0f172a0a}}.metric{{padding:16px}}.metric span{{display:block;color:var(--muted);font-size:13px}}.metric strong{{display:block;margin-top:7px;font-size:24px}}
.panel{{padding:18px;margin-top:16px}}.chart svg{{display:block;width:100%;height:auto}}.months{{display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:8px}}.month{{padding:10px;border-radius:9px;background:color-mix(in srgb,var(--green) calc(var(--strength)*22%),white);border:1px solid color-mix(in srgb,var(--green) calc(var(--strength)*35%),var(--line))}}.month.negative{{background:color-mix(in srgb,var(--red) calc(var(--strength)*18%),white);border-color:color-mix(in srgb,var(--red) calc(var(--strength)*35%),var(--line))}}.month span{{display:block;color:var(--muted);font-size:12px}}.month strong{{display:block;margin-top:4px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--muted);font-weight:500}}code{{white-space:normal}}.notice{{border-left:4px solid #f59e0b;background:#fff7ed;padding:13px 15px;border-radius:8px;margin-top:16px}}.links a{{color:var(--blue);margin-right:18px}}@media(max-width:800px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}}}@media(max-width:460px){{.metrics{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Alice 单次回测报告</h1><p class="subtitle">{html.escape(start)} 至 {html.escape(end)} · 每次运行独立保存，不覆盖历史结果</p>
<section class="metrics">{card_html}</section>
<section class="panel chart"><h2>净值与回撤</h2>{equity_svg(equity, benchmark)}</section>
<section class="panel"><h2>月度收益</h2><div class="months">{_monthly_html(equity)}</div></section>
<section class="two"><div class="panel"><h2>调仓决策审计</h2><table><tbody>
<tr><th>调仓次数</th><td>{decision_statistics["rebalance_count"]}</td></tr>
<tr><th>市场允许 / 阻断</th><td>{decision_statistics["market_allowed_count"]} / {decision_statistics["market_blocked_count"]}</td></tr>
<tr><th>实际发生买入的调仓</th><td>{decision_statistics["rebalance_with_buys"]}</td></tr>
<tr><th>允许买入时平均候选数</th><td>{decision_statistics["average_candidates_when_allowed"]}</td></tr>
<tr><th>实际买入评分范围</th><td>{_format_value(decision_statistics["selected_score_min"])} / {_format_value(decision_statistics["selected_score_median"])} / {_format_value(decision_statistics["selected_score_max"])}</td></tr>
</tbody></table><p class="muted">评分范围依次为最小值 / 中位数 / 最大值。</p></div>
<div class="panel chart"><h2>实际买入评分分布</h2>{decision_score_svg(decisions)}</div></section>
<section class="two"><div class="panel"><h2>交易与容量提示</h2><table><tbody>
<tr><th>完整卖出交易</th><td>{metrics["total_trades"]}笔</td></tr><tr><th>平均持仓数量</th><td>{statistics["average_positions"]}</td></tr>
<tr><th>平均资金暴露</th><td>{statistics["average_exposure_pct"]:.2f}%</td></tr><tr><th>累计交易摩擦</th><td>¥{statistics["total_friction"]:,.2f}</td></tr>
<tr><th>摩擦/初始资金</th><td>{statistics["friction_pct_of_initial"]:.3f}%</td></tr><tr><th>年化换手估算</th><td>{statistics["annualized_turnover"]:.2f}倍</td></tr>
</tbody></table></div><div class="panel"><h2>相对默认参数的变化</h2><table><thead><tr><th>参数</th><th>默认值</th><th>本次值</th></tr></thead><tbody>{change_rows}</tbody></table></div></section>
<div class="notice"><strong>阅读顺序：</strong>先看多个样本外区间的夏普和最大回撤，再看收益；不要根据这一个区间自动选择参数。当前报告不构成投资建议。</div>
<p class="links"><a href="reports/metrics.json">指标 JSON</a><a href="reports/equity.csv">净值 CSV</a><a href="reports/trades.csv">交易 CSV</a><a href="reports/rebalance_decisions.csv">调仓审计 CSV</a><a href="reports/effective_config.json">有效参数</a></p>
</main></body></html>"""
    path = root / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
