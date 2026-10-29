#!/usr/bin/env python3
"""Create a self-contained visual comparison dashboard without plotting dependencies."""

import argparse
import json
import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "navy": "#17324d",
    "blue": "#2474a6",
    "orange": "#d47b24",
    "green": "#228b68",
    "red": "#c64b4b",
    "gray": "#687782",
    "grid": "#dce4e9",
    "paper": "#ffffff",
    "ink": "#17212b",
}


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(description="生成新旧策略可视化看板")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--research-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _canvas(title, subtitle=""):
    image = Image.new("RGB", (1600, 800), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 32), title, fill=COLORS["ink"], font=font(34, True))
    if subtitle:
        draw.text((72, 78), subtitle, fill=COLORS["gray"], font=font(19))
    return image, draw


def _line_chart(path, title, subtitle, series, percent=True):
    image, draw = _canvas(title, subtitle)
    left, top, right, bottom = 120, 125, 1535, 690
    draw.rectangle((left, top, right, bottom), outline=COLORS["grid"], width=2)

    all_values = np.concatenate([item[1]["value"].dropna().to_numpy() for item in series])
    y_min, y_max = float(np.min(all_values)), float(np.max(all_values))
    padding = max((y_max - y_min) * 0.12, 0.02 if percent else 1.0)
    y_min -= padding
    y_max += padding
    dates = np.concatenate(
        [
            item[1]["date"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
            for item in series
        ]
    )
    x_min, x_max = int(dates.min()), int(dates.max())

    def x_coord(value):
        return left + (int(value) - x_min) / max(x_max - x_min, 1) * (right - left)

    def y_coord(value):
        return bottom - (float(value) - y_min) / max(y_max - y_min, 1e-12) * (bottom - top)

    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = y_coord(value)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        label = f"{value * 100:.0f}%" if percent else f"{value:.1f}"
        draw.text((25, y - 11), label, fill=COLORS["gray"], font=font(17))

    for step in range(6):
        value = int(x_min + (x_max - x_min) * step / 5)
        x = x_coord(value)
        label = pd.Timestamp(value).strftime("%Y-%m")
        draw.text((x - 34, bottom + 15), label, fill=COLORS["gray"], font=font(15))

    for name, frame, color in series:
        points = [
            (x_coord(row.date.value), y_coord(row.value))
            for row in frame.itertuples()
            if pd.notna(row.value)
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4, joint="curve")

    legend_x = 115
    for name, _, color in series:
        draw.line((legend_x, 740, legend_x + 38, 740), fill=color, width=5)
        draw.text((legend_x + 48, 726), name, fill=COLORS["ink"], font=font(19))
        legend_x += 270
    image.save(path, quality=95)


def _bar_chart(path, title, subtitle, labels, groups, percent=True):
    image, draw = _canvas(title, subtitle)
    left, top, right, bottom = 110, 130, 1535, 675
    values = np.concatenate([np.asarray(group[1], dtype=float) for group in groups])
    lower = min(float(np.nanmin(values)), 0.0)
    upper = max(float(np.nanmax(values)), 0.0)
    padding = max((upper - lower) * 0.12, 1.0)
    lower -= padding
    upper += padding

    def y_coord(value):
        return bottom - (float(value) - lower) / max(upper - lower, 1e-12) * (bottom - top)

    for step in range(6):
        value = lower + (upper - lower) * step / 5
        y = y_coord(value)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        suffix = "%" if percent else ""
        draw.text((28, y - 10), f"{value:.1f}{suffix}", fill=COLORS["gray"], font=font(15))

    zero = y_coord(0)
    slot = (right - left) / max(len(labels), 1)
    group_width = min(slot * 0.72, 46)
    bar_width = group_width / max(len(groups), 1)
    for group_index, (name, data, color) in enumerate(groups):
        for index, value in enumerate(data):
            center = left + slot * (index + 0.5)
            x0 = center - group_width / 2 + group_index * bar_width
            x1 = x0 + bar_width - 2
            draw.rectangle(
                (x0, min(zero, y_coord(value)), x1, max(zero, y_coord(value))), fill=color
            )

    tick_step = max(1, len(labels) // 12)
    for index, label in enumerate(labels):
        if index % tick_step == 0 or index == len(labels) - 1:
            center = left + slot * (index + 0.5)
            draw.text((center - 28, 690), str(label), fill=COLORS["gray"], font=font(14))
    legend_x = 115
    for name, _, color in groups:
        draw.rectangle((legend_x, 747, legend_x + 28, 765), fill=color)
        draw.text((legend_x + 38, 741), name, fill=COLORS["ink"], font=font(18))
        legend_x += 280
    image.save(path, quality=95)


def _horizontal_bars(path, title, subtitle, labels, values, color):
    image, draw = _canvas(title, subtitle)
    left, top, right, bottom = 500, 130, 1500, 720
    maximum = max(float(np.max(values)), 0.001)
    minimum = min(float(np.min(values)), 0.0)
    span = max(maximum - minimum, 1e-9)
    row_height = (bottom - top) / max(len(labels), 1)
    zero_x = left + (0 - minimum) / span * (right - left)
    draw.line((zero_x, top, zero_x, bottom), fill=COLORS["gray"], width=2)
    for index, (label, value) in enumerate(zip(labels, values)):
        y0 = top + index * row_height + 8
        y1 = top + (index + 1) * row_height - 8
        value_x = left + (float(value) - minimum) / span * (right - left)
        draw.rectangle((min(zero_x, value_x), y0, max(zero_x, value_x), y1), fill=color)
        draw.text((30, y0 + 4), str(label), fill=COLORS["ink"], font=font(18))
        draw.text(
            (value_x + 10 if value >= 0 else value_x - 70, y0 + 4),
            f"{value:.3f}",
            fill=COLORS["ink"],
            font=font(17),
        )
    image.save(path, quality=95)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_rows(legacy, factor):
    fields = [
        ("总收益", "total_return_pct", "%"),
        ("年化收益", "cagr_pct", "%"),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("年化波动", "annual_volatility_pct", "%"),
        ("夏普比率", "sharpe_ratio", ""),
        ("超额收益", "excess_return_pct", "%"),
        ("交易次数", "total_trades", ""),
        ("胜率", "win_rate_pct", "%"),
        ("盈亏比", "profit_factor", ""),
    ]
    rows = []
    for label, key, suffix in fields:
        rows.append(
            f"<tr><th>{label}</th><td>{legacy.get(key, '—')}{suffix}</td>"
            f"<td>{factor.get(key, '—')}{suffix}</td></tr>"
        )
    return "".join(rows)


def main():
    args = parse_args()
    assets = os.path.join(args.output_dir, "assets")
    os.makedirs(assets, exist_ok=True)

    baseline_equity = pd.read_csv(
        os.path.join(args.baseline_dir, "reports", "equity.csv"), parse_dates=["date"]
    )
    factor_equity = pd.read_csv(
        os.path.join(args.research_dir, "full_period", "equity.csv"), parse_dates=["date"]
    )
    baseline_trades = pd.read_csv(os.path.join(args.baseline_dir, "reports", "trades.csv"))
    factor_trades = pd.read_csv(os.path.join(args.research_dir, "full_period", "trades.csv"))
    comparison = _load_json(os.path.join(args.research_dir, "comparison.json"))
    legacy_metrics = comparison["legacy_full_period"]
    factor_metrics = comparison["full_period"]
    legacy_oos = _load_json(os.path.join(args.research_dir, "legacy_oos_metrics.json"))
    factor_oos = comparison["out_of_sample"]
    config = _load_json(os.path.join(args.research_dir, "selected_config.json"))
    sweep = pd.read_csv(os.path.join(args.research_dir, "training_sweep.csv"))

    with open(args.cache, "rb") as handle:
        cache = pickle.load(handle)
    benchmark = cache["sh.000300"][["date", "close"]].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    start = max(baseline_equity["date"].min(), factor_equity["date"].min())
    end = min(baseline_equity["date"].max(), factor_equity["date"].max())
    benchmark = benchmark[benchmark["date"].between(start, end)]

    equity_series = []
    for name, frame, color in (
        ("生产基线", baseline_equity, COLORS["navy"]),
        ("截面因子研究", factor_equity, COLORS["orange"]),
    ):
        current = frame[frame["date"].between(start, end)][["date", "total"]].copy()
        current["value"] = current["total"] / current["total"].iloc[0] - 1
        equity_series.append((name, current[["date", "value"]], color))
    benchmark["value"] = benchmark["close"] / benchmark["close"].iloc[0] - 1
    equity_series.append(("沪深300", benchmark[["date", "value"]], COLORS["gray"]))
    _line_chart(
        os.path.join(assets, "equity_comparison.png"),
        "净值对比：原策略仍占优",
        "同一缓存、同一交易成本、前收盘信号在下一开盘成交",
        equity_series,
    )

    drawdown_series = []
    for name, frame, color in (
        ("生产基线", baseline_equity, COLORS["navy"]),
        ("截面因子研究", factor_equity, COLORS["orange"]),
    ):
        current = frame[frame["date"].between(start, end)][["date", "total"]].copy()
        current["value"] = current["total"] / current["total"].cummax() - 1
        drawdown_series.append((name, current[["date", "value"]], color))
    _line_chart(
        os.path.join(assets, "drawdown_comparison.png"),
        "回撤对比：越接近零越稳",
        "生产基线最大回撤 -7.23%；研究模型 -11.84%",
        drawdown_series,
    )

    monthly = {}
    for name, frame in (("生产基线", baseline_equity), ("截面因子研究", factor_equity)):
        indexed = frame.set_index("date")["total"].sort_index()
        monthly[name] = indexed.resample("ME").last().pct_change().mul(100)
    months = sorted(
        set(monthly["生产基线"].dropna().index) & set(monthly["截面因子研究"].dropna().index)
    )
    labels = [date.strftime("%y-%m") for date in months]
    _bar_chart(
        os.path.join(assets, "monthly_returns.png"),
        "月度收益：观察策略在哪些市场阶段失效",
        "单月为负并不可怕；连续同类环境下失效才值得检查",
        labels,
        [
            ("生产基线", [monthly["生产基线"].loc[date] for date in months], COLORS["navy"]),
            (
                "截面因子研究",
                [monthly["截面因子研究"].loc[date] for date in months],
                COLORS["orange"],
            ),
        ],
    )

    baseline_pnl = (
        baseline_trades.loc[baseline_trades["action"] == "SELL", "pnl"].dropna().to_numpy()
    )
    factor_pnl = factor_trades.loc[factor_trades["action"] == "SELL", "pnl"].dropna().to_numpy()
    combined = np.concatenate([baseline_pnl, factor_pnl])
    low, high = np.percentile(combined, [2, 98])
    bins = np.linspace(low, high, 15)
    base_hist, _ = np.histogram(np.clip(baseline_pnl, low, high), bins=bins)
    factor_hist, _ = np.histogram(np.clip(factor_pnl, low, high), bins=bins)
    bin_labels = [f"{int((bins[i] + bins[i + 1]) / 2)}" for i in range(len(bins) - 1)]
    _bar_chart(
        os.path.join(assets, "trade_distribution.png"),
        "单笔卖出盈亏分布",
        "横轴为人民币盈亏区间中心；两端按2%/98%分位截断展示",
        bin_labels,
        [("生产基线", base_hist, COLORS["navy"]), ("截面因子研究", factor_hist, COLORS["orange"])],
        percent=False,
    )

    top_sweep = sweep.sort_values("objective", ascending=False).head(10).iloc[::-1]
    _horizontal_bars(
        os.path.join(assets, "parameter_ranking.png"),
        "训练期参数排名",
        "目标函数 = Sharpe + 0.03×CAGR + 0.04×最大回撤；只使用2025年",
        top_sweep["label"].tolist(),
        top_sweep["objective"].to_numpy(),
        COLORS["blue"],
    )

    weight_labels = [key.replace("_rank", "") for key in config["factor_weights"]]
    weight_values = [config["factor_weights"][key] for key in config["factor_weights"]]
    _horizontal_bars(
        os.path.join(assets, "factor_weights.png"),
        "入选研究模型的因子权重",
        "权重只决定当日截面排名的组合方式；不是预期收益率",
        weight_labels,
        weight_values,
        COLORS["green"],
    )

    dashboard = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>量化策略回测看板</title>
<style>
:root{{--navy:#17324d;--orange:#d47b24;--ink:#17212b;--muted:#687782;--paper:#fff;--bg:#eef2f4}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font-family:Arial,'Microsoft YaHei',sans-serif;color:var(--ink)}}
header{{background:var(--navy);color:white;padding:42px max(28px,calc((100% - 1280px)/2))}}header h1{{margin:0 0 10px;font-size:36px}}header p{{margin:0;color:#dbe7ee}}
main{{max-width:1280px;margin:28px auto;padding:0 20px}}.notice{{background:#fff4d7;border-left:5px solid #d69b22;padding:18px 22px;border-radius:8px;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.card{{background:white;border-radius:12px;padding:22px;box-shadow:0 6px 20px #17324d10}}h2{{margin-top:0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e5ebef;text-align:right}}th:first-child{{text-align:left}}thead th{{background:#f3f6f8}}.chart{{background:white;border-radius:12px;margin:22px 0;padding:12px;box-shadow:0 6px 20px #17324d10}}.chart img{{width:100%;display:block}}code{{background:#edf2f5;padding:2px 5px;border-radius:4px}}.good{{color:#1e8060}}.warn{{color:#b4651d}}footer{{color:var(--muted);padding:20px 0 40px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>量化策略回测看板</h1><p>数据截至 2026-08-06 · 缓存 3,272 个标的 · 研究用途</p></header><main>
<div class="notice"><strong>结论：</strong>生产基线仍然更优，不替换 cron。截面因子模型保留为研究观察名单和交叉验证工具；后续新增基本面质量与行业数据后再重新评估。</div>
<div class="grid"><section class="card"><h2>全区间（2025-01—2026-08）</h2><table><thead><tr><th>指标</th><th>生产基线</th><th>截面因子研究</th></tr></thead><tbody>{_metric_rows(legacy_metrics, factor_metrics)}</tbody></table></section>
<section class="card"><h2>2026 样本外</h2><table><thead><tr><th>指标</th><th>生产基线</th><th>截面因子研究</th></tr></thead><tbody>{_metric_rows(legacy_oos, factor_oos)}</tbody></table><p class="warn">生产基线子区间沿用2025末持仓；研究模型样本外从现金重新开始，二者不完全等价，因此以方向性比较为主。</p></section></div>
<div class="chart"><img src="assets/equity_comparison.png" alt="净值对比"></div>
<div class="chart"><img src="assets/drawdown_comparison.png" alt="回撤对比"></div>
<div class="chart"><img src="assets/monthly_returns.png" alt="月度收益"></div>
<div class="chart"><img src="assets/trade_distribution.png" alt="交易盈亏分布"></div>
<div class="grid"><div class="chart"><img src="assets/parameter_ranking.png" alt="参数排名"></div><div class="chart"><img src="assets/factor_weights.png" alt="因子权重"></div></div>
<section class="card"><h2>怎样读这份结果</h2><p>先看样本外是否仍有超额，再看回撤和波动是否可承受，最后才看总收益。研究模型在2026年跑赢沪深300，但仍落后生产基线，因此它证明“因子有信息”，没有证明“应替换现有策略”。</p><p>完整数据位于 <code>output/factor_research</code>；每日研究名单位于 <code>output/daily_factor</code>。</p></section>
<footer>回测可能受幸存者偏差、复权口径、停牌/涨跌停成交假设和短样本影响，不构成投资建议。</footer></main></body></html>"""
    output_path = os.path.join(args.output_dir, "index.html")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(dashboard)
    print(output_path)


if __name__ == "__main__":
    main()
