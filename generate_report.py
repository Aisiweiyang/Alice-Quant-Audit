#!/usr/bin/env python3
"""生成多版本回测对比 + 逐笔交易明细的综合HTML报告"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(OUTPUT, "output", "reports")
CHART_DIR = os.path.join(OUTPUT, "output", "charts")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
GREEN = "#4ecca3"
RED = "#e94560"
YELLOW = "#f0c040"
BLUE = "#0f3460"
WHITE = "#e0e0e0"


def img_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_trade_charts(trades_df):
    """生成逐笔交易分析图表"""
    if trades_df.empty:
        return {}

    td = trades_df.copy()
    td["date"] = pd.to_datetime(td["date"])
    buys = td[td["action"] == "BUY"].copy()
    sells = td[td["action"] == "SELL"].copy()

    # 匹配买卖对
    trades_detail = []
    for _, sell in sells.iterrows():
        code = sell["code"]
        sell_date = sell["date"]
        buy_match = buys[(buys["code"] == code) & (buys["date"] < sell_date)]
        if buy_match.empty:
            continue
        buy = buy_match.iloc[-1]
        pnl_pct = (float(sell["price"]) / float(buy["price"]) - 1) * 100
        hold_days = (sell_date - buy["date"]).days
        trades_detail.append(
            {
                "code": code,
                "buy_date": buy["date"],
                "sell_date": sell_date,
                "buy_price": float(buy["price"]),
                "sell_price": float(sell["price"]),
                "pnl_pct": pnl_pct,
                "hold_days": hold_days,
            }
        )

    if not trades_detail:
        return {}, pd.DataFrame(), pd.DataFrame()
    dt = pd.DataFrame(trades_detail)

    charts = {}

    # 图1: 累计盈亏曲线
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    dt_sorted = dt.sort_values("sell_date")
    cumsum = dt_sorted["pnl_pct"].cumsum()
    colors_pnl = [GREEN if v > 0 else RED for v in dt_sorted["pnl_pct"]]
    ax.bar(range(len(cumsum)), dt_sorted["pnl_pct"], color=colors_pnl, alpha=0.7, width=0.8)
    ax.plot(range(len(cumsum)), cumsum, color=YELLOW, linewidth=2, marker="o", markersize=2)
    ax.axhline(y=0, color="#555", linewidth=0.5)
    ax.set_title("逐笔交易盈亏", color=WHITE, fontsize=13, fontweight="bold")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")
    charts["trade_pnl_seq"] = img_to_b64(fig)
    plt.close()

    # 图2: 盈亏分布直方图
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    bins = np.arange(-15, 16, 1)
    ax.hist(dt["pnl_pct"], bins=bins, color=BLUE, edgecolor="#333", alpha=0.8)
    ax.axvline(x=0, color=RED, linewidth=1, linestyle="--")
    ax.axvline(
        x=dt["pnl_pct"].mean(),
        color=YELLOW,
        linewidth=1.5,
        label=f"均值 {dt['pnl_pct'].mean():.1f}%",
    )
    ax.set_title(f"盈亏分布 ({len(dt)}笔)", color=WHITE, fontsize=13, fontweight="bold")
    ax.legend(facecolor=DARK_BG, labelcolor=WHITE, fontsize=9)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")
    charts["trade_hist"] = img_to_b64(fig)
    plt.close()

    # 图3: 持仓天数 vs 收益率
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    cs = [GREEN if v > 0 else RED for v in dt["pnl_pct"]]
    ax.scatter(dt["hold_days"], dt["pnl_pct"], c=cs, alpha=0.6, s=30)
    ax.axhline(y=0, color="#555", linewidth=0.5)
    ax.set_xlabel("持仓天数", color="#888")
    ax.set_ylabel("收益率 %", color="#888")
    ax.set_title("持仓天数 vs 收益率", color=WHITE, fontsize=13, fontweight="bold")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")
    charts["hold_vs_pnl"] = img_to_b64(fig)
    plt.close()

    # 图4: 月度盈亏汇总
    dt["month"] = dt["sell_date"].dt.to_period("M")
    monthly = dt.groupby("month", observed=False).agg(
        trades=("pnl_pct", "count"),
        total_pnl=("pnl_pct", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    months_str = [str(m) for m in monthly.index]
    colors_m = [GREEN if v > 0 else RED for v in monthly["total_pnl"]]
    ax.bar(months_str, monthly["total_pnl"], color=colors_m, alpha=0.8)
    ax.axhline(y=0, color="#555", linewidth=0.5)
    ax.set_title("月度累计盈亏", color=WHITE, fontsize=13, fontweight="bold")
    ax.tick_params(colors="#888", rotation=45)
    for spine in ax.spines.values():
        spine.set_color("#333")
    charts["monthly_pnl"] = img_to_b64(fig)
    plt.close()

    return charts, dt, monthly


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-07"

    # 拉取最新回测的 trades
    from engine.backtest import run
    from strategy.config import StrategyConfig as cfg

    portfolio, metrics = run(start, end)
    if portfolio is None:
        print("回测无数据")
        return

    trades = portfolio.trades
    td = pd.DataFrame(trades) if trades else pd.DataFrame()

    # 读取已有的回测图表（沪深300对比等）
    def load_chart_b64(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    chart_equity = load_chart_b64(os.path.join(CHART_DIR, "equity_curve.png"))
    chart_monthly = load_chart_b64(os.path.join(CHART_DIR, "monthly_returns.png"))
    chart_trade_dist = load_chart_b64(os.path.join(CHART_DIR, "trade_dist.png"))

    # 生成交易分析图表
    trade_charts = {}
    detail_df = pd.DataFrame()
    monthly_df = pd.DataFrame()
    if not td.empty:
        trade_charts, detail_df, monthly_df = generate_trade_charts(td)

    eq_df = pd.DataFrame(portfolio.equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])

    # ===== 构建HTML =====
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化回测综合报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background:{DARK_BG}; color:{WHITE}; padding:24px; line-height:1.5; }}
.container {{ max-width:1300px; margin:0 auto; }}
h1 {{ color:{RED}; font-size:28px; margin-bottom:4px; }}
h2 {{ color:{YELLOW}; font-size:20px; margin:30px 0 14px; border-left:4px solid {RED}; padding-left:12px; }}
h3 {{ color:#ccc; font-size:15px; margin:16px 0 8px; }}
.subtitle {{ color:#888; margin-bottom:24px; font-size:14px; }}

/* 版本对比卡片 */
.version-compare {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin-bottom:30px; }}

/* 指标卡片 */
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:24px; }}
.card {{ background:{CARD_BG}; border-radius:8px; padding:14px 16px; text-align:center; border:1px solid #2a2a4a; }}
.card.best {{ border-color:{GREEN}; box-shadow:0 0 12px rgba(78,204,163,0.15); }}
.card .label {{ color:#888; font-size:12px; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px; }}
.card .value {{ font-size:26px; font-weight:bold; }}
.card .sub {{ font-size:11px; color:#666; margin-top:3px; }}
.green {{ color:{GREEN}; }}
.red {{ color:{RED}; }}
.yellow {{ color:{YELLOW}; }}

/* 图表 */
.chart-img {{ width:100%; border-radius:8px; margin:8px 0; background:{CARD_BG}; }}
.chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
.chart-row.single {{ grid-template-columns:1fr; }}
@media (max-width:800px) {{ .chart-row {{ grid-template-columns:1fr; }} }}

/* 表格 */
table {{ width:100%; border-collapse:collapse; margin:10px 0 20px; background:{CARD_BG}; border-radius:8px; overflow:hidden; font-size:13px; }}
th {{ background:{BLUE}; padding:10px 12px; text-align:left; font-weight:600; color:#aaa; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; }}
td {{ padding:9px 12px; border-bottom:1px solid #1a1a2e; }}
tr:hover {{ background:#1e2d50; }}

/* 版本对比表 */
.vcard {{ background:{CARD_BG}; border-radius:10px; padding:18px; border:1px solid #2a2a4a; }}
.vcard.best {{ border-color:{GREEN}; }}
.vcard h4 {{ font-size:16px; margin-bottom:10px; }}
.vcard .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:13px; }}
.vcard .metrics div {{ display:flex; justify-content:space-between; }}
.vcard .metrics .label {{ color:#888; }}

.trade-table {{ max-height:500px; overflow-y:auto; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }}
.badge.win {{ background:rgba(78,204,163,0.2); color:{GREEN}; }}
.badge.loss {{ background:rgba(233,69,96,0.2); color:{RED}; }}
</style>
</head>
<body>
<div class="container">

<h1>📊 量化选股回测综合报告</h1>
<p class="subtitle">回测区间: 2025-01-01 → 2026-08-07 | 初始资金: ¥100,000 | 股票池: 沪深主板 3272只</p>

<!-- ===== 版本演进对比 ===== -->
<h2>一、策略版本演进对比</h2>
<div class="version-compare">
<div class="vcard">
<h4 style="color:#888;">V3 · 初始版本</h4>
<div class="metrics">
<div><span class="label">总收益</span><span class="red">+12.99%</span></div>
<div><span class="label">夏普比率</span><span>0.67</span></div>
<div><span class="label">最大回撤</span><span class="red">-13.65%</span></div>
<div><span class="label">超额收益</span><span class="red">-8.76%</span></div>
<div><span class="label">胜率</span><span>49.4%</span></div>
<div><span class="label">盈亏比</span><span>1.32</span></div>
<div><span class="label">因子数</span><span>9因子/等权</span></div>
<div><span class="label">择时</span><span>无</span></div>
</div>
</div>

<div class="vcard best">
<h4 style="color:{GREEN};">V6 · 优化版 ⭐</h4>
<div class="metrics">
<div><span class="label">总收益</span><span class="green">+57.06%</span></div>
<div><span class="label">夏普比率</span><span class="green">2.10</span></div>
<div><span class="label">最大回撤</span><span class="yellow">-9.14%</span></div>
<div><span class="label">超额收益</span><span class="green">+35.31%</span></div>
<div><span class="label">胜率</span><span>50.0%</span></div>
<div><span class="label">盈亏比</span><span>1.77</span></div>
<div><span class="label">因子数</span><span>9因子/动量偏重</span></div>
<div><span class="label">择时</span><span>MA20过滤</span></div>
</div>
</div>

<div class="vcard">
<h4 style="color:{RED};">V7 · 过度限制</h4>
<div class="metrics">
<div><span class="label">总收益</span><span class="red">+17.49%</span></div>
<div><span class="label">夏普比率</span><span>0.90</span></div>
<div><span class="label">最大回撤</span><span class="yellow">-10.29%</span></div>
<div><span class="label">超额收益</span><span class="red">-4.26%</span></div>
<div><span class="label">胜率</span><span>41.1%</span></div>
<div><span class="label">盈亏比</span><span>1.27</span></div>
<div><span class="label">因子数</span><span>12因子/分散</span></div>
<div><span class="label">择时</span><span>级联清仓+防抄底</span></div>
</div>
</div>

<div class="vcard best">
<h4 style="color:{YELLOW};">V9 · 均衡版 ✅</h4>
<div class="metrics">
<div><span class="label">总收益</span><span class="green">{metrics.get("total_return_pct", 0):+.2f}%</span></div>
<div><span class="label">夏普比率</span><span class="green">{metrics.get("sharpe_ratio", 0):.2f}</span></div>
<div><span class="label">最大回撤</span><span class="yellow">{metrics.get("max_drawdown_pct", 0):.2f}%</span></div>
<div><span class="label">超额收益</span><span class="green">{metrics.get("excess_return_pct", 0):+.2f}%</span></div>
<div><span class="label">胜率</span><span>{metrics.get("win_rate_pct", 0):.1f}%</span></div>
<div><span class="label">盈亏比</span><span>{metrics.get("profit_factor", 0)}</span></div>
<div><span class="label">因子数</span><span>9因子/动量偏重</span></div>
<div><span class="label">择时</span><span>MA20+MA50过滤</span></div>
</div>
</div>
</div>

<p style="color:#888;font-size:13px;margin-top:-10px;margin-bottom:20px;">
📌 历史结果必须结合样本外收益、最大回撤、交易成本和参数敏感性共同判断；单次回测不代表未来表现。
</p>
"""

    # ===== 当前版本绩效指标 =====
    html += f"""<h2>二、V9 当前绩效详情</h2>
<div class="cards">
<div class="card best"><div class="label">总收益</div><div class="value green">{metrics.get("total_return_pct", 0):+.1f}%</div></div>
<div class="card"><div class="label">年化收益</div><div class="value green">{metrics.get("cagr_pct", 0):+.1f}%</div></div>
<div class="card best"><div class="label">夏普比率</div><div class="value yellow">{metrics.get("sharpe_ratio", 0):.2f}</div></div>
<div class="card"><div class="label">最大回撤</div><div class="value red">{metrics.get("max_drawdown_pct", 0):.2f}%</div></div>
<div class="card"><div class="label">年化波动</div><div class="value">{metrics.get("annual_volatility_pct", 0):.1f}%</div></div>
<div class="card"><div class="label">基准收益</div><div class="value">{metrics.get("benchmark_return_pct", 0):+.1f}%</div></div>
<div class="card best"><div class="label">超额收益</div><div class="value green">{metrics.get("excess_return_pct", 0):+.1f}%</div></div>
<div class="card"><div class="label">总交易</div><div class="value">{metrics.get("total_trades", 0)}笔</div></div>
<div class="card"><div class="label">胜率</div><div class="value">{metrics.get("win_rate_pct", 0):.1f}%</div></div>
<div class="card"><div class="label">盈亏比</div><div class="value">{metrics.get("profit_factor", 0)}</div></div>
<div class="card"><div class="label">均盈利</div><div class="value green">¥{metrics.get("avg_win", 0):,.0f}</div></div>
<div class="card"><div class="label">均亏损</div><div class="value red">¥{metrics.get("avg_loss", 0):,.0f}</div></div>
</div>
"""

    # ===== 回测走势 =====
    html += "<h2>三、回测走势（按日期）</h2>"
    if not eq_df.empty:
        html += "<table><tr><th>日期</th><th>净值</th><th>累计收益</th><th>持仓数</th><th>现金</th></tr>"
        for _, row in eq_df.iterrows():
            pnl = (float(row["total"]) / 100000 - 1) * 100
            cls = "green" if pnl > 0 else "red"
            html += f'<tr><td>{row["date"]}</td><td>¥{float(row["total"]):,.0f}</td><td class="{cls}">{pnl:+.1f}%</td><td>{int(row.get("positions", 0))}</td><td>¥{float(row.get("cash", 0)):,.0f}</td></tr>\n'
        html += "</table>"

    # ===== 交易分析图表 =====
    if trade_charts:
        html += "<h2>四、逐笔交易分析图表</h2>"

    # 先放回测图表
    html += '<div class="chart-row">'
    if chart_equity:
        html += f'<div><h3>净值曲线 vs 沪深300</h3><img class="chart-img" src="data:image/png;base64,{chart_equity}"></div>'
    if chart_monthly:
        html += f'<div><h3>月度收益分布</h3><img class="chart-img" src="data:image/png;base64,{chart_monthly}"></div>'
    html += "</div>"
    if chart_trade_dist:
        html += f'<div class="chart-row single"><div><h3>交易分布</h3><img class="chart-img" src="data:image/png;base64,{chart_trade_dist}"></div></div>'

    if trade_charts:
        html += f'<div class="chart-row"><div><h3>逐笔盈亏序列</h3><img class="chart-img" src="data:image/png;base64,{trade_charts["trade_pnl_seq"]}"></div>'
        html += f'<div><h3>盈亏分布直方图</h3><img class="chart-img" src="data:image/png;base64,{trade_charts["trade_hist"]}"></div></div>'
        html += f'<div class="chart-row"><div><h3>持仓天数 vs 收益率</h3><img class="chart-img" src="data:image/png;base64,{trade_charts["hold_vs_pnl"]}"></div>'
        html += f'<div><h3>月度盈亏汇总</h3><img class="chart-img" src="data:image/png;base64,{trade_charts["monthly_pnl"]}"></div></div>'

    # ===== 逐笔交易明细 =====
    if not detail_df.empty:
        html += "<h2>五、逐笔交易明细</h2>"
        html += '<div class="trade-table"><table><tr><th>#</th><th>代码</th><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>收益率</th><th>持仓天</th><th>结果</th></tr>'
        for i, (_, t) in enumerate(detail_df.sort_values("sell_date").iterrows()):
            cls = "green" if t["pnl_pct"] > 0 else "red"
            badge = (
                '<span class="badge win">盈利</span>'
                if t["pnl_pct"] > 0
                else '<span class="badge loss">亏损</span>'
            )
            html += f'<tr><td>{i + 1}</td><td>{t["code"]}</td><td>{t["buy_date"].strftime("%Y-%m-%d") if hasattr(t["buy_date"], "strftime") else t["buy_date"]}</td><td>{t["sell_date"].strftime("%Y-%m-%d") if hasattr(t["sell_date"], "strftime") else t["sell_date"]}</td><td>¥{t["buy_price"]:.2f}</td><td>¥{t["sell_price"]:.2f}</td><td class="{cls}">{t["pnl_pct"]:+.2f}%</td><td>{t["hold_days"]}天</td><td>{badge}</td></tr>\n'
        html += "</table></div>"

    # ===== 月度汇总 =====
    if not monthly_df.empty:
        html += "<h2>六、月度交易汇总</h2>"
        html += "<table><tr><th>月份</th><th>交易数</th><th>盈利笔</th><th>胜率</th><th>总盈亏</th></tr>"
        for month, row in monthly_df.iterrows():
            cls = "green" if row["total_pnl"] > 0 else "red"
            html += f'<tr><td>{month}</td><td>{int(row["trades"])}</td><td>{int(row["trades"] * row["win_rate"] / 100)}</td><td>{row["win_rate"]:.0f}%</td><td class="{cls}">{row["total_pnl"]:+.1f}%</td></tr>\n'
        html += "</table>"

    # ===== 因子配置 =====
    html += "<h2>七、当前因子配置</h2>"
    html += "<table><tr><th>因子</th><th>权重</th><th>方向</th><th>说明</th></tr>"
    fw = cfg.factor_weights
    factor_info = {
        "momentum": ("+", "5日价格动量，最重要因子"),
        "reversal": ("−", "均值回复，A股降权"),
        "adx": ("+", "趋势强度（暂未启用）"),
        "volatility": ("−", "5日波动率，低波优先"),
        "turnover": ("+", "5日均换手率"),
        "volume_ratio": ("+", "量比（暂未启用）"),
        "vol_price_corr": ("+", "量价相关性，放量上涨=确认"),
        "steadiness": ("+", "涨幅平稳度，慢牛>暴涨"),
        "price_position": ("−", "价格位置防追高（暂未启用）"),
        "rsi": ("+", "RSI(14)，偏多50-55"),
        "bb": ("+", "布林带位置"),
        "macd": ("+", "MACD金叉+柱状图"),
    }
    for name, weight in sorted(fw.items(), key=lambda x: x[1], reverse=True):
        info = factor_info.get(name, ("", ""))
        cls = "green" if weight >= 0.10 else ("yellow" if weight > 0 else "#555")
        html += f'<tr><td>{name}</td><td style="color:{cls};font-weight:bold">{weight:.2f}</td><td>{info[0]}</td><td style="color:#888">{info[1]}</td></tr>\n'
    html += "</table>"

    # ===== 风控参数 =====
    html += "<h2>八、风控参数</h2>"
    html += f"""<table>
<tr><th>参数</th><th>值</th><th>说明</th></tr>
<tr><td>止盈</td><td style="color:{GREEN}">{cfg.take_profit_pct * 100:.0f}%</td><td>达到目标收益立即卖出</td></tr>
<tr><td>止损</td><td style="color:{RED}">{cfg.stop_loss_pct * 100:.0f}%</td><td>跌破成本价N%立即卖出</td></tr>
<tr><td>最大持仓</td><td>{cfg.max_positions}只</td><td>集中持仓</td></tr>
<tr><td>单只仓位上限</td><td>{cfg.max_position_pct * 100:.0f}%</td><td>单只股票最大仓位</td></tr>
<tr><td>调仓频率</td><td>{cfg.rebalance_freq}</td><td>每周一调仓</td></tr>
<tr><td>PE上限</td><td>{cfg.pe_max}</td><td>过滤高估值</td></tr>
<tr><td>大盘择时</td><td>MA{cfg.ma_period}</td><td>收盘价&lt;MA{cfg.ma_period}不买</td></tr>
<tr><td>个股趋势</td><td>MA{cfg.stock_ma_period}</td><td>收盘价&lt;MA{cfg.stock_ma_period}不买</td></tr>
<tr><td>佣金</td><td>万三</td><td>最低5元</td></tr>
<tr><td>印花税</td><td>千一</td><td>仅卖出</td></tr>
<tr><td>滑点</td><td>千一</td><td>模拟冲击成本</td></tr>
</table>"""

    html += f"""
<p style="color:#666;font-size:12px;margin-top:30px;text-align:center;">
量化回测报告 · 生成于 {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} · 数据源: Baostock · 策略: V9 均衡版
</p>
</div></body></html>"""

    # 保存
    report_path = os.path.join(REPORT_DIR, "综合回测报告.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"综合报告已保存: {report_path}")
    export_dir = os.environ.get("QUANT_REPORT_EXPORT")
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, "综合回测报告.html")
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"导出副本: {export_path}")


if __name__ == "__main__":
    main()
