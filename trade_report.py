#!/usr/bin/env python3
"""生成逐笔交易明细报告"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

OUTPUT = os.path.dirname(os.path.abspath(__file__))


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-07"

    # 读取缓存跑回测
    from engine.backtest import run

    portfolio, metrics = run(start, end)

    if portfolio is None:
        print("无数据")
        return

    trades = portfolio.trades
    if not trades:
        print("无交易记录")
        return

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])

    # 计算每笔交易的持有时长和收益率
    trades_detail = []
    buys = df[df["action"] == "buy"].copy()
    sells = df[df["action"] == "sell"].copy()

    # 按股票+买入时间匹配买卖
    for _, sell in sells.iterrows():
        code = sell["code"]
        sell_date = sell["date"]
        # 找最近一次同股票买入
        buy_match = buys[(buys["code"] == code) & (buys["date"] < sell_date)]
        if buy_match.empty:
            continue
        buy = buy_match.iloc[-1]
        buy_price = float(buy["price"])
        sell_price = float(sell["price"])
        pnl_pct = (sell_price / buy_price - 1) * 100
        hold_days = (sell_date - buy["date"]).days

        trades_detail.append(
            {
                "code": code,
                "buy_date": buy["date"].strftime("%Y-%m-%d"),
                "sell_date": sell_date.strftime("%Y-%m-%d"),
                "buy_price": round(buy_price, 2),
                "sell_price": round(sell_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "hold_days": hold_days,
                "shares": int(buy.get("shares", 0)),
            }
        )

    td = pd.DataFrame(trades_detail)
    if td.empty:
        print("无完整交易")
        return

    # 统计
    wins = td[td["pnl_pct"] > 0]
    losses = td[td["pnl_pct"] <= 0]
    win_rate = len(wins) / len(td) * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    avg_hold = td["hold_days"].mean()

    # 生成HTML报告
    html_path = os.path.join(OUTPUT, "output", "reports", "trade_detail.html")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # 交易分布数据
    pnl_bins = pd.cut(
        td["pnl_pct"],
        bins=[-100, -5, -3, -1, 0, 1, 3, 5, 10, 20, 100],
        labels=[
            "<-5%",
            "-5~-3%",
            "-3~-1%",
            "-1~0%",
            "0~1%",
            "1~3%",
            "3~5%",
            "5~10%",
            "10~20%",
            ">20%",
        ],
    )
    pnl_dist = pnl_bins.value_counts().sort_index()

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>逐笔交易明细报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background:#1a1a2e; color:#e0e0e0; padding:20px; }}
.container {{ max-width:1200px; margin:0 auto; }}
h1 {{ color:#e94560; margin-bottom:5px; }}
.subtitle {{ color:#888; margin-bottom:20px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:30px; }}
.card {{ background:#16213e; border-radius:10px; padding:16px; text-align:center; }}
.card .label {{ color:#888; font-size:13px; margin-bottom:6px; }}
.card .value {{ font-size:28px; font-weight:bold; }}
.card .value.green {{ color:#4ecca3; }}
.card .value.red {{ color:#e94560; }}
.card .value.yellow {{ color:#f0c040; }}

table {{ width:100%; border-collapse:collapse; margin-bottom:30px; background:#16213e; border-radius:10px; overflow:hidden; }}
th {{ background:#0f3460; padding:12px 14px; text-align:left; font-weight:600; color:#ccc; font-size:13px; }}
td {{ padding:10px 14px; border-bottom:1px solid #1a1a2e; font-size:13px; }}
tr:hover {{ background:#1a3a5c; }}
.green {{ color:#4ecca3; }}
.red {{ color:#e94560; }}

.chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }}
.chart-box {{ background:#16213e; border-radius:10px; padding:16px; }}
.chart-box h3 {{ color:#ccc; margin-bottom:10px; font-size:14px; }}
.bar {{ display:flex; align-items:center; margin:4px 0; }}
.bar .label {{ width:80px; font-size:12px; color:#888; text-align:right; margin-right:10px; }}
.bar .fill {{ height:20px; border-radius:3px; min-width:2px; }}
.bar .count {{ font-size:11px; margin-left:6px; color:#ccc; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 逐笔交易明细报告</h1>
<p class="subtitle">{start} → {end} | 初始资金 ¥100,000 | 策略 V6</p>

<div class="cards">
<div class="card"><div class="label">总收益</div><div class="value {"green" if metrics.get("total_return_pct", 0) > 0 else "red"}">{metrics.get("total_return_pct", 0):+.1f}%</div></div>
<div class="card"><div class="label">总交易</div><div class="value yellow">{len(td)}笔</div></div>
<div class="card"><div class="label">胜率</div><div class="value {"green" if win_rate > 50 else "yellow"}">{win_rate:.1f}%</div></div>
<div class="card"><div class="label">平均盈利</div><div class="value green">{avg_win:+.2f}%</div></div>
<div class="card"><div class="label">平均亏损</div><div class="value red">{avg_loss:+.2f}%</div></div>
<div class="card"><div class="label">平均持仓</div><div class="value yellow">{avg_hold:.0f}天</div></div>
</div>

<div class="chart-row">
<div class="chart-box">
<h3>盈亏分布</h3>
"""
    colors = [
        "#e94560",
        "#e97060",
        "#f09070",
        "#f0c040",
        "#a0d060",
        "#60c060",
        "#40b040",
        "#30a830",
        "#209020",
        "#4ecca3",
    ]
    for i, (label, count) in enumerate(pnl_dist.items()):
        pct = count / len(td) * 100
        html += f'<div class="bar"><span class="label">{label}</span><div class="fill" style="width:{max(pct * 3, 1)}px;background:{colors[i]}"></div><span class="count">{count}笔 ({pct:.0f}%)</span></div>\n'
    html += "</div>"

    html += """
<div class="chart-box">
<h3>持仓天数 vs 收益</h3>
<table>
<tr><th>持仓天数</th><th>笔数</th><th>平均收益</th><th>胜率</th></tr>
"""
    td["hold_bucket"] = pd.cut(
        td["hold_days"],
        bins=[0, 3, 7, 14, 30, 365],
        labels=["1-3天", "4-7天", "8-14天", "15-30天", ">30天"],
    )
    for bucket, group in td.groupby("hold_bucket", observed=False):
        wr = len(group[group["pnl_pct"] > 0]) / len(group) * 100
        avg = group["pnl_pct"].mean()
        html += f'<tr><td>{bucket}</td><td>{len(group)}</td><td class="{"green" if avg > 0 else "red"}">{avg:+.2f}%</td><td>{wr:.0f}%</td></tr>\n'
    html += "</table></div></div>"

    # 交易明细表
    html += '<h2 style="margin:20px 0 10px;">全部交易明细</h2>'
    html += "<table><tr><th>#</th><th>代码</th><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>收益率</th><th>持仓天</th></tr>"
    for i, (_, t) in enumerate(td.sort_values("sell_date").iterrows()):
        cls = "green" if t["pnl_pct"] > 0 else "red"
        html += f'<tr><td>{i + 1}</td><td>{t["code"]}</td><td>{t["buy_date"]}</td><td>{t["sell_date"]}</td><td>{t["buy_price"]}</td><td>{t["sell_price"]}</td><td class="{cls}">{t["pnl_pct"]:+.2f}%</td><td>{t["hold_days"]}天</td></tr>\n'
    html += "</table>"

    # 月度汇总
    html += '<h2 style="margin:20px 0 10px;">月度交易汇总</h2>'
    td["month"] = pd.to_datetime(td["sell_date"]).dt.to_period("M")
    html += (
        "<table><tr><th>月份</th><th>交易数</th><th>盈利笔</th><th>胜率</th><th>总盈亏</th></tr>"
    )
    for month, group in td.groupby("month", observed=False):
        w = len(group[group["pnl_pct"] > 0])
        html += f'<tr><td>{month}</td><td>{len(group)}</td><td>{w}</td><td>{w / len(group) * 100:.0f}%</td><td class="{"green" if group["pnl_pct"].sum() > 0 else "red"}">{group["pnl_pct"].sum():+.1f}%</td></tr>\n'
    html += "</table>"

    html += "</div></body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"交易报告已保存: {html_path}")

    # 同时保存JSON
    json_path = os.path.join(OUTPUT, "output", "reports", "trades.json")
    td.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"交易JSON已保存: {json_path}")


if __name__ == "__main__":
    main()
