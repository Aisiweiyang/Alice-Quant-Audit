"""
可视化模块 —— 资金曲线、回撤图、收益分布
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# Force font rebuild on import
import os
import glob

for f in glob.glob(os.path.expanduser("~/.cache/matplotlib/fontlist*.json")):
    try:
        os.remove(f)
    except Exception:
        pass


def plot_equity_curve(equity_df, benchmark_df=None, save_path=None):
    """绘制资金曲线 + 回撤图"""
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # 资金曲线
    ax1.plot(eq["date"], eq["total"] / eq["total"].iloc[0] * 100, "b-", linewidth=1.5, label="策略")
    if benchmark_df is not None and not benchmark_df.empty:
        bm = benchmark_df.copy()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm[bm["date"].isin(eq["date"])]
        if len(bm) > 1:
            ax1.plot(
                bm["date"],
                bm["close"] / bm["close"].iloc[0] * 100,
                "gray",
                linewidth=1,
                alpha=0.6,
                label="沪深300",
            )
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("净值 (初始=100)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("回测资金曲线")

    # 回撤
    peak = eq["total"].expanding().max()
    drawdown = (eq["total"] - peak) / peak * 100
    ax2.fill_between(eq["date"], drawdown, 0, color="red", alpha=0.3)
    ax2.plot(eq["date"], drawdown, "r-", linewidth=0.8)
    ax2.set_ylabel("回撤(%)")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("回撤曲线")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  图表已保存: {save_path}")
    plt.close()


def plot_monthly_returns(equity_df, save_path=None):
    """月度收益热力图"""
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["month"] = eq["date"].dt.to_period("M")
    monthly = eq.groupby("month")["total"].last().pct_change() * 100
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["green" if v > 0 else "red" for v in monthly]
    ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=60, fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_ylabel("月收益(%)")
    ax.set_title("月度收益分布")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  图表已保存: {save_path}")
    plt.close()


def plot_trade_distribution(trades_df, save_path=None):
    """单笔交易收益分布直方图"""
    if trades_df is None or trades_df.empty:
        return
    sells = trades_df[trades_df["action"] == "SELL"]
    if sells.empty:
        return
    pnls = sells["pnl"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pnls, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--")
    ax.axvline(x=np.mean(pnls), color="green", linestyle="-", label=f"均值: ¥{np.mean(pnls):,.0f}")
    ax.set_xlabel("单笔盈亏(¥)")
    ax.set_ylabel("次数")
    ax.set_title(f"交易收益分布 (共{len(pnls)}笔, 胜率{(pnls > 0).mean() * 100:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  图表已保存: {save_path}")
    plt.close()
