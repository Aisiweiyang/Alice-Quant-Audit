"""
券商研报数据 —— 多数据源（东方财富/同花顺/新浪），获取最新研报评级和摘要
"""

import akshare as ak
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.expanduser("~/.quant_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== 个股研报 ====================


def get_reports_东方财富(code, days=30):
    """东方财富研报（数据最全）"""
    clean = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    try:
        df = ak.stock_research_report_em(symbol=clean)
        if df is None or df.empty:
            return []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent = df[df["研究报告日期"] >= cutoff]
        results = []
        for _, row in recent.iterrows():
            results.append(
                {
                    "source": "东方财富",
                    "date": str(row.get("研究报告日期", "")),
                    "title": str(row.get("研究报告名称", "")),
                    "rating": str(row.get("评级", "")),
                    "org": str(row.get("研究机构", "")),
                    "analyst": str(row.get("作者", "")),
                }
            )
        return results[:5]
    except Exception:
        return []


def get_reports_同花顺(code, days=30):
    """同花顺研报（备用源）"""
    clean = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    try:
        df = ak.stock_research_report_ths(symbol=clean)
        if df is None or df.empty:
            return []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent = df[df["日期"] >= cutoff]
        results = []
        for _, row in recent.iterrows():
            results.append(
                {
                    "source": "同花顺",
                    "date": str(row.get("日期", "")),
                    "title": str(row.get("研报标题", "")),
                    "rating": str(row.get("评级", "")),
                    "org": str(row.get("机构", "")),
                    "analyst": "",
                }
            )
        return results[:5]
    except Exception:
        return []


def get_reports_新浪(code, days=30):
    """新浪财经研报（备用源）"""
    clean = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    try:
        df = ak.stock_research_report_sina(symbol=clean)
        if df is None or df.empty:
            return []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        if "日期" in df.columns:
            recent = df[df["日期"] >= cutoff]
        else:
            recent = df.head(10)
        results = []
        for _, row in recent.iterrows():
            results.append(
                {
                    "source": "新浪",
                    "date": str(row.iloc[0]) if len(row) > 0 else "",
                    "title": str(row.iloc[1]) if len(row) > 1 else "",
                    "rating": str(row.iloc[2]) if len(row) > 2 else "",
                    "org": str(row.iloc[3]) if len(row) > 3 else "",
                    "analyst": "",
                }
            )
        return results[:5]
    except Exception:
        return []


def get_stock_reports(code, days=30):
    """多源合并获取个股研报（自动切换可用源）"""
    sources = [
        ("东方财富", get_reports_东方财富),
        ("同花顺", get_reports_同花顺),
        ("新浪", get_reports_新浪),
    ]
    all_reports = []
    for name, func in sources:
        try:
            r = func(code, days)
            if r:
                all_reports.extend(r)
                if len(all_reports) >= 5:
                    break  # 至少拿到5篇就停
        except Exception:
            pass
    return sorted(all_reports, key=lambda x: x["date"], reverse=True)[:8]


# ==================== 全市场热门研报 ====================


def get_hot_reports(days=7, top_n=20):
    """获取近期全市场热门买入评级研报"""
    all_reports = []
    try:
        df = ak.stock_research_report_em(symbol="")
        if df is not None and not df.empty:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            recent = df[df["研究报告日期"] >= cutoff]
            recent = recent[recent["评级"].str.contains("买入|增持|推荐", na=False)]
            for _, row in recent.head(top_n).iterrows():
                all_reports.append(
                    {
                        "code": str(row.get("股票代码", "")),
                        "name": str(row.get("股票名称", "")),
                        "date": str(row.get("研究报告日期", "")),
                        "title": str(row.get("研究报告名称", "")),
                        "rating": str(row.get("评级", "")),
                        "org": str(row.get("研究机构", "")),
                    }
                )
    except Exception:
        pass
    return all_reports[:top_n]


# ==================== 个股公告/新闻 ====================


def get_stock_news(code, days=7):
    """获取个股近期重要公告标题"""
    clean = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    news = []
    # 东方财富公告
    try:
        df = ak.stock_notice_report(symbol=clean)
        if df is not None and not df.empty:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            if "日期" in df.columns:
                recent = df[df["日期"] >= cutoff]
            else:
                recent = df.head(10)
            for _, row in recent.iterrows():
                news.append(str(row.iloc[0])[:80])
    except Exception:
        pass
    # 新浪新闻
    try:
        df2 = ak.stock_news_sina(symbol=clean)
        if df2 is not None and not df2.empty:
            for _, row in df2.head(5).iterrows():
                t = str(row.iloc[0])[:80]
                if t not in news:
                    news.append(t)
    except Exception:
        pass
    return news[:15]


# ==================== 批量增强 ====================


def enrich_results(results_df):
    """为选股结果添加研报评级和最近公告数"""
    ratings, news_counts = [], []
    for _, row in results_df.iterrows():
        code = row["代码"]
        reports = get_stock_reports(code, days=30)
        if reports:
            latest = reports[0]
            ratings.append(f"{latest['rating']}({latest['org'][:6]})")
        else:
            ratings.append("-")
        n = get_stock_news(code, days=7)
        news_counts.append(len(n))
    results_df["最新研报"] = ratings
    results_df["近期公告"] = news_counts
    return results_df
