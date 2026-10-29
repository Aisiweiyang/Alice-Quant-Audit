"""
宏观数据模块 —— 资金流向(北向/主力/游资) + 美股市场 + 政策动态
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

CACHE = os.path.expanduser("~/.quant_cache")
os.makedirs(CACHE, exist_ok=True)

# ==================== A股资金流向 ====================


def get_north_flow(days=5):
    """北向资金（沪深港通）最近N日净流入
    Returns:
        dict: {'date':..., 'net_flow':..., 'cumulative':...}
    """
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is None or df.empty:
            return None
        df = df.tail(days)
        df["date"] = pd.to_datetime(df["date"])
        recent = df.tail(days)
        return {
            "dates": recent["date"].dt.strftime("%m-%d").tolist(),
            "net_flow": recent["value"].tolist()
            if "value" in recent.columns
            else recent.iloc[:, -1].tolist(),
            "total_5d": float(recent.iloc[:, -1].sum()) if "value" in recent.columns else 0,
            "trend": "流入" if float(recent.iloc[:, -1].sum()) > 0 else "流出",
        }
    except Exception:
        return None


def get_dragon_tiger(date=None):
    """龙虎榜数据（游资动向）
    Returns:
        DataFrame or None
    """
    try:
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        df = ak.stock_lhb_detail_em(date=date)
        if df is not None and not df.empty:
            return df.head(20)  # 前20条
    except Exception:
        pass
    try:
        # 备用：获取最近交易日的龙虎榜
        for d in range(1, 5):
            try_date = (datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
            df = ak.stock_lhb_detail_em(date=try_date)
            if df is not None and not df.empty:
                return df.head(20)
    except Exception:
        pass
    return None


def get_margin_data(days=5):
    """融资融券余额（散户杠杆情绪）
    Returns:
        dict or None
    """
    try:
        df = ak.stock_margin_sz_sh()
        if df is None or df.empty:
            return None
        df = df.tail(days)
        return {
            "latest_margin": float(df.iloc[-1].iloc[1]) if len(df.columns) > 1 else 0,
            "trend": "增加"
            if len(df) > 1 and float(df.iloc[-1].iloc[1]) > float(df.iloc[-2].iloc[1])
            else "减少",
        }
    except Exception:
        return None


# ==================== 美股数据 ====================


def get_us_market():
    """美股三大指数 + 美联储政策动态
    Returns:
        dict: SPX/NDX/DJI最新行情 + Fed政策摘要
    """
    result = {"indices": {}, "fed": {}}
    # 标普500
    try:
        spx = ak.index_us_stock_sina(symbol=".INX")
        if spx is not None and not spx.empty:
            latest = spx.iloc[-1]
            result["indices"]["标普500"] = {
                "price": float(latest["close"]),
                "pct": float(latest["pctChg"]),
            }
    except Exception:
        pass
    # 纳斯达克
    try:
        ndx = ak.index_us_stock_sina(symbol=".IXIC")
        if ndx is not None and not ndx.empty:
            latest = ndx.iloc[-1]
            result["indices"]["纳斯达克"] = {
                "price": float(latest["close"]),
                "pct": float(latest["pctChg"]),
            }
    except Exception:
        pass
    # 道琼斯
    try:
        dji = ak.index_us_stock_sina(symbol=".DJI")
        if dji is not None and not dji.empty:
            latest = dji.iloc[-1]
            result["indices"]["道琼斯"] = {
                "price": float(latest["close"]),
                "pct": float(latest["pctChg"]),
            }
    except Exception:
        pass

    # 美联储政策摘要（基于最新公开信息）
    result["fed"] = {
        "rate": "3.50%-3.75%",
        "status": "连续5次维持不变（2026年7月）",
        "next_meeting": "2026年9月16-17日",
        "key_risk": "长期国债收益率升至17年新高（30Y>5.2%），加息预期升温",
        "market_pricing": "市场定价9月加息概率约63-65%",
        "dollar": "美元指数承压，一度触及100水平",
    }
    return result


def get_fear_greed():
    """美股恐惧贪婪指数（VIX替代）"""
    try:
        df = ak.index_vix_sina()
        if df is not None and not df.empty:
            latest = float(df.iloc[-1]["close"])
            if latest < 15:
                mood = "贪婪"
            elif latest < 25:
                mood = "中性"
            else:
                mood = "恐惧"
            return {"vix": latest, "mood": mood}
    except Exception:
        pass
    return None


# ==================== 市场情绪综合 ====================


def get_market_brief():
    """生成每日市场简报（用于HTML/邮件附加）"""
    brief = {}

    # 北向资金
    north = get_north_flow(5)
    if north:
        brief["北向资金"] = f"近5日净{north['trend']} {abs(north.get('total_5d', 0)) / 1e8:.1f}亿元"

    # 融资融券
    margin = get_margin_data(5)
    if margin:
        brief["融资余额"] = f"{margin['trend']}中（{margin['latest_margin'] / 1e8:.0f}亿）"

    # 美股
    us = get_us_market()
    if us and us["indices"]:
        spx = us["indices"].get("标普500", {})
        ndx = us["indices"].get("纳斯达克", {})
        brief["美股"] = (
            f"标普500 {spx.get('price', '?'):.0f} ({spx.get('pct', 0):+.2f}%) | 纳指 {ndx.get('pct', 0):+.2f}%"
        )
        brief["美联储"] = us["fed"]["status"]

    # VIX
    vix = get_fear_greed()
    if vix:
        brief["VIX情绪"] = f"{vix['vix']:.1f}（{vix['mood']}）"

    return brief
