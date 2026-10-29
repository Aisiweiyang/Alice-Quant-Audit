"""
数据清洗模块 —— 处理停牌、复权、缺失值、ST过滤、未来数据泄露防护
"""

import pandas as pd
import numpy as np


def remove_st_stocks(data_dict):
    """从数据字典中移除ST股票（ST股票数据异常，不适合回测）"""
    # Baostock已在获取时过滤，此函数作为二次保险
    return {k: v for k, v in data_dict.items() if v is not None and len(v) >= 20}


def handle_suspension(df, max_gap_days=5):
    """处理停牌：删除连续停牌超过max_gap_days的区间
    停牌期间price不变，volume=0。回测时应跳过这些区间。
    Returns:
        df with 'suspended' column: True=停牌日, False=正常交易日
    """
    df = df.copy()
    df["suspended"] = (df["volume"] == 0) | (df["volume"].isna())
    # 向前填充停牌期间的close（用于计算均线等技术指标）
    df["close_filled"] = df["close"].replace(0, np.nan).ffill()
    return df


def clean_price_data(df):
    """清洗价格数据：移除极端值、处理缺失
    Returns:
        cleaned DataFrame
    """
    df = df.copy()
    # 移除价格为0或负的行
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    # 移除涨跌幅异常的交易日（>20%涨停但可能数据错误）
    if "pctChg" in df.columns:
        df = df[df["pctChg"].abs() < 21]
    # 填充少量缺失值（向前填充）
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def filter_tradable(df, date, lookback_days=60):
    """判断某只股票在某个日期是否可以交易
    Args:
        df: 该股票的历史数据
        date: 交易日期
        lookback_days: 需要至少多少天历史数据
    Returns:
        (bool, str): (是否可交易, 原因)
    """
    if df is None or len(df) < lookback_days:
        return False, "历史数据不足"
    target = pd.Timestamp(date).normalize()
    before = df[df["date"] <= target]
    if len(before) < lookback_days:
        return False, f"近期数据不足({len(before)}<{lookback_days})"
    last_row = before.iloc[-1]
    last_date = pd.Timestamp(last_row["date"]).normalize()
    if last_date != target:
        return False, "当日无行情（未上市、已退市或停牌缺行）"
    # 停牌检查
    if (
        pd.to_numeric(pd.Series([last_row.get("tradestatus", 1)]), errors="coerce")
        .fillna(1)
        .iloc[0]
        == 0
    ):
        return False, "停牌中"
    if last_row.get("volume", 0) == 0:
        return False, "停牌中"
    is_st = pd.to_numeric(pd.Series([last_row.get("isST", 0)]), errors="coerce").fillna(0).iloc[0]
    if is_st == 1:
        return False, "历史ST状态"
    return True, "OK"


def prevent_lookahead(df, date):
    """防未来数据泄露：截断date之后的数据
    Returns:
        DataFrame: 只包含date及之前的数据
    """
    return df[df["date"] <= date].copy()


def prepare_universe(data_dict, date):
    """准备某个日期的可交易股票池
    Returns:
        list of str: 可交易的股票代码列表
    """
    universe = []
    for code, df in data_dict.items():
        ok, _ = filter_tradable(df, date)
        if ok:
            universe.append(code)
    return universe
