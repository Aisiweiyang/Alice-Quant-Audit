"""
因子库 —— 从K线数据计算量化因子
可扩展：添加新因子只需在compute_all()中增加计算逻辑
"""

import pandas as pd
import numpy as np


def n_day_return(closes, n):
    """最近N个交易日的涨跌幅(%)，与日pctChg的复利口径一致。

    N=1即当日涨跌幅：c[-1]/c[-2]-1。因此N日动量需要N+1个收盘价。
    """
    if len(closes) < n + 1:
        return None
    base = closes[-(n + 1)]
    if base <= 0:
        return None
    return (closes[-1] / base - 1) * 100


def wilder_adx(high, low, close, period=14):
    """标准 Wilder ADX：TR/+DM/-DM 用 alpha=1/period 平滑，DX 再做同口径平滑。"""
    n = len(close)
    if n < period * 2 + 1:
        return None
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # 首行没有前值，丢弃。
    tr_s = pd.Series(tr[1:])
    plus_s = pd.Series(plus_dm[1:])
    minus_s = pd.Series(minus_dm[1:])
    atr = tr_s.ewm(alpha=1.0 / period, adjust=False).mean()
    if (atr <= 0).any() or atr.iloc[-1] <= 0:
        return None
    plus_di = plus_s.ewm(alpha=1.0 / period, adjust=False).mean() / atr * 100
    minus_di = minus_s.ewm(alpha=1.0 / period, adjust=False).mean() / atr * 100
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum * 100).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    value = float(adx.iloc[-1])
    return value if np.isfinite(value) else None


def compute_all(df, cfg):
    """计算全部因子
    Args:
        df: 单只股票的OHLCV DataFrame
        cfg: StrategyConfig实例
    Returns:
        dict: {因子名: 值} or None
    """
    if df is None or len(df) < 30:
        return None

    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    chg = (
        df["pctChg"].values.astype(float)
        if "pctChg" in df.columns
        else np.diff(c, prepend=c[0]) / c[0] * 100
    )

    f = {}

    # === 趋势类 ===
    # N日动量统一为最近N个交易日涨跌幅（复利口径，与pctChg一致）。
    momentum_5 = n_day_return(c, 5)
    momentum_20 = n_day_return(c, 20)
    if momentum_5 is not None:
        f["momentum"] = momentum_5  # 5日动量
    if momentum_20 is not None:
        f["momentum_20"] = momentum_20  # 20日动量
    if len(c) >= 5:
        f["reversal"] = (c[-1] - np.mean(c[-5:])) / max(np.mean(c[-5:]), 0.01) * 100  # 均值回复

    # === 趋势强度 ADX（标准Wilder定义） ===
    adx_value = wilder_adx(h, lo, c, period=14)
    if adx_value is not None:
        f["adx"] = adx_value

    # === 波动类 ===
    if len(chg) >= 5:
        f["volatility"] = np.std(chg[-5:])  # 5日波动率

    # === 流动性 ===
    # 注意：这是5日平均成交量（百万股），不是换手率——没有流通股本无法计算换手率。
    if len(v) >= 5:
        f["avg_volume_5d"] = np.mean(v[-5:]) / 1e6
        f["volume_ratio"] = v[-1] / max(np.mean(v[-20:]), 1) if len(v) >= 20 else 1.0  # 量比

    # === 量价关系 ===
    if len(c) >= 6 and len(v) >= 6:
        dp = np.diff(c[-6:])
        dv = np.diff(v[-6:])
        if len(dp) == len(dv) and np.std(dp) > 0:
            corr = np.corrcoef(dp, dv)[0, 1]
            f["vol_price_corr"] = max(0, corr if not np.isnan(corr) else 0)

    # === 平稳度 ===
    cd = cfg.consecutive_up_days
    if len(chg) >= cd + 1:
        gains = [v for v in chg[-cd:] if not np.isnan(v)]
        f["steadiness"] = 1.0 / (1.0 + np.std(gains)) if len(gains) >= 2 else 0.5

    # === 价格位置（20日高低区间） ===
    if len(c) >= 20:
        high20, low20 = np.max(h[-20:]), np.min(lo[-20:])
        f["price_position"] = (c[-1] - low20) / max(high20 - low20, 0.01)  # 0~1, 高=近高点

    # === RSI(14) ===
    if len(chg) >= 14:
        up = np.sum(chg[-14:][chg[-14:] > 0])
        dn = abs(np.sum(chg[-14:][chg[-14:] < 0]))
        rs = up / max(dn, 0.01)
        f["rsi"] = 100 - 100 / (1 + rs)

    # === 布林带 ===
    if len(c) >= 20:
        ma20 = np.mean(c[-20:])
        std20 = np.std(c[-20:])
        f["bb_position"] = (c[-1] - ma20) / max(std20, 0.01)

    # === MACD ===
    if len(c) >= 26:
        ema12 = pd.Series(c).ewm(span=12).mean().values
        ema26 = pd.Series(c).ewm(span=26).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9).mean().values
        f["macd_hist"] = (dif[-1] - dea[-1]) / max(c[-1], 0.01) * 100
        f["macd_golden"] = 1 if (dif[-2] <= dea[-2] and dif[-1] > dea[-1]) else 0

    # === 买卖参考 ===
    atr5 = np.mean(h[-5:] - lo[-5:])
    support = min(lo[-3:])
    resistance = max(h[-3:])
    f["buy_price"] = round(max(support, c[-1] * 0.98), 2)
    f["sell_price"] = round(min(resistance + atr5, c[-1] * 1.05), 2)
    mom = abs(f.get("momentum", 0))
    f["hold_days"] = max(3, min(14, int(10 / (1 + mom / 3))))
    f["risk"] = "高" if f.get("volatility", 0) > 5 else "中" if f.get("volatility", 0) > 3 else "低"

    return f


def compute_all_batch(df, positions, cfg):
    """在多个行号位置批量计算市场因子（等价于逐位置截断后调用compute_all）。

    数值口径（等价性红线，tests/test_perf_sprint.py 有逐值对比测试）：
    - ewm因子（ADX/MACD）在全长序列上按因果方式递推，位置p的取值与
      对df.iloc[:p+1]调用compute_all逐位一致；
    - 窗口因子在相同的尾部窗口切片上使用同一公式；
    - 键的存在性也与compute_all一致（如ADX受`atr<=0`整史门禁、vol_price_corr
      受`std(dp)>0`门禁）。
    只输出市场因子键，不含buy_price/sell_price/hold_days/risk等交易参考值
    （这些值不进入因子诊断）。位置行数<30时该位置结果为None，与compute_all一致。
    Args:
        df: 单只股票的OHLCV DataFrame（行序与compute_all看到的hist一致）
        positions: 行号列表（0起，含），每个位置p代表截断到第p行
        cfg: StrategyConfig实例或类
    Returns:
        dict: {position: 因子dict或None}
    """
    result = {}
    if df is None or len(df) == 0:
        return {p: None for p in positions}

    n = len(df)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    chg = (
        df["pctChg"].values.astype(float)
        if "pctChg" in df.columns
        else np.diff(c, prepend=c[0]) / c[0] * 100
    )

    # 全序列ewm：ewm递推因果，位置p的值 == 对前p+1行ewm的末值（逐位一致）。
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9).mean().values

    # Wilder ADX全序列：与wilder_adx完全相同的丢首行+ewm(alpha=1/14)递推。
    adx_series = None
    adx_broken = None
    if n >= 2:
        prev_close = np.roll(c, 1)
        prev_high = np.roll(h, 1)
        prev_low = np.roll(lo, 1)
        tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)))
        up_move = h - prev_high
        down_move = prev_low - lo
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_s = pd.Series(tr[1:])
        plus_s = pd.Series(plus_dm[1:])
        minus_s = pd.Series(minus_dm[1:])
        atr = tr_s.ewm(alpha=1.0 / 14, adjust=False).mean()
        # adx_broken[j]：atr[0..j]中是否出现过<=0（对应原始行2..j+2）；与
        # wilder_adx的`if (atr <= 0).any()`整史门禁逐位置等价。
        adx_broken = np.logical_or.accumulate((atr <= 0).to_numpy())
        plus_di = plus_s.ewm(alpha=1.0 / 14, adjust=False).mean() / atr * 100
        minus_di = minus_s.ewm(alpha=1.0 / 14, adjust=False).mean() / atr * 100
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / di_sum * 100).fillna(0.0)
        adx_series = dx.ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()

    for p in positions:
        if p is None or p < 0 or p >= n:
            result[p] = None
            continue
        m = p + 1
        if m < 30:
            result[p] = None
            continue
        f = {}
        momentum_5 = n_day_return(c[:m], 5)
        momentum_20 = n_day_return(c[:m], 20)
        if momentum_5 is not None:
            f["momentum"] = momentum_5
        if momentum_20 is not None:
            f["momentum_20"] = momentum_20
        if m >= 5:
            mean5 = np.mean(c[p - 4 : p + 1])
            f["reversal"] = (c[p] - mean5) / max(mean5, 0.01) * 100
        if adx_series is not None and m >= 14 * 2 + 1 and not adx_broken[p - 1]:
            adx_value = float(adx_series[p - 1])
            if np.isfinite(adx_value):
                f["adx"] = adx_value
        if m >= 5:
            f["volatility"] = np.std(chg[p - 4 : p + 1])
        if m >= 5:
            f["avg_volume_5d"] = np.mean(v[p - 4 : p + 1]) / 1e6
            if m >= 20:
                f["volume_ratio"] = v[p] / max(np.mean(v[p - 19 : p + 1]), 1)
            else:
                f["volume_ratio"] = 1.0
        if m >= 6:
            dp = np.diff(c[p - 5 : p + 1])
            dv = np.diff(v[p - 5 : p + 1])
            if np.std(dp) > 0:
                corr = np.corrcoef(dp, dv)[0, 1]
                f["vol_price_corr"] = max(0, corr if not np.isnan(corr) else 0)
        cd = cfg.consecutive_up_days
        if m >= cd + 1:
            gains = [x for x in chg[p - cd + 1 : p + 1] if not np.isnan(x)]
            f["steadiness"] = 1.0 / (1.0 + np.std(gains)) if len(gains) >= 2 else 0.5
        if m >= 20:
            high20, low20 = np.max(h[p - 19 : p + 1]), np.min(lo[p - 19 : p + 1])
            f["price_position"] = (c[p] - low20) / max(high20 - low20, 0.01)
        if m >= 14:
            window = chg[p - 13 : p + 1]
            up = np.sum(window[window > 0])
            dn = abs(np.sum(window[window < 0]))
            rs = up / max(dn, 0.01)
            f["rsi"] = 100 - 100 / (1 + rs)
        if m >= 20:
            ma20 = np.mean(c[p - 19 : p + 1])
            std20 = np.std(c[p - 19 : p + 1])
            f["bb_position"] = (c[p] - ma20) / max(std20, 0.01)
        if m >= 26:
            f["macd_hist"] = (dif[p] - dea[p]) / max(c[p], 0.01) * 100
            f["macd_golden"] = 1 if (dif[p - 1] <= dea[p - 1] and dif[p] > dea[p]) else 0
        result[p] = f
    return result


def score(f, weights=None):
    """综合因子评分
    Args:
        f: compute_all()的结果
        weights: dict of {因子名: 权重}，默认使用配置中的
    Returns:
        float: 0-10分
    """
    if f is None:
        return 0
    if weights is None:
        from strategy.config import StrategyConfig

        weights = StrategyConfig.factor_weights

    s = 0.0
    # 趋势
    s += min(max(f.get("momentum", 0), 0), 10) / 10 * weights.get("momentum", 0.25) * 10
    s += (
        min(max(f.get("momentum_20", 0), 0), 20) / 20 * weights.get("momentum", 0.25) * 10 * 0.3
    )  # 20日动量辅助
    s += (1 - min(abs(f.get("reversal", 0)) / 5, 1)) * weights.get("reversal", 0.05) * 10
    # ADX趋势强度
    adx = f.get("adx", 0)
    s += min(adx / 40, 1) * weights.get("adx", 0) * 10  # ADX>40=强趋势
    # 波动
    s += (1 - min(f.get("volatility", 5) / 10, 1)) * weights.get("volatility", 0.05) * 10
    # 流动性：5日均量（百万股）。曾误命名为turnover，已更名。
    s += min(f.get("avg_volume_5d", 0) / 50, 1) * weights.get("avg_volume_5d", 0.10) * 10
    # 量比（放量加分，但不过度）
    vr = f.get("volume_ratio", 1.0)
    s += (min(vr, 2.5) - 0.5) / 2 * weights.get("volume_ratio", 0) * 10 if vr > 0.8 else 0
    # 量价
    s += max(f.get("vol_price_corr", 0), 0) * weights.get("vol_price_corr", 0.15) * 10
    # 平稳
    s += f.get("steadiness", 0.5) * weights.get("steadiness", 0.15) * 10
    # 价格位置（中低位好，高位差——防追高）
    pp = f.get("price_position", 0.5)
    s += max(0, 1 - abs(pp - 0.35)) * weights.get("price_position", 0) * 10
    # RSI
    rsi = f.get("rsi", 50)
    s += max(0, 1 - abs(rsi - 55) / 30) * weights.get("rsi", 0.10) * 10  # 偏多RSI
    # 布林
    bb = abs(f.get("bb_position", 0))
    s += max(0, 1 - bb / 2) * weights.get("bb", 0.05) * 10
    # MACD
    s += f.get("macd_golden", 0) * weights.get("macd", 0.10) * 5
    s += max(0, f.get("macd_hist", 0)) * weights.get("macd", 0.10) * 5

    return round(s, 2)
