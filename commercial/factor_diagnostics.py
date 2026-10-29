"""因子诊断（商业版）：覆盖率、RankIC、ICIR、正向比例、分组收益、衰减、
翻转、相关矩阵、重复因子、分年与分市场状态表现、波动率暴露。

只使用信号日当日及之前的数据；数据不足的项显式返回"数据不足"，不伪造结论。
"""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd

from strategy.config import StrategyConfig
from strategy.factors import compute_all_batch

DIAGNOSTIC_FACTORS = [
    "momentum",
    "momentum_20",
    "reversal",
    "adx",
    "volatility",
    "avg_volume_5d",
    "volume_ratio",
    "vol_price_corr",
    "steadiness",
    "price_position",
    "rsi",
    "bb_position",
    "macd_hist",
]
HORIZONS = (1, 5, 10, 20)
MIN_HISTORY = 60
MIN_SIGNALS = 6
DUPLICATE_CORR_THRESHOLD = 0.90
MAX_UNIVERSE = 800


class FactorDataInsufficient(RuntimeError):
    """因子诊断所需数据不足。"""


def _signal_dates(data, start, end, benchmark, max_signals=40):
    dates = set()
    for frame in data.values():
        d = pd.to_datetime(frame["date"], errors="coerce").dropna()
        dates.update(d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].dt.normalize())
    calendar = sorted(dates)
    if not calendar:
        return []
    # 每周第一个交易日作信号日，避免重算过于频繁。
    grouped = {}
    for date in calendar:
        iso = date.isocalendar()
        grouped.setdefault((iso.year, iso.week), []).append(date)
    signals = [group[0] for group in grouped.values()]
    return signals[:max_signals] if len(signals) > max_signals else signals


def _market_states(data, benchmark, signals, ma_period=20):
    frame = data.get(benchmark)
    states = {}
    if frame is None or frame.empty:
        return states
    series = frame[["date", "close"]].copy().sort_values("date")
    series["ma"] = series["close"].rolling(ma_period).mean()
    state_by_date = {
        pd.Timestamp(row.date).normalize(): bool(row.close >= row.ma)
        for row in series.itertuples()
        if pd.notna(row.ma)
    }
    return {date: state_by_date.get(date) for date in signals}


def _stock_signal_positions(frame, signals, min_history=MIN_HISTORY):
    """每只股票一次定位全部信号日：返回 {signal_date: 行号或None}。

    行号为该信号日及其之前最后一行的行号（与 hist = frame.loc[date <= 信号日]
    的截断口径一致）；历史不足min_history行的信号日不参与。
    """
    positions = {}
    if frame is None or frame.empty:
        return {date: None for date in signals}
    dates = pd.DatetimeIndex(frame["date"])
    for date in signals:
        pos = int(dates.searchsorted(pd.Timestamp(date), side="right") - 1)
        positions[date] = pos if pos >= min_history - 1 else None
    return positions


def _precompute_factor_values(data, stocks, signals):
    """按股票一次批量预计算全部信号日因子（替代逐信号compute_all全历史重算）。

    返回 (factor_values, base_closes)：
    - factor_values: {code: {signal_date: 因子dict或None}}，因子值与逐信号
      compute_all(hist)逐值一致（≤1e-6，有等价性测试）；
    - base_closes: {code: {signal_date: 信号日最后已知收盘价float}}，
      对应原实现 hist["close"].iloc[-1]。
    """
    factor_values = {}
    base_closes = {}
    for code in stocks:
        frame = data[code]
        positions = _stock_signal_positions(frame, signals)
        valid = {p for p in positions.values() if p is not None}
        computed = compute_all_batch(frame, sorted(valid), StrategyConfig) if valid else {}
        factor_values[code] = {
            date: computed.get(pos) if pos is not None else None for date, pos in positions.items()
        }
        closes = frame["close"].to_numpy()
        base_closes[code] = {
            date: float(closes[pos])
            for date, pos in positions.items()
            if pos is not None and pos < len(closes)
        }
    return factor_values, base_closes


def _cross_section_rows(signal_date, stocks, horizon_closes, factor_values, base_closes):
    rows = []
    for code in stocks:
        factors = (factor_values.get(code) or {}).get(signal_date)
        if not factors:
            continue
        row = {"code": code}
        for name in DIAGNOSTIC_FACTORS:
            row[name] = factors.get(name, np.nan)
        last_close = (base_closes.get(code) or {}).get(signal_date)
        for horizon, future_close in horizon_closes[code]:
            row[f"fwd_{horizon}d"] = (
                future_close / last_close - 1.0
                if future_close and last_close and last_close > 0
                else np.nan
            )
        rows.append(row)
    return rows


def _forward_map(frame, signal_date, dates=None, closes=None):
    """返回 {horizon: 信号日+N交易日收盘价}；不足N天则None。

    dates/closes 可传入按股票预计算的 DatetimeIndex 与收盘价数组以避免
    逐信号重复构建（结果与每次重建完全一致）。
    """
    if dates is None:
        dates = pd.DatetimeIndex(frame["date"]).normalize()
    if closes is None:
        closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
    position = int(dates.searchsorted(pd.Timestamp(signal_date), side="right") - 1)
    if position < 0 or dates[position] != pd.Timestamp(signal_date).normalize():
        return None
    result = {}
    for horizon in HORIZONS:
        target = position + horizon
        result[horizon] = float(closes[target]) if target < len(closes) else None
    return result


def _rank_ic(values, returns):
    valid = np.isfinite(values) & np.isfinite(returns)
    if valid.sum() < 8:
        return np.nan, 0.0
    v = pd.Series(values[valid]).rank()
    r = pd.Series(returns[valid]).rank()
    if float(v.std()) == 0 or float(r.std()) == 0:
        return np.nan, valid.mean()
    ic = float(np.corrcoef(v, r)[0, 1])
    return (ic if np.isfinite(ic) else np.nan), float(valid.mean())


def _quintile_means(values, returns, buckets: int = 5):
    """按因子值均分五组的未来收益均值（fraction，Q1最低→Q5最高）。

    口径参考alphalens的分位数收益分析（quantile returns）；样本<10返回None。
    """
    valid = np.isfinite(values) & np.isfinite(returns)
    if valid.sum() < 10:
        return None
    ordered = np.argsort(values[valid], kind="mergesort")
    sample = returns[valid][ordered]
    return [
        float(np.mean(chunk)) if len(chunk) else np.nan for chunk in np.array_split(sample, buckets)
    ]


def _monotonic_label(means) -> str:
    """分位数收益单调性标签：递增/递减/非单调/数据不足。"""
    if means is None or any(not np.isfinite(m) for m in means):
        return "数据不足"
    diffs = np.diff(means)
    if all(d > 0 for d in diffs):
        return "递增"
    if all(d < 0 for d in diffs):
        return "递减"
    return "非单调"


def _quintile_spread(values, returns):
    valid = np.isfinite(values) & np.isfinite(returns)
    if valid.sum() < 10:
        return np.nan
    ordered = np.argsort(values[valid], kind="mergesort")
    sample = returns[valid][ordered]
    size = max(len(sample) // 5, 1)
    return float(np.mean(sample[-size:]) - np.mean(sample[:size]))


def compute_factor_diagnostics(
    data,
    start,
    end,
    benchmark="sh.000300",
    forward_days=5,
    verbose=False,
):
    """主入口：返回 (逐因子汇总DataFrame, 元数据dict)。

    数据不足以支撑诊断时抛出 FactorDataInsufficient。
    """
    all_stocks = [code for code in data if code != benchmark]
    signals = _signal_dates(data, start, end, benchmark)
    if len(all_stocks) < 5 or len(signals) < MIN_SIGNALS:
        raise FactorDataInsufficient(
            f"信号日仅{len(signals)}个(要求≥{MIN_SIGNALS})或股票仅{len(all_stocks)}只(要求≥5)，"
            "因子诊断需要更长历史"
        )
    # 大股票池等距抽样控制计算量；抽样方式确定性的，可复现。
    sampling = {"applied": False, "requested": len(all_stocks), "used": len(all_stocks)}
    stocks = all_stocks
    if len(all_stocks) > MAX_UNIVERSE:
        ordered = sorted(all_stocks)
        step = len(ordered) / MAX_UNIVERSE
        stocks = sorted({ordered[int(index * step)] for index in range(MAX_UNIVERSE)})
        sampling = {
            "applied": True,
            "requested": len(all_stocks),
            "used": len(stocks),
        }

    # 预取每只股票在各信号日的未来收盘价（每股票一次构建日期/收盘索引）。
    forward_by_code = {}
    for code in stocks:
        frame = data[code]
        dates = pd.DatetimeIndex(frame["date"]).normalize()
        closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy()
        per_signal = {}
        for date in signals:
            mapping = _forward_map(frame, date, dates=dates, closes=closes)
            if mapping:
                per_signal[date] = sorted(mapping.items())
        forward_by_code[code] = per_signal

    factor_values, base_closes = _precompute_factor_values(data, stocks, signals)

    records = []
    states = _market_states(data, benchmark, signals)
    for date in signals:
        horizon_closes = {code: forward_by_code.get(code, {}).get(date, []) for code in stocks}
        usable = [code for code in stocks if forward_by_code.get(code, {}).get(date)]
        if len(usable) < 5:
            continue
        for row in _cross_section_rows(date, usable, horizon_closes, factor_values, base_closes):
            row["signal_date"] = date
            row["market_state"] = "bull" if states.get(date) else "bear"
            records.append(row)
    if not records:
        raise FactorDataInsufficient("有效信号日截面不足，无法计算因子诊断")

    panel = pd.DataFrame(records)
    primary = f"fwd_{forward_days}d"
    summaries = []
    ic_series_by_factor = {}
    ics_by_factor = {}
    spread_by_factor = {}
    pooled = {}
    # 波动率暴露的逐股票缓存（与因子无关，整个因子循环共享）。
    vol_frame_cache = {}
    # 秩自相关用的按信号日截面索引（code索引的因子值Series）。
    signal_dates_sorted = sorted(panel["signal_date"].unique())
    date_groups = {date: group.set_index("code") for date, group in panel.groupby("signal_date")}
    for name in DIAGNOSTIC_FACTORS:
        values = panel[name].astype(float)
        coverage = float(np.isfinite(values).mean())
        if coverage < 0.3:
            summaries.append(
                {
                    "feature": name,
                    "status": "数据不足",
                    "detail": f"因子值覆盖率仅{coverage:.0%}，不足以计算诊断",
                }
            )
            continue
        ics, spreads = [], []
        spread_points = []
        quintile_acc = []
        for date, group in panel.groupby("signal_date"):
            ic, _ = _rank_ic(
                group[name].to_numpy(dtype=float), group[primary].to_numpy(dtype=float)
            )
            if np.isfinite(ic):
                ics.append({"signal_date": date, "rank_ic": ic})
            spread = _quintile_spread(
                group[name].to_numpy(dtype=float), group[primary].to_numpy(dtype=float)
            )
            if np.isfinite(spread):
                spreads.append(spread)
                spread_points.append((date, spread))
            means = _quintile_means(
                group[name].to_numpy(dtype=float), group[primary].to_numpy(dtype=float)
            )
            if means is not None:
                quintile_acc.append(means)
        if len(ics) < MIN_SIGNALS // 2:
            summaries.append(
                {
                    "feature": name,
                    "status": "数据不足",
                    "detail": "有效信号日截面过少，RankIC不可靠",
                }
            )
            continue
        ic_series = pd.DataFrame(ics)["rank_ic"]
        ic_series_by_factor[name] = ic_series
        ics_by_factor[name] = ics
        spread_by_factor[name] = spread_points
        pooled[name] = values
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
        entry = {
            "feature": name,
            "status": "ok",
            "coverage_pct": round(coverage * 100, 1),
            "dates": int(len(ic_series)),
            "mean_rank_ic": round(float(ic_series.mean()), 4),
            "rank_icir": round(float(ic_series.mean()) / ic_std, 3) if ic_std > 0 else 0.0,
            "positive_ic_rate_pct": round(float((ic_series > 0).mean() * 100), 1),
            "quintile_spread_pct": round(float(np.mean(spreads)) * 100, 3) if spreads else np.nan,
        }
        # 分位数收益单调性（alphalens口径）：Q1→Q5均值逐组比较。
        quintile_mean = np.mean(np.array(quintile_acc), axis=0) if quintile_acc else None
        if quintile_mean is not None:
            for index, value in enumerate(quintile_mean, start=1):
                entry[f"q{index}_mean_fwd_pct"] = (
                    round(float(value) * 100, 3) if np.isfinite(value) else np.nan
                )
        entry["quantile_monotonic"] = _monotonic_label(
            quintile_mean.tolist() if quintile_mean is not None else None
        )
        # 因子秩自相关（alphalens口径的factor rank autocorrelation）：
        # 相邻信号日截面因子秩相关中位数；越高说明因子换手越慢、排名越稳定。
        autocorrs = []
        for prev_date, cur_date in zip(signal_dates_sorted[:-1], signal_dates_sorted[1:]):
            prev = date_groups[prev_date].get(name)
            cur = date_groups[cur_date].get(name)
            if prev is None or cur is None:
                continue
            common = prev.index.intersection(cur.index)
            if len(common) < 8:
                continue
            corr = float(prev[common].rank().corr(cur[common].rank()))
            if np.isfinite(corr):
                autocorrs.append(corr)
        entry["rank_autocorr"] = round(float(np.median(autocorrs)), 3) if autocorrs else np.nan
        # 衰减：不同持有期IC均值。
        for horizon in HORIZONS:
            column = f"fwd_{horizon}d"
            if column not in panel:
                entry[f"ic_{horizon}d"] = np.nan
                continue
            horizon_ics = []
            for _, group in panel.groupby("signal_date"):
                ic, _ = _rank_ic(
                    group[name].to_numpy(dtype=float), group[column].to_numpy(dtype=float)
                )
                if np.isfinite(ic):
                    horizon_ics.append(ic)
            entry[f"ic_{horizon}d"] = (
                round(float(np.mean(horizon_ics)), 4) if horizon_ics else np.nan
            )
        # 分年表现与翻转。
        panel["year"] = pd.to_datetime(panel["signal_date"]).dt.year
        yearly = []
        for year, group in panel.groupby("year"):
            year_ics = []
            for _, sub in group.groupby("signal_date"):
                ic, _ = _rank_ic(
                    sub[name].to_numpy(dtype=float), sub[primary].to_numpy(dtype=float)
                )
                if np.isfinite(ic):
                    year_ics.append(ic)
            if year_ics:
                yearly.append((int(year), float(np.mean(year_ics))))
        if len(yearly) >= 2:
            for year, mean_ic in yearly:
                entry[f"ic_{year}"] = round(mean_ic, 4)
            first_sign = np.sign(yearly[0][1])
            last_sign = np.sign(yearly[-1][1])
            entry["direction_flip"] = bool(
                first_sign != 0
                and last_sign != 0
                and first_sign != last_sign
                and abs(yearly[0][1]) >= 0.02
                and abs(yearly[-1][1]) >= 0.02
            )
        else:
            entry["direction_flip"] = None
            entry["yearly_detail"] = "数据不足（区间不足两年）"
        # 分市场状态表现。
        for state_label in ("bull", "bear"):
            state_ics = []
            for date, group in panel[panel["market_state"] == state_label].groupby("signal_date"):
                ic, _ = _rank_ic(
                    group[name].to_numpy(dtype=float), group[primary].to_numpy(dtype=float)
                )
                if np.isfinite(ic):
                    state_ics.append(ic)
            entry[f"ic_{state_label}"] = (
                round(float(np.mean(state_ics)), 4) if state_ics else np.nan
            )
        # 波动率暴露：因子与已实现20日波动率的秩相关。
        vol_exposures = []
        for code, group in panel.groupby("code"):
            frame = data[code]
            hist = frame[frame["date"] <= group["signal_date"].max()]
            if len(hist) < 25:
                continue
            if code not in vol_frame_cache:
                vol_frame_cache[code] = (
                    pd.DatetimeIndex(frame["date"]).normalize(),
                    pd.to_numeric(frame["pctChg"], errors="coerce").rolling(20).std(),
                )
            dates, realized_vol = vol_frame_cache[code]
            vol_at_signal = []
            for _, row in group.iterrows():
                pos = dates.searchsorted(pd.Timestamp(row["signal_date"]), side="right") - 1
                if 0 <= pos < len(realized_vol) and np.isfinite(realized_vol.iloc[pos]):
                    vol_at_signal.append((row.get(name, np.nan), float(realized_vol.iloc[pos])))
            if len(vol_at_signal) >= 8:
                pairs = np.array(vol_at_signal, dtype=float)
                ic, _ = _rank_ic(pairs[:, 0], pairs[:, 1])
                if np.isfinite(ic):
                    vol_exposures.append(ic)
        entry["volatility_exposure"] = (
            round(float(np.mean(vol_exposures)), 3) if vol_exposures else np.nan
        )
        summaries.append(entry)

    summary = pd.DataFrame(summaries)

    # 逐日IC序列（因子×信号日），供报告绘制IC热力图；日期序列化为ISO字符串。
    ic_by_date = {
        name: [[str(item["signal_date"])[:10], round(float(item["rank_ic"]), 4)] for item in items]
        for name, items in ics_by_factor.items()
    }

    # 逐信号日多空spread序列（Q5组均值收益 − Q1组均值收益，alphalens口径的
    # long-short factor spread；×100换算为百分比），供报告绘制多空累计曲线。
    long_short_spread_by_date = {
        name: [[str(date)[:10], round(float(spread) * 100, 4)] for date, spread in points]
        for name, points in spread_by_factor.items()
    }

    # 因子相关矩阵与重复因子。
    corr_matrix = pd.DataFrame()
    duplicates = []
    ok_factors = [s["feature"] for s in summaries if s.get("status") == "ok"]
    if len(ok_factors) >= 2:
        corr_matrix = panel[ok_factors].corr(method="spearman").round(3)
        for i, left in enumerate(ok_factors):
            for right in ok_factors[i + 1 :]:
                value = corr_matrix.loc[left, right]
                if np.isfinite(value) and abs(value) >= DUPLICATE_CORR_THRESHOLD:
                    duplicates.append(
                        {
                            "pair": f"{left} / {right}",
                            "spearman": round(float(value), 3),
                        }
                    )
    if sampling["applied"]:
        sampling["note"] = (
            f"股票池{sampling['requested']}只超过上限{MAX_UNIVERSE}，"
            f"按代码排序等距抽样{sampling['used']}只参与因子诊断；结论为抽样估计"
        )
    metadata = {
        "sampling": sampling,
        "signal_rule": "每周第一个交易日为信号日，仅使用信号日及之前数据",
        "primary_forward_days": forward_days,
        "horizons_days": list(HORIZONS),
        "signal_count": int(panel["signal_date"].nunique()),
        "stock_count": int(panel["code"].nunique()),
        "ic_by_date": ic_by_date,
        "long_short_spread_by_date": long_short_spread_by_date,
        "long_short_note": (
            "多空spread=信号日Q5组平均未来收益−Q1组平均未来收益；累计曲线为"
            "逐信号日简单累加（未复利、未计交易成本，相邻5日窗口部分重叠）"
        ),
        "duplicate_threshold": DUPLICATE_CORR_THRESHOLD,
        "duplicate_pairs": duplicates,
        "exposure_note": "行业与市值暴露需要行业分类/股本字段，当前数据不足，未提供。",
        "disclaimer": "因子诊断描述历史截面统计特征，不构成投资建议，不预测收益。",
    }
    return summary, corr_matrix, metadata


def render_factor_section_html(
    summary: pd.DataFrame, corr_matrix: pd.DataFrame, metadata: dict
) -> str:
    import html

    esc = html.escape
    if summary is None or summary.empty:
        return "<p class='muted'>因子诊断数据不足，无法输出。</p>"
    rows = []
    for _, row in summary.iterrows():
        if row.get("status") == "数据不足":
            rows.append(
                f"<tr><td>{esc(str(row['feature']))}</td>"
                f"<td colspan='8'>{esc(str(row.get('detail', '数据不足')))}</td></tr>"
            )
            continue
        flip = row.get("direction_flip")
        flip_text = "是" if flip else ("数据不足" if flip is None else "否")
        autocorr = row.get("rank_autocorr")
        autocorr_text = f"{autocorr}" if autocorr is not None and np.isfinite(autocorr) else "—"
        rows.append(
            f"<tr><td>{esc(str(row['feature']))}</td>"
            f"<td>{row.get('coverage_pct', '—')}%</td>"
            f"<td>{row.get('mean_rank_ic', '—')}</td>"
            f"<td>{row.get('rank_icir', '—')}</td>"
            f"<td>{row.get('positive_ic_rate_pct', '—')}%</td>"
            f"<td>{row.get('ic_5d', '—')} / {row.get('ic_20d', '—')}</td>"
            f"<td>{esc(str(row.get('quantile_monotonic', '—')))}</td>"
            f"<td>{autocorr_text}</td>"
            f"<td>{flip_text}</td></tr>"
        )
    corr_html = ""
    if corr_matrix is not None and not corr_matrix.empty:
        header = "".join(f"<th>{esc(str(c))}</th>" for c in corr_matrix.columns)
        body = "".join(
            "<tr><th>"
            + esc(str(idx))
            + "</th>"
            + "".join(f"<td>{'' if not np.isfinite(v) else v}</td>" for v in row_values)
            + "</tr>"
            for idx, row_values in zip(corr_matrix.index, corr_matrix.to_numpy())
        )
        corr_html = (
            f"<p class='muted'>因子秩相关矩阵（|ρ|≥{metadata.get('duplicate_threshold')}视为高度重复）：</p>"
            f"<div style='overflow-x:auto'><table><thead><tr><th></th>{header}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
    dup = metadata.get("duplicate_pairs") or []
    dup_text = "、".join(item["pair"] for item in dup) if dup else "未发现高度重复因子"
    sampling = (metadata or {}).get("sampling") or {}
    sampling_badge = ""
    if sampling.get("applied"):
        sampling_badge = (
            "<div class='notice'><strong>抽样说明</strong>："
            f"{esc(str(sampling.get('note', '')))}；"
            "本页因子诊断结论为等距抽样下的抽样估计，不代表全市场截面。</div>"
        )
    return f"""
{sampling_badge}
<table><thead><tr><th>因子</th><th>覆盖率</th><th>RankIC</th><th>ICIR</th>
<th>IC正值比例</th><th>IC衰减(5d/20d)</th><th>分位单调(Q1→Q5)</th>
<th>秩自相关</th><th>方向翻转</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="muted">{esc(metadata.get("signal_rule", ""))}；信号数 {metadata.get("signal_count")}。
分位单调性为五等分因子组的未来收益均值排序（Q1最低组→Q5最高组，alphalens口径）；
秩自相关为相邻信号日因子秩相关的中位数，越高代表排名越稳定、隐含换手越低。
{esc(dup_text)}。{esc(metadata.get("exposure_note", ""))}</p>
{corr_html}
<p class="muted">{esc(metadata.get("disclaimer", ""))}</p>"""
