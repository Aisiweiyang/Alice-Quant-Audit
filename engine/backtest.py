"""回测引擎 —— 核心事件驱动循环。"""

from datetime import timedelta
import os
import sys

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cache_store import (
    build_cache_key,
    build_metadata,
    load_market_cache,
    save_market_cache,
)
from data.fetcher import get_batch_data, get_index_data, get_liquid_universe, get_stock_list
from engine.broker import Broker
from engine.execution import order_is_executable
from engine.fast_index import build_stock_indexes
from engine.portfolio import Portfolio
from strategy.config import StrategyConfig as cfg
from strategy.selection import (
    market_allows_buy,
    passes_strategy_filters,
    select_candidates as shared_select_candidates,
)


def _row_on(df, date):
    """返回指定交易日的一行行情；找不到时返回None。"""
    target = pd.Timestamp(date)
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            row = df.loc[target]
        except KeyError:
            return None
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row
    rows = df.loc[df["date"].eq(target)]
    return None if rows.empty else rows.iloc[-1]


def _build_trade_dates(data, start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = set()
    for df in data.values():
        current = pd.to_datetime(df["date"])
        dates.update(current[(current >= start) & (current <= end)].dt.normalize())
    return sorted(dates)


def _rebalance_dates(trade_dates, frequency, weekday="monday"):
    """按实际交易日生成调仓日，节假日不会让整周调仓缺失。"""
    if frequency == "daily":
        return set(trade_dates)
    if not trade_dates:
        return set()

    if frequency == "monthly":
        grouped = {}
        for date in trade_dates:
            grouped.setdefault((date.year, date.month), []).append(date)
        return {dates[0] for dates in grouped.values()}

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
    }
    target = weekday_map.get(str(weekday).lower(), 0)
    grouped = {}
    for date in trade_dates:
        iso = date.isocalendar()
        grouped.setdefault((iso.year, iso.week), []).append(date)

    result = set()
    for dates in grouped.values():
        on_or_after = [date for date in dates if date.weekday() >= target]
        result.add(on_or_after[0] if on_or_after else dates[-1])
    return result


def _passes_strategy_filters(hist, config=cfg):
    """Compatibility wrapper around the shared selection rule."""
    return passes_strategy_filters(hist, config)


def select_candidates(data, signal_date, held_codes=(), config=cfg, **kwargs):
    """Compatibility wrapper used by existing callers and tests."""
    return shared_select_candidates(data, signal_date, held_codes, config, **kwargs)


def _load_market_cache(cache_path, allow_incomplete_data=True, config=cfg):
    """Load a trusted local pickle cache and validate its schema."""
    data, metadata = load_market_cache(cache_path, config.benchmark_index)
    print(f"  读取指定缓存: {os.path.abspath(os.path.expanduser(cache_path))}")
    capabilities = (metadata or {}).get("capabilities") or {}
    quality = (metadata or {}).get("quality") or {}
    limitations = []
    if not capabilities.get("historical_trade_status", False):
        limitations.append("完整历史ST/停牌字段")
    if (config.use_pe_filter or config.use_pb_filter) and not capabilities.get(
        "historical_valuation_fields", False
    ):
        limitations.append("可用的历史PE/PB字段")
    if not quality:
        limitations.append("离线数据质量审计结果")
    elif not quality.get("formal_ready", False):
        limitations.append(f"通过质量审计的数据（失败{quality.get('failed_codes', '未知')}只）")
    if limitations and not allow_incomplete_data:
        raise ValueError(
            "缓存缺少" + "、".join(limitations) + "。正式回测已停止；"
            "确认只做兼容研究时添加 --allow-incomplete-data。"
        )
    if limitations:
        print(f"  警告: 缓存缺少{'、'.join(limitations)}，将使用兼容研究模式")
    return data


def run(
    start_date="2025-01-01",
    end_date="2026-08-01",
    verbose=True,
    cache_path=None,
    allow_incomplete_data=False,
    config=None,
    data=None,
):
    """运行回测，信号使用前一交易日数据并在下一交易日开盘成交。

    ``config`` 缺省时使用生产配置 ``strategy.config.StrategyConfig``；
    商业化入口可传入参数视图而不影响全局配置。
    ``data`` 非空时直接使用调用方提供的已校验行情（跳过缓存加载与质量门禁），
    供压力测试/滚动窗口复用同一份数据，避免大缓存反复读盘；
    数据质量门禁由调用方负责（商业管线以数据审计结论为准）。
    """
    config = config or cfg
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date不能晚于end_date")

    print(f"\n{'=' * 60}")
    print("  回测引擎启动")
    print(f"  区间: {start.date()} → {end.date()}")
    print(f"  初始资金: ¥{config.initial_capital:,}  持仓上限: {config.max_positions}只")
    print(f"{'=' * 60}\n")

    # 提前取120个自然日作为因子预热，但净值只从用户指定日期开始。
    print("[1/4] 获取股票数据...")
    if data is not None:
        benchmark = data.get(config.benchmark_index)
        if benchmark is None or benchmark.empty:
            raise ValueError(f"缓存缺少基准指数 {config.benchmark_index}")
        print(f"  使用调用方提供的已校验数据: {len(data)}个代码")
    elif cache_path:
        data = _load_market_cache(
            cache_path, allow_incomplete_data=allow_incomplete_data, config=config
        )
    else:
        codes = get_liquid_universe(top_n=config.top_volume_n)
        if codes is None:
            print("  流动性筛选失败，回退到主板全量(慢)")
            codes = get_stock_list(market="main")
        print(f"  流动性筛选后股票池: {len(codes)}只")
        warmup_start = (start - timedelta(days=120)).strftime("%Y-%m-%d")

        # 缓存键包含实际股票代码，避免两个等长股票池错误共用缓存。
        key = build_cache_key(
            codes=codes,
            start_date=warmup_start,
            end_date=end.strftime("%Y-%m-%d"),
            adjust="2",
            benchmark=config.benchmark_index,
            universe_mode="current-liquid",
        )
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_root = os.path.expanduser(
            os.environ.get(
                "ALICE_MARKET_CACHE_DIR", os.path.join(project_dir, "local_data", "cache")
            )
        )
        generated_cache = os.path.join(cache_root, f"market_{key}.pkl")
        if os.path.exists(generated_cache):
            print(f"  读取缓存 {generated_cache}")
            data, _ = load_market_cache(generated_cache, config.benchmark_index)
        else:
            data = get_batch_data(codes, warmup_start, end.strftime("%Y-%m-%d"), verbose=verbose)
            benchmark_data = get_index_data(
                config.benchmark_index, warmup_start, end.strftime("%Y-%m-%d")
            )
            if benchmark_data is None or benchmark_data.empty:
                raise ValueError(f"无法获取基准指数 {config.benchmark_index}")
            data[config.benchmark_index] = benchmark_data
            try:
                metadata = build_metadata(
                    data,
                    benchmark=config.benchmark_index,
                    adjust="2",
                    universe_mode="current-liquid",
                    source="baostock+akshare",
                    warnings=["当前股票池用于历史区间时存在幸存者偏差。"],
                )
                save_market_cache(generated_cache, data, metadata)
                print(f"  缓存已保存: {generated_cache}")
            except OSError as exc:
                print(f"  缓存写入跳过: {exc}")
    if not data:
        print("  无可用股票数据")
        return None, None

    benchmark = data.get(config.benchmark_index)
    if benchmark is None or benchmark.empty:
        raise ValueError(f"缓存缺少基准指数 {config.benchmark_index}")

    benchmark = benchmark[benchmark["date"].between(start - timedelta(days=120), end)][
        ["date", "close"]
    ].copy()
    stock_data = {code: frame for code, frame in data.items() if code != config.benchmark_index}
    print(f"  获取成功: {len(stock_data)}只股票 + 1个基准指数")
    if not stock_data:
        print("  无可用股票数据")
        return None, None

    trade_dates = _build_trade_dates(stock_data, start, end)
    if not trade_dates:
        print("  指定区间内没有交易日")
        return None, None

    # 按股票建立日期快速索引；个别帧不满足前提时该股票
    # 回退原逐行查找，行为保持等价。
    indexes = build_stock_indexes(stock_data)
    indexed_items = [(code, index) for code, index in indexes.items() if index is not None]
    legacy_items = [(code, stock_data[code]) for code, index in indexes.items() if index is None]

    broker = Broker(config)
    portfolio = Portfolio(
        config.initial_capital, broker, config.max_positions, config.max_position_pct
    )
    rebalance_dates = _rebalance_dates(trade_dates, config.rebalance_freq, config.rebalance_day)
    print(f"  交易日: {len(trade_dates)}天")
    print(f"  调仓日: {len(rebalance_dates)}次\n")

    # 大盘择时：预计算基准均线
    bench_ma = {}
    bench_ma50 = {}
    if config.use_market_timing and benchmark is not None and not benchmark.empty:
        bm = benchmark.copy()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm.sort_values("date")
        bm["ma"] = bm["close"].rolling(config.ma_period).mean()
        bm["ma50"] = bm["close"].rolling(50).mean()
        for _, row in bm.iterrows():
            d = pd.Timestamp(row["date"]).normalize()
            if pd.notna(row["ma"]):
                bench_ma[d] = float(row["ma"])
            if pd.notna(row["ma50"]):
                bench_ma50[d] = float(row["ma50"])
        print(f"  大盘择时: 收盘<MA{config.ma_period}时空仓, MA{config.ma_period}<MA50清仓\n")

    print("[2/4] 运行回测...")
    previous_closes = {}
    last_prices = {}

    for i, date in enumerate(trade_dates):
        opens = {}
        open_rows = {}
        closes = {}
        date_value = date.value
        # 快速路径：字典定位行号 + 预取列数组，替代逐股票布尔掩码/索引查找。
        for code, index in indexed_items:
            pos = index.position_of.get(date_value)
            if pos is None:
                continue
            open_price = float(index.column("open")[pos])
            close_price = float(index.column("close")[pos])
            if open_price > 0:
                opens[code] = open_price
                open_rows[code] = index.execution_row(pos)
            if close_price > 0:
                closes[code] = close_price
        # 回退路径：索引不可用的股票按原实现逐行查找。
        for code, df in legacy_items:
            row = _row_on(df, date)
            if row is None:
                continue
            open_price = float(row["open"])
            close_price = float(row["close"])
            if open_price > 0:
                opens[code] = open_price
                open_rows[code] = row
            if close_price > 0:
                closes[code] = close_price
        last_prices.update(closes)

        # 级联清仓：MA20 < MA50 → 卖出所有持仓
        if config.use_market_timing and config.ma_cascade:
            prev_date = trade_dates[i - 1] if i > 0 else date
            if prev_date in bench_ma and prev_date in bench_ma50:
                if bench_ma[prev_date] < bench_ma50[prev_date]:
                    for code in list(portfolio.holdings):
                        executable, _ = order_is_executable(code, open_rows.get(code, {}), "SELL")
                        if code in opens and executable:
                            portfolio.sell(code, opens[code], date.strftime("%Y-%m-%d"))

        # 前一日收盘产生止盈/止损信号，今日开盘执行，避免同收盘信号同价成交。
        for code in list(portfolio.holdings):
            if code not in previous_closes or code not in opens:
                continue
            reference = previous_closes[code]
            buy_price = portfolio.holdings[code]["avg_cost"]
            if reference >= buy_price * (1 + config.take_profit_pct) or reference <= buy_price * (
                1 - config.stop_loss_pct
            ):
                executable, _ = order_is_executable(code, open_rows[code], "SELL")
                if executable:
                    portfolio.sell(code, opens[code], date.strftime("%Y-%m-%d"))

        # 前一交易日收盘选股，当前调仓日开盘买入；signal_lag_days 默认1，
        # 压力测试可设为2模拟"信号延迟一天才买入"。
        signal_lag = max(1, int(getattr(config, "signal_lag_days", 1)))
        if date in rebalance_dates and i >= signal_lag:
            signal_date = trade_dates[i - signal_lag]
            can_buy, market_reason = market_allows_buy(benchmark, signal_date, config)
            decision = {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "execution_date": date.strftime("%Y-%m-%d"),
                "market_allowed": bool(can_buy),
                "market_reason": market_reason,
                "positions_before": len(portfolio.holdings),
                "available_slots": max(0, config.max_positions - len(portfolio.holdings)),
                "candidate_count": 0,
                "top_candidate_score": None,
                "selected": [],
            }
            if can_buy:
                ranked = select_candidates(
                    stock_data,
                    signal_date,
                    portfolio.holdings.keys(),
                    config,
                    indexes=indexes,
                )
                slots = config.max_positions - len(portfolio.holdings)
                decision["candidate_count"] = len(ranked)
                decision["top_candidate_score"] = float(ranked[0][1]) if ranked else None
                portfolio_value = portfolio.total_value(last_prices)
                for code, factor_score in ranked[:slots]:
                    if code in opens:
                        executable, _ = order_is_executable(code, open_rows[code], "BUY")
                        if executable:
                            bought = portfolio.buy(
                                code,
                                opens[code],
                                date.strftime("%Y-%m-%d"),
                                portfolio_value=portfolio_value,
                            )
                            if bought:
                                decision["selected"].append(
                                    {
                                        "code": code,
                                        "score": float(factor_score),
                                    }
                                )
            decision["positions_after"] = len(portfolio.holdings)
            portfolio.rebalance_decisions.append(decision)

        portfolio.record(date.strftime("%Y-%m-%d"), last_prices)
        previous_closes = closes

        if verbose and i % 50 == 0:
            total = portfolio.total_value(last_prices)
            pnl_pct = (total / config.initial_capital - 1) * 100
            print(
                f"    {date.date()} | 净值: {total:,.0f} ({pnl_pct:+.1f}%) "
                f"| 持仓: {len(portfolio.holdings)}只"
            )

    print("\n[3/4] 回测结束，按末日收盘价清仓...")
    final_date = trade_dates[-1].strftime("%Y-%m-%d")
    final_value = trade_dates[-1].value
    final_rows = {}
    for code, index in indexed_items:
        pos = index.position_of.get(final_value)
        if pos is not None:
            final_rows[code] = index.execution_row(pos)
    for code, df in legacy_items:
        if (row := _row_on(df, trade_dates[-1])) is not None:
            final_rows[code] = row
    final_closes = {
        code: float(row["close"]) for code, row in final_rows.items() if float(row["close"]) > 0
    }
    # 期末清仓同样受停牌/跌停约束：卖不出的持仓按最后已知价估值并记录原因。
    portfolio.close_all(final_date, final_closes, rows=final_rows)
    # 替换末日盘中记录，避免同一天出现两个净值点。
    if portfolio.equity_curve and portfolio.equity_curve[-1]["date"] == final_date:
        portfolio.equity_curve.pop()
    portfolio.record(final_date, last_prices)

    print("[4/4] 计算绩效指标...")
    metrics = calculate_metrics(portfolio, benchmark)
    print_metrics(metrics)
    return portfolio, metrics


def _max_consecutive(returns: pd.Series) -> tuple[int, int]:
    """最大连续亏损/盈利天数（口径同empyrical.max_consecutive_losses/wins）。"""
    signs = np.sign(returns.to_numpy(dtype=float))
    max_loss = max_win = current = 0
    previous = 0
    for sign in signs:
        if sign == 0:
            current = 0
            previous = 0
            continue
        if sign == previous:
            current += 1
        else:
            current = 1
            previous = sign
        if sign < 0:
            max_loss = max(max_loss, current)
        else:
            max_win = max(max_win, current)
    return max_loss, max_win


def _bootstrap_sharpe_ci(returns: pd.Series, cycles: int = 1000, block: int = 10, seed: int = 42):
    """Sharpe的95%自助置信区间：环形移动块自助（块长10个交易日，固定种子）。

    收益存在自相关/波动聚集时，i.i.d.重采样会低估区间宽度；按块重采样
    保留短期依赖结构（arch的StationaryBootstrap/PyBroker的块自助同思路，
    纯numpy向量化实现）。返回(百分位2.5, 百分位97.5)；样本<20返回None。
    """
    values = returns.to_numpy(dtype=float)
    n = len(values)
    if n < 20:
        return None
    blocks = -(-n // block)  # ceil
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(cycles, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = values[indices].reshape(cycles, blocks * block)[:, :n]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    valid = stds > 0
    if not valid.any():
        return None
    sharpe = means[valid] / stds[valid] * np.sqrt(252)
    low, high = np.percentile(sharpe, [2.5, 97.5])
    return float(low), float(high)


def _dated_returns(eq: pd.DataFrame) -> pd.Series:
    """净值序列 → 以交易日为索引的日收益序列（供与基准逐日对齐）。"""
    returns = eq["total"].pct_change().dropna()
    dates = pd.DatetimeIndex(pd.to_datetime(eq["date"]))
    return pd.Series(returns.to_numpy(dtype=float), index=dates[1:])


def _benchmark_daily_returns(benchmark_df: pd.DataFrame) -> pd.Series:
    """基准收盘价 → 以日期为索引的基准日收益序列；同日重复记录保留最后一条。"""
    bm = benchmark_df.copy()
    bm["date"] = pd.to_datetime(bm["date"])
    bm = bm.sort_values("date").drop_duplicates(subset="date", keep="last")
    close = pd.Series(
        pd.to_numeric(bm["close"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(bm["date"]),
    )
    return close.pct_change().dropna()


def _finite_or_none(value, digits):
    """四舍五入并保证有限：NaN/Inf一律降级为None，绝不让无效数字进报告。"""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _top_drawdown_periods(totals, dates, top=5):
    """前大回撤区间明细（pyfolio performance.gen_drawdown_table口径，top=5）。

    totals为按日升序的净值绝对值序列（首值为初始资金，等价于相对净值1.0）。
    峰值日=下行开始前最后一个处于历史高点的交易日；谷底日=区间内最深点；
    恢复日=净值回到前高的首个交易日；期末仍未恢复时recovery_date与
    recovery_days如实为None（pyfolio表中Recovery date为NaN的同一语义）。
    返回按深度从深到浅排序的明细列表；最深一项的depth_pct与
    max_drawdown_pct同源同号（均为负的百分比），有等价性测试保护。
    """
    equity = np.asarray(totals, dtype=float)
    if equity.size == 0:
        return []
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    running_max = np.maximum.accumulate(equity)
    underwater = equity / running_max - 1.0  # ≤0，0表示处于历史高点
    periods = []
    episode_start = None
    for i in range(equity.size + 1):
        at_peak = i == equity.size or underwater[i] >= 0
        if not at_peak:
            if episode_start is None:
                episode_start = i
            continue
        if episode_start is not None:
            valley_idx = episode_start + int(np.argmin(underwater[episode_start:i]))
            peak_idx = max(episode_start - 1, 0)
            recovered = i < equity.size
            periods.append(
                {
                    "peak_date": dates[peak_idx].strftime("%Y-%m-%d"),
                    "valley_date": dates[valley_idx].strftime("%Y-%m-%d"),
                    "recovery_date": dates[i].strftime("%Y-%m-%d") if recovered else None,
                    "depth_pct": round(float(underwater[valley_idx]) * 100, 2),
                    "decline_days": int(valley_idx - peak_idx),
                    "recovery_days": int(i - valley_idx) if recovered else None,
                }
            )
            episode_start = None
    periods.sort(key=lambda item: item["depth_pct"])
    return periods[:top]


def _trade_round_trips(trades) -> list[dict]:
    """把交易记录按FIFO配成"买入→卖出"回合（backtesting.py交易统计口径）。

    本引擎同一代码同时最多持有一仓，因此按代码内第k次BUY配第k次SELL即可，
    与通用FIFO等价。每回合给出：入场/出场日期、净盈亏pnl（含双边摩擦）、
    成本基数cost（含买入佣金）、单笔收益率return_pct=pnl/cost、
    持仓天数holding_days（自然日）。
    """
    per_code: dict[str, list] = {}
    for trade in trades:
        per_code.setdefault(str(trade.get("code")), []).append(trade)
    trips = []
    for items in per_code.values():
        buys = [t for t in items if t.get("action") == "BUY"]
        sells = [t for t in items if t.get("action") == "SELL"]
        for buy, sell in zip(buys, sells):
            basis = float(buy.get("cost") or 0)
            pnl = float(sell.get("pnl") or 0)
            trips.append(
                {
                    "entry_date": str(buy.get("date")),
                    "exit_date": str(sell.get("date")),
                    "pnl": pnl,
                    "cost_basis": basis,
                    "return_pct": (pnl / basis * 100) if basis > 0 else None,
                    "holding_days": max(
                        (pd.Timestamp(str(sell["date"])) - pd.Timestamp(str(buy["date"]))).days, 0
                    ),
                }
            )
    return trips


def _ulcer_index(equity_values: np.ndarray) -> float:
    """Ulcer Index（Peter G. Martin公开公式，quantstats/vectorbt生态常用）。

    UI = sqrt( mean( dd_i^2 ) )，dd_i为距历史峰值的回撤深度（百分比正值）。
    与最大回撤只看最深一点不同，UI同时惩罚回撤的深度与持续时间。
    返回百分数口径的UI；无回撤时为0。
    """
    equity = np.asarray(equity_values, dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (running_max - equity) / running_max * 100  # 深度取正
    return float(np.sqrt(np.mean(drawdown_pct**2)))


def _omega_ratio(returns, threshold: float = 0.0):
    """Omega比率（empyrical stats.omega_ratio口径，已逐行核对源码）。

    ω = Σ(r>L) / |Σ(r<L)|：日收益超出阈值L的部分之和，除以低于阈值部分的
    损失绝对值之和。阈值取日收益0%（empyrical对required_return=0按
    (1+0)^(1/252)-1=0折算到日频，等价）。分母为0（无下方损失）时与
    empyrical一致返回None。样本≥20的门槛由调用方把关（empyrical只挡<2）。
    """
    values = np.asarray(returns, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        return None
    excess = values - threshold
    gains = float(excess[excess > 0.0].sum())
    losses = float(-excess[excess < 0.0].sum())
    if losses <= 0.0:
        return None
    return gains / losses


def _mc_drawdown_stats(
    returns: pd.Series, cycles: int = 1000, block: int = 10, seed: int = 42
) -> dict | None:
    """蒙特卡洛重构回撤/终值分布（环形移动块自助，与Sharpe CI同机制）。

    对历史日收益做1000次块重排（块长10个交易日，保留短期依赖结构，固定种子
    可复现），每次重排视为"同样收益特征的一次等价实现"，计算该路径的最大
    回撤与终值收益，给出分布分位数。已实现回撤只是其中一条路径——该统计
    用于回答"换个收益顺序，回撤会有多深"，是历史样本的统计重构，
    不是对未来回撤的预测。样本<60时返回None（分位数不稳定）。
    """
    values = returns.to_numpy(dtype=float)
    n = len(values)
    if n < 60:
        return None
    blocks = -(-n // block)  # ceil
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(cycles, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = values[indices].reshape(cycles, blocks * block)[:, :n]
    equity = np.cumprod(1.0 + samples, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown = (equity - running_max) / running_max  # ≤0
    max_drawdowns = drawdown.min(axis=1) * 100  # 负的百分比
    terminal_returns = (equity[:, -1] - 1.0) * 100
    return {
        "mc_max_drawdown_pct_p50": round(float(np.percentile(max_drawdowns, 50)), 2),
        "mc_max_drawdown_pct_p95": round(float(np.percentile(max_drawdowns, 95)), 2),
        "mc_terminal_return_p5_pct": round(float(np.percentile(terminal_returns, 5)), 2),
        "mc_paths": int(cycles),
        "mc_observations": int(n),
    }


def _tmc_drawdown_stats(
    round_trips, initial_cash: float, cycles: int = 1000, seed: int = 42
) -> dict | None:
    """交易级蒙特卡洛（tmc_*）：已完成回合的盈亏额随机重排后叠加到资金曲线。

    与日收益块自助(_mc_drawdown_stats)并列的第二口径：块自助重排日收益、
    保留时序依赖结构；交易重排以"已完成回合"为最小结算单元，贴合本引擎
    按调仓日离散换仓的实现。对每笔净盈亏（含双边摩擦，金额口径）做1000次
    随机重排（置换，非自助，固定种子可复现），每次从initial_cash出发依次
    累加，统计最大回撤与终值分布。

    口径说明：盈亏额（而非收益率）重排是刻意的——加法不变性保证任意重排
    终值都与已实现总收益精确一致，回撤分布单独回答"同样的盈亏换个实现
    顺序会有多深"；若按收益率全仓复利，会丢掉仓位管理与现金拖累，终值
    严重失真（真实数据审读实例：203笔重排给出+1628%终值，而已实现仅
    +45%），该口径已废弃。某路径中途资金≤0时该路径回撤按-100%（资金先
    归零）计；终值≤0（盈亏合计亏光本金）时整体返回None。已完成回合<30
    时返回None（分位数不稳定，如实缺失，不编造）。
    """
    values = np.asarray(
        [t.get("pnl") for t in round_trips if t.get("pnl") is not None],
        dtype=float,
    )
    n = len(values)
    cash = float(initial_cash) if initial_cash is not None else 0.0
    if n < 30 or not np.isfinite(cash) or cash <= 0:
        return None
    if not np.isfinite(values).all():
        return None
    rng = np.random.default_rng(seed)
    # 每行独立生成一个随机置换（argsort随机键=均匀随机排列）。
    indices = np.argsort(rng.random((cycles, n)), axis=1)
    equity = cash + np.concatenate(
        [np.zeros((cycles, 1)), np.cumsum(values[indices], axis=1)], axis=1
    )
    if (equity[:, -1] <= 0).any():
        # 盈亏合计亏光本金：终值分位数无意义，如实"数据不足"。
        return None
    running_max = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = (equity - running_max) / running_max  # ≤0
    drawdown = np.where(equity <= 0, -1.0, drawdown)  # 穿零路径按全亏计
    max_drawdowns = np.nanmin(np.where(np.isfinite(drawdown), drawdown, 0.0), axis=1) * 100
    terminal_returns = (equity[:, -1] / cash - 1.0) * 100
    return {
        "tmc_max_drawdown_pct_p50": round(float(np.percentile(max_drawdowns, 50)), 2),
        "tmc_max_drawdown_pct_p95": round(float(np.percentile(max_drawdowns, 95)), 2),
        "tmc_terminal_return_p5_pct": round(float(np.percentile(terminal_returns, 5)), 2),
        "tmc_round_trips": int(n),
        "tmc_paths": int(cycles),
    }


def calculate_metrics(portfolio, benchmark_df=None):
    """计算回测绩效指标。"""
    eq = pd.DataFrame(portfolio.equity_curve)
    if eq.empty:
        return {}

    eq["date"] = pd.to_datetime(eq["date"])
    returns = eq["total"].pct_change().dropna()
    total_return = (eq["total"].iloc[-1] / portfolio.initial_cash - 1) * 100
    days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    cagr = ((eq["total"].iloc[-1] / portfolio.initial_cash) ** (365 / max(days, 1)) - 1) * 100

    peak = eq["total"].cummax()
    drawdown = (eq["total"] - peak) / peak * 100
    max_dd = float(drawdown.min())
    daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annual_vol = daily_std * np.sqrt(252) * 100
    sharpe = (float(returns.mean()) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    # 短样本防误导：观测不足时仍输出数值（兼容旧消费方），但显式标记不可信。
    min_sharpe_observations = 20
    sharpe_observations = int(len(returns))
    sharpe_reliable = sharpe_observations >= min_sharpe_observations
    reliability_warnings = []
    if not sharpe_reliable:
        reliability_warnings.append(
            f"日收益观测仅{sharpe_observations}条(<{min_sharpe_observations})，"
            "Sharpe与年化波动不具统计意义"
        )

    # 风险调整补充指标（empyrical口径）：Sortino只罚下行波动；Calmar=年化收益/最大回撤。
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (float(returns.mean()) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    calmar = None
    if max_dd < 0:
        calmar = float((cagr / 100) / abs(max_dd / 100))
    max_consec_losses, max_consec_wins = _max_consecutive(returns)

    # Sharpe的95%块自助置信区间：区间包含零说明正收益不具统计显著性（研究诚实度提示）。
    # 注意：该提示走sharpe_ci_*专用字段（不含入reliability_warnings，其"样本充足
    # 即无Sharpe警告"的历史契约有测试保护），由报告与建议层消费。
    sharpe_ci = _bootstrap_sharpe_ci(returns)
    sharpe_ci_low = round(sharpe_ci[0], 2) if sharpe_ci else None
    sharpe_ci_high = round(sharpe_ci[1], 2) if sharpe_ci else None
    sharpe_ci_includes_zero = bool(sharpe_ci[0] <= 0.0 <= sharpe_ci[1]) if sharpe_ci else None

    # 收益质量指标（增量字段，不改旧字段）。口径出处：
    # - alpha/beta：pyfolio/performance.py::alpha_beta（对基准日收益做最小二乘
    #   CAPM回归；本文取回归斜率为beta、截距按(1+α)^252-1复利年化为alpha；
    #   empyrical的alpha用"年化收益-β×基准年化收益"口径，两者在此注明差异）；
    # - information_ratio：quantstats/stats.py::information_ratio
    #   （超额日收益均值/标准差×√252，标准差为ddof=1）；
    # - up/down_capture：pyfolio capture口径（基准涨/跌日的策略收益均值/基准收益均值）；
    # - tail_ratio：empyrical/statistics.py::tail_ratio（日收益P95与P5绝对值之比，
    #   衡量左右尾不对称；本文对两侧都取绝对值，与提示词口径一致）；
    # - recovery_factor：区间收益/最大回撤深度（总收益为回撤深度的多少倍）。
    # 基准缺失或对齐观测不足时本组字段全部如实None（含纯策略口径的tail_ratio与
    # recovery_factor：按提示词口径整组与基准绑定，保持语义一致便于审计）。
    min_quality_observations = 20  # 与sharpe_reliable同门槛
    quality_metrics = {
        "alpha_annualized_pct": None,
        "beta": None,
        "information_ratio": None,
        "tail_ratio": None,
        "up_capture_pct": None,
        "down_capture_pct": None,
        "recovery_factor": None,
    }
    if benchmark_df is not None and not benchmark_df.empty:
        bench_daily = _benchmark_daily_returns(benchmark_df)
        strat_daily = _dated_returns(eq)
        common = strat_daily.index.intersection(bench_daily.index)
        if len(common) >= min_quality_observations:
            strat_v = strat_daily.loc[common].to_numpy(dtype=float)
            bench_v = bench_daily.loc[common].to_numpy(dtype=float)
            if np.isfinite(strat_v).all() and np.isfinite(bench_v).all():
                bench_std = float(bench_v.std(ddof=1))
                if bench_std > 0:
                    slope, intercept = np.polyfit(bench_v, strat_v, 1)
                    quality_metrics["beta"] = _finite_or_none(slope, 3)
                    quality_metrics["alpha_annualized_pct"] = _finite_or_none(
                        ((1.0 + float(intercept)) ** 252 - 1) * 100, 2
                    )
                excess = strat_v - bench_v
                excess_std = float(excess.std(ddof=1))
                if excess_std > 0:
                    quality_metrics["information_ratio"] = _finite_or_none(
                        float(excess.mean()) / excess_std * float(np.sqrt(252)), 3
                    )
                up_mask = bench_v > 0
                down_mask = bench_v < 0
                if up_mask.any() and float(bench_v[up_mask].mean()) != 0:
                    quality_metrics["up_capture_pct"] = _finite_or_none(
                        float(strat_v[up_mask].mean()) / float(bench_v[up_mask].mean()) * 100, 2
                    )
                if down_mask.any() and float(bench_v[down_mask].mean()) != 0:
                    quality_metrics["down_capture_pct"] = _finite_or_none(
                        float(strat_v[down_mask].mean()) / float(bench_v[down_mask].mean()) * 100, 2
                    )
                p95 = float(np.percentile(strat_v, 95))
                p05 = float(np.percentile(strat_v, 5))
                if abs(p05) > 0:
                    quality_metrics["tail_ratio"] = _finite_or_none(abs(p95) / abs(p05), 3)
                if max_dd < 0:
                    quality_metrics["recovery_factor"] = _finite_or_none(
                        (total_return / 100) / abs(max_dd / 100), 3
                    )

    # 分布与尾部指标（empyrical/scipy口径；样本≥20才计算，否则如实None）：
    # - var_95_pct / cvar_95_pct：empyrical stats.value_at_risk /
    #   conditional_value_at_risk（cutoff=5%；保留收益原始符号，负值=损失幅度）；
    # - skewness / excess_kurtosis：scipy.stats.skew / kurtosis 的默认口径
    #   （总体矩；峰度为超额峰度，正态=0），empyrical SIMPLE_STAT_FUNCS 同此。
    distribution = {
        "var_95_pct": None,
        "cvar_95_pct": None,
        "skewness": None,
        "excess_kurtosis": None,
    }
    returns_values = returns.to_numpy(dtype=float)
    if len(returns_values) >= 20 and np.isfinite(returns_values).all():
        cutoff_index = int((len(returns_values) - 1) * 0.05)
        partitioned = np.partition(returns_values, cutoff_index)
        distribution["var_95_pct"] = _finite_or_none(partitioned[cutoff_index] * 100, 2)
        distribution["cvar_95_pct"] = _finite_or_none(
            partitioned[: cutoff_index + 1].mean() * 100, 2
        )
        centered = returns_values - returns_values.mean()
        m2 = float(np.mean(centered**2))
        if m2 > 0:
            distribution["skewness"] = _finite_or_none(float(np.mean(centered**3)) / m2**1.5, 3)
            distribution["excess_kurtosis"] = _finite_or_none(
                float(np.mean(centered**4)) / (m2**2) - 3.0, 3
            )

    # Ulcer Index与Martin ratio（Peter G. Martin公开公式）：
    # UI同时惩罚回撤深度与持续时间；Martin=年化收益/UI（无风险利率取0）。
    ulcer = _ulcer_index(eq["total"].to_numpy(dtype=float))
    martin_ratio = None
    if ulcer > 0:
        martin_ratio = _finite_or_none((cagr / 100) / (ulcer / 100), 3)

    # 交易级统计（backtesting.py _stats.py口径，MIT）：
    # 期望值=单笔收益率算术均值；最好/最差=单笔收益率极值；
    # 持仓天数=回合自然日；SQN=sqrt(N)×单笔盈亏均值/标准差（Van K. Tharp
    # 公式，建议≥30笔才有意义，不足时如实None）。
    trade_stats = {
        "expectancy_pct": None,
        "best_trade_pct": None,
        "worst_trade_pct": None,
        "avg_trade_duration_days": None,
        "max_trade_duration_days": None,
        "sqn": None,
    }
    trips = _trade_round_trips(getattr(portfolio, "trades", None) or [])
    completed_returns = [t["return_pct"] for t in trips if t["return_pct"] is not None]
    if completed_returns:
        trade_stats["expectancy_pct"] = _finite_or_none(float(np.mean(completed_returns)), 3)
        trade_stats["best_trade_pct"] = _finite_or_none(float(np.max(completed_returns)), 2)
        trade_stats["worst_trade_pct"] = _finite_or_none(float(np.min(completed_returns)), 2)
        durations = [t["holding_days"] for t in trips]
        trade_stats["avg_trade_duration_days"] = _finite_or_none(float(np.mean(durations)), 1)
        trade_stats["max_trade_duration_days"] = int(max(durations))
    pnls = [t["pnl"] for t in trips]
    if len(pnls) >= 30:
        pnl_std = float(np.std(pnls, ddof=1))
        if pnl_std > 0:
            trade_stats["sqn"] = _finite_or_none(
                float(np.sqrt(len(pnls)) * np.mean(pnls) / pnl_std), 2
            )

    # 蒙特卡洛重构（块自助重排，与Sharpe CI同机制）：已实现回撤只是等价路径
    # 中的一条；p95回答"换个收益顺序，回撤会有多深"。统计重构，不是预测。
    mc_stats = _mc_drawdown_stats(returns) or {}

    # 交易级蒙特卡洛重排（tmc_*，第二口径）：以已完成回合净盈亏（金额）为
    # 结算单元重排，终值恒等于已实现总收益；回合<30时如实None。
    tmc_stats = _tmc_drawdown_stats(trips, portfolio.initial_cash) or {}

    # Omega比率（empyrical口径，阈值=日收益0%）；样本≥20才计算，否则如实None。
    omega_ratio = _omega_ratio(returns_values) if len(returns_values) >= 20 else None

    # 最大回撤明细（pyfolio gen_drawdown_table口径，前5大；与max_drawdown_pct同源）。
    drawdown_periods = _top_drawdown_periods(eq["total"].to_numpy(dtype=float), eq["date"])

    bench_return = 0.0
    if benchmark_df is not None and not benchmark_df.empty:
        bm = benchmark_df.copy()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm[bm["date"].isin(eq["date"])].sort_values("date")
        if len(bm) > 1:
            bench_return = (bm["close"].iloc[-1] / bm["close"].iloc[0] - 1) * 100

    summary = portfolio.summary() or {}
    gross_profit = float(summary.get("gross_profit", 0))
    gross_loss = float(summary.get("gross_loss", 0))
    if gross_loss < 0:
        profit_factor = round(gross_profit / abs(gross_loss), 2)
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    total_trades = int(summary.get("total_trades", 0))
    if total_trades < 5:
        reliability_warnings.append(f"完整卖出交易仅{total_trades}笔(<5)，胜率与盈亏比不稳定")
    unsold_final = getattr(portfolio, "final_unsold", None) or {}
    if unsold_final:
        reliability_warnings.append(
            f"期末有{len(unsold_final)}个持仓无法按末日收盘真实卖出，按最后已知价估值"
        )

    return {
        "total_return_pct": round(float(total_return), 2),
        "cagr_pct": round(float(cagr), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "annual_volatility_pct": round(float(annual_vol), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "sharpe_observations": sharpe_observations,
        "sharpe_reliable": bool(sharpe_reliable),
        "sharpe_ci_low": sharpe_ci_low,
        "sharpe_ci_high": sharpe_ci_high,
        "sharpe_ci_includes_zero": sharpe_ci_includes_zero,
        "sortino_ratio": round(float(sortino), 2),
        "calmar_ratio": round(float(calmar), 2) if calmar is not None else None,
        "max_consecutive_losses": int(max_consec_losses),
        "max_consecutive_wins": int(max_consec_wins),
        "alpha_annualized_pct": quality_metrics["alpha_annualized_pct"],
        "beta": quality_metrics["beta"],
        "information_ratio": quality_metrics["information_ratio"],
        "tail_ratio": quality_metrics["tail_ratio"],
        "up_capture_pct": quality_metrics["up_capture_pct"],
        "down_capture_pct": quality_metrics["down_capture_pct"],
        "recovery_factor": quality_metrics["recovery_factor"],
        "drawdown_periods": drawdown_periods,
        "var_95_pct": distribution["var_95_pct"],
        "cvar_95_pct": distribution["cvar_95_pct"],
        "skewness": distribution["skewness"],
        "excess_kurtosis": distribution["excess_kurtosis"],
        "ulcer_index_pct": _finite_or_none(ulcer, 3),
        "martin_ratio": martin_ratio,
        "expectancy_pct": trade_stats["expectancy_pct"],
        "best_trade_pct": trade_stats["best_trade_pct"],
        "worst_trade_pct": trade_stats["worst_trade_pct"],
        "avg_trade_duration_days": trade_stats["avg_trade_duration_days"],
        "max_trade_duration_days": trade_stats["max_trade_duration_days"],
        "sqn": trade_stats["sqn"],
        "mc_max_drawdown_pct_p50": mc_stats.get("mc_max_drawdown_pct_p50"),
        "mc_max_drawdown_pct_p95": mc_stats.get("mc_max_drawdown_pct_p95"),
        "mc_terminal_return_p5_pct": mc_stats.get("mc_terminal_return_p5_pct"),
        "mc_paths": mc_stats.get("mc_paths"),
        "tmc_max_drawdown_pct_p50": tmc_stats.get("tmc_max_drawdown_pct_p50"),
        "tmc_max_drawdown_pct_p95": tmc_stats.get("tmc_max_drawdown_pct_p95"),
        "tmc_terminal_return_p5_pct": tmc_stats.get("tmc_terminal_return_p5_pct"),
        "tmc_round_trips": tmc_stats.get("tmc_round_trips"),
        "tmc_paths": tmc_stats.get("tmc_paths"),
        "omega_ratio": _finite_or_none(omega_ratio, 3),
        "benchmark_return_pct": round(float(bench_return), 2),
        "excess_return_pct": round(float(total_return - bench_return), 2),
        "total_trades": total_trades,
        "win_rate_pct": round(float(summary.get("win_rate", 0)), 1),
        "avg_win": round(float(summary.get("avg_win", 0)), 2),
        "avg_loss": round(float(summary.get("avg_loss", 0)), 2),
        "profit_factor": profit_factor,
        "unsold_final_positions": len(unsold_final),
        "reliability_warnings": reliability_warnings,
    }


def print_metrics(metrics):
    """打印绩效指标。"""
    if not metrics:
        print("  没有足够数据计算指标")
        return
    profit_factor = "∞" if metrics["profit_factor"] is None else f"{metrics['profit_factor']:.2f}"
    print(f"\n{'=' * 50}")
    print("  回测绩效报告")
    print(f"{'=' * 50}")
    print(f"  总收益:     {metrics['total_return_pct']:+.2f}%")
    print(f"  年化收益:   {metrics['cagr_pct']:+.2f}%")
    print(f"  最大回撤:   {metrics['max_drawdown_pct']:.2f}%")
    print(f"  年化波动:   {metrics['annual_volatility_pct']:.2f}%")
    print(f"  夏普比率:   {metrics['sharpe_ratio']:.2f}")
    print(f"  基准收益:   {metrics['benchmark_return_pct']:+.2f}%")
    print(f"  超额收益:   {metrics['excess_return_pct']:+.2f}%")
    print(f"  交易次数:   {metrics['total_trades']}")
    print(f"  胜率:       {metrics['win_rate_pct']:.1f}%")
    print(f"  均盈利:     ¥{metrics['avg_win']:,.2f}")
    print(f"  均亏损:     ¥{metrics['avg_loss']:,.2f}")
    print(f"  盈亏比:     {profit_factor}")
    print(f"{'=' * 50}")
