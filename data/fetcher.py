"""
数据获取模块 —— 统一的A股历史行情接口
使用 Baostock；网络可达性、接口可用性和来源数据质量需要使用者另行核实。
"""

try:
    import baostock as bs
except ImportError:  # Offline backtests only need the local cache.
    bs = None
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import socket
import time

CACHE_DIR = os.path.expanduser(os.environ.get("QUANT_CACHE_DIR", "~/.quant_cache"))
BAOSTOCK_SOCKET_TIMEOUT = float(os.environ.get("BAOSTOCK_SOCKET_TIMEOUT", "45"))
BAOSTOCK_RECONNECT_EVERY = max(1, int(os.environ.get("BAOSTOCK_RECONNECT_EVERY", "5")))
BAOSTOCK_RECONNECT_MAX = max(
    BAOSTOCK_RECONNECT_EVERY,
    int(os.environ.get("BAOSTOCK_RECONNECT_MAX", "10")),
)
BAOSTOCK_WORKERS = max(1, int(os.environ.get("BAOSTOCK_WORKERS", "1")))
BAOSTOCK_MIN_WORKERS = min(
    BAOSTOCK_WORKERS,
    max(1, int(os.environ.get("BAOSTOCK_MIN_WORKERS", "2"))),
)
BAOSTOCK_INITIAL_WORKERS = min(
    BAOSTOCK_WORKERS,
    max(BAOSTOCK_MIN_WORKERS, int(os.environ.get("BAOSTOCK_INITIAL_WORKERS", "3"))),
)
BAOSTOCK_ADAPTIVE = os.environ.get("BAOSTOCK_ADAPTIVE", "1") != "0"

_BAOSTOCK_EFFECTIVE_WORKERS = BAOSTOCK_INITIAL_WORKERS
_BAOSTOCK_HEALTHY_BATCHES = 0

DAILY_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,pctChg,turn,peTTM,pbMRQ,tradestatus,isST"
)
DAILY_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "pctChg",
    "turn",
    "pe",
    "pb",
    "tradestatus",
    "isST",
]
NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "pctChg",
    "turn",
    "pe",
    "pb",
    "tradestatus",
    "isST",
]


def _daily_frame(rows):
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _login():
    if bs is None:
        raise RuntimeError(
            "baostock is not installed. Install it for online downloads, "
            "or run the backtest with --cache."
        )
    if BAOSTOCK_SOCKET_TIMEOUT > 0:
        socket.setdefaulttimeout(BAOSTOCK_SOCKET_TIMEOUT)
    result = bs.login()
    if getattr(result, "error_code", "0") != "0":
        raise RuntimeError(f"Baostock登录失败: {result.error_msg}")


def _logout():
    if bs is not None:
        bs.logout()


def _login_with_retry(attempts=3):
    """Login without allowing a transient socket/decoding error to kill a worker."""
    attempts = max(1, int(attempts))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            _login()
            return
        except Exception as exc:
            last_error = exc
            try:
                _logout()
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(min(8.0, 1.5**attempt))
    raise RuntimeError(f"Baostock登录重试{attempts}次仍失败: {last_error}") from last_error


# ==================== 基础数据 ====================


def get_stock_list(market="main"):
    """获取股票列表
    Args:
        market: 'all'=全部, 'main'=主板(60/00), 'hs300'=沪深300
    Returns:
        list of str: Baostock格式代码 ['sh.600519', 'sz.000001', ...]
    """
    _login()
    try:
        if market == "hs300":
            result = bs.query_hs300_stocks()
            if getattr(result, "error_code", "0") != "0":
                raise RuntimeError(f"沪深300成分股获取失败: {result.error_msg}")
            codes = set()
            while result.next():
                row = result.get_row_data()
                # query_hs300_stocks 返回 ['updateDate', 'code', 'code_name']
                if row:
                    codes.add(row[1] if len(row) >= 2 else row[0])
            if not codes:
                raise RuntimeError("沪深300成分股列表为空")
            return sorted(codes)

        result = bs.query_stock_basic()
        codes = []
        while result.next():
            row = result.get_row_data()
            code, name = row[0], row[1]
            tail = code.split(".")[-1]
            # 排除B股、退市
            if tail.startswith("900") or tail.startswith("200"):
                continue
            if "退" in name or "ST" in name:
                continue
            if market == "main":
                # 沪深主板 + 中小板
                if tail.startswith("60") or (
                    tail.startswith("00")
                    and not tail.startswith("300")
                    and not tail.startswith("301")
                    and not tail.startswith("002")
                    and not tail.startswith("003")
                ):
                    pass
                elif tail.startswith("002") or tail.startswith("003"):
                    pass  # 中小板也算主板
                else:
                    continue
            codes.append(code)
        return codes
    finally:
        _logout()


def get_security_master():
    """Return all securities, including IPO/out dates and delisted records."""
    _login()
    try:
        rs = bs.query_stock_basic()
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(f"证券主数据获取失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        fields = list(rs.fields)
    finally:
        _logout()
    return pd.DataFrame(rows, columns=fields)


def get_liquid_universe(top_n=500):
    """用AKShare实时行情按成交额筛选流动性最好的N只主板股票。
    一次API调用，不触发反爬。返回Baostock格式代码列表。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        # 列: 代码, 名称, 最新价, 涨跌幅, ..., 成交额, ...
        df = df.rename(columns={"代码": "code", "名称": "name", "成交额": "amount"})
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # 过滤：只要主板（6开头 或 0/2/3开头但不是创业板/科创板）
        def _is_main(code):
            tail = str(code)[-6:] if len(str(code)) >= 6 else str(code)
            if tail.startswith("6"):
                return True
            if tail.startswith("00") and not tail.startswith("300") and not tail.startswith("301"):
                return True
            if tail.startswith("002") or tail.startswith("003"):
                return True
            return False

        df = df[df["code"].apply(_is_main)]
        # 排除ST和退市
        df = df[~df["name"].str.contains("ST|退", na=False)]
        # 按成交额降序取top N
        df = df.sort_values("amount", ascending=False).head(top_n)
        # 转为Baostock格式: sh.600519 / sz.000001
        codes = []
        for _, row in df.iterrows():
            code = str(row["code"])
            if code.startswith("6"):
                codes.append(f"sh.{code}")
            else:
                codes.append(f"sz.{code}")
        return codes
    except Exception as e:
        print(f"  AKShare实时行情获取失败: {e}")
        return None


def get_daily_data(code, start_date, end_date, adjust="2"):
    """获取单只股票日线数据（前复权）
    Args:
        code: Baostock格式 'sh.600519'
        start_date: '2025-01-01'
        end_date: '2026-08-01'
        adjust: '1'=后复权 '2'=前复权 '3'=不复权
    Returns:
        DataFrame or None
    """
    _login()
    rs = bs.query_history_k_data_plus(code, DAILY_FIELDS, start_date, end_date, "d", adjust)
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    _logout()
    return _daily_frame(data)


def _get_batch_data_serial(
    codes,
    start_date,
    end_date,
    adjust="2",
    verbose=True,
    retries=3,
    _return_stats=False,
):
    """在当前进程串行下载，并定期重建BaoStock连接。"""
    codes = list(codes)
    if not codes:
        empty_stats = {"requested": 0, "retry_errors": 0, "failed_codes": 0}
        return ({}, empty_stats) if _return_stats else {}
    retries = max(1, int(retries))
    results = {}
    total = len(codes)
    retry_errors = 0
    failed_codes = 0
    try:
        _login_with_retry(retries)
    except Exception as exc:
        if verbose:
            print(f"    当前分片登录失败，留待断点续传: {exc}", flush=True)
        stats = {
            "requested": total,
            "retry_errors": retries,
            "failed_codes": total,
        }
        return (results, stats) if _return_stats else results
    reconnect_every = BAOSTOCK_RECONNECT_EVERY
    codes_since_reconnect = 0
    stable_codes = 0
    try:
        for i, code in enumerate(codes):
            if verbose:
                print(f"    {i + 1}/{total} {code} 请求中...", flush=True)
            last_error = None
            had_retry = False
            for attempt in range(1, retries + 1):
                try:
                    rs = bs.query_history_k_data_plus(
                        code, DAILY_FIELDS, start_date, end_date, "d", adjust
                    )
                    if getattr(rs, "error_code", "0") != "0":
                        raise RuntimeError(f"{code}: {rs.error_msg}")
                    data = []
                    while rs.next():
                        data.append(rs.get_row_data())
                    df = _daily_frame(data)
                    if df is not None:
                        results[code] = df
                    last_error = None
                    break
                except Exception as exc:
                    retry_errors += 1
                    had_retry = True
                    last_error = exc
                    if attempt >= retries:
                        break
                    try:
                        _logout()
                    except Exception:
                        pass
                    time.sleep(min(8.0, 1.5**attempt))
                    try:
                        _login_with_retry(retries)
                    except Exception as login_exc:
                        last_error = login_exc
                        break
            if last_error is not None:
                failed_codes += 1
                if verbose:
                    print(f"    {code} 失败({retries}次): {last_error}", flush=True)
            elif verbose and ((i + 1) % 5 == 0 or i + 1 == total):
                print(f"    {i + 1}/{total} 已完成", flush=True)

            if last_error is not None or had_retry:
                reconnect_every = BAOSTOCK_RECONNECT_EVERY
                codes_since_reconnect = 0
                stable_codes = 0
            else:
                codes_since_reconnect += 1
                stable_codes += 1
                if reconnect_every < BAOSTOCK_RECONNECT_MAX and stable_codes >= reconnect_every * 2:
                    reconnect_every += 1
                    stable_codes = 0
            should_reconnect = last_error is not None or (
                not had_retry and codes_since_reconnect >= reconnect_every
            )
            if should_reconnect and i + 1 < total:
                try:
                    _logout()
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    _login_with_retry(retries)
                except Exception as exc:
                    remaining = total - (i + 1)
                    failed_codes += remaining
                    retry_errors += 1
                    if verbose:
                        print(
                            f"    当前分片重连失败，剩余{remaining}只留待断点续传: {exc}",
                            flush=True,
                        )
                    break
                codes_since_reconnect = 0
    finally:
        try:
            _logout()
        except Exception:
            pass
    stats = {
        "requested": total,
        "retry_errors": retry_errors,
        "failed_codes": failed_codes,
    }
    return (results, stats) if _return_stats else results


def _split_worker_codes(codes, workers):
    """Split codes deterministically without duplicates or empty workers."""
    values = list(codes)
    count = min(max(1, int(workers)), len(values)) if values else 0
    return [values[offset::count] for offset in range(count)] if count else []


def _batch_worker(payload):
    return _get_batch_data_serial(*payload, _return_stats=True)


def _adapt_worker_count(
    current,
    healthy_batches,
    *,
    requested,
    retry_errors,
    failed_codes,
    min_workers,
    max_workers,
):
    """Choose the next batch's process count from recent provider health."""
    current = min(max_workers, max(min_workers, int(current)))
    requested = max(1, int(requested))
    error_ratio = int(retry_errors) / requested
    if failed_codes or error_ratio >= 0.10:
        return max(min_workers, current - 1), 0
    if retry_errors:
        return current, 0
    healthy_batches += 1
    if healthy_batches >= 2 and current < max_workers:
        return current + 1, 0
    return current, healthy_batches


def _record_batch_health(stats):
    global _BAOSTOCK_EFFECTIVE_WORKERS, _BAOSTOCK_HEALTHY_BATCHES
    if not BAOSTOCK_ADAPTIVE or BAOSTOCK_WORKERS <= 1:
        return
    previous = _BAOSTOCK_EFFECTIVE_WORKERS
    _BAOSTOCK_EFFECTIVE_WORKERS, _BAOSTOCK_HEALTHY_BATCHES = _adapt_worker_count(
        previous,
        _BAOSTOCK_HEALTHY_BATCHES,
        requested=stats["requested"],
        retry_errors=stats["retry_errors"],
        failed_codes=stats["failed_codes"],
        min_workers=BAOSTOCK_MIN_WORKERS,
        max_workers=BAOSTOCK_WORKERS,
    )
    if previous != _BAOSTOCK_EFFECTIVE_WORKERS:
        print(
            f"    BaoStock并发自适应: {previous} -> "
            f"{_BAOSTOCK_EFFECTIVE_WORKERS} "
            f"(瞬时错误{stats['retry_errors']}，失败{stats['failed_codes']})",
            flush=True,
        )


def get_batch_data(codes, start_date, end_date, adjust="2", verbose=True, retries=3):
    """批量获取多只股票日线数据。

    BaoStock使用模块级socket上下文，不能在线程间共享。并行模式使用独立
    Python进程，每个进程维护自己的登录和socket，主进程只负责合并结果。

    Returns:
        dict: {code: DataFrame}
    """
    groups = _split_worker_codes(codes, _BAOSTOCK_EFFECTIVE_WORKERS)
    if len(groups) <= 1:
        result, stats = _get_batch_data_serial(
            codes,
            start_date,
            end_date,
            adjust,
            verbose,
            retries,
            _return_stats=True,
        )
        _record_batch_health(stats)
        return result

    print(
        f"    BaoStock并行下载: {len(groups)}个独立进程，"
        f"分片大小 {[len(group) for group in groups]}",
        flush=True,
    )
    payloads = [(group, start_date, end_date, adjust, verbose, retries) for group in groups]
    combined = {}
    stats = {"requested": 0, "retry_errors": 0, "failed_codes": 0}
    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = {
            executor.submit(_batch_worker, payload): group
            for payload, group in zip(payloads, groups)
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                worker_result, worker_stats = future.result()
            except Exception as exc:
                print(
                    f"    BaoStock子进程异常已隔离，{len(group)}只留待断点续传: {exc}",
                    flush=True,
                )
                worker_result = {}
                worker_stats = {
                    "requested": len(group),
                    "retry_errors": 1,
                    "failed_codes": len(group),
                }
            combined.update(worker_result)
            for key in stats:
                stats[key] += worker_stats[key]
    _record_batch_health(stats)
    return combined


def get_index_data(code, start_date, end_date):
    """获取指数数据（沪深300/中证500等）
    Args:
        code: 'sh.000300'=沪深300, 'sh.000905'=中证500
    """
    return get_daily_data(code, start_date, end_date, adjust="3")


# ==================== 财务数据 ====================


def get_financial_data(code, start_date, end_date):
    """获取财务数据（季报/年报披露日期的ROE、营收增速等）
    注意：Baostock的财务数据接口数据有限，更多财务数据后续可接Tushare
    """
    _login()
    bs.query_growth_data(code, year=int(start_date[:4]), quarter=1)
    # 简化版：返回盈利能力数据
    data = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for q in range(1, 5):
            rs2 = bs.query_profit_data(code, year=year, quarter=q)
            while rs2.next():
                row = rs2.get_row_data()
                if len(row) >= 8:
                    data.append(
                        {
                            "year": year,
                            "quarter": q,
                            "roe": float(row[3]) if row[3] else np.nan,
                            "eps": float(row[1]) if row[1] else np.nan,
                        }
                    )
    _logout()
    return pd.DataFrame(data) if data else None


# ==================== 缓存 ====================


def get_cached_data(code, start_date, end_date):
    """带缓存的单股票数据获取"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{code}_{start_date}_{end_date}"
    # pickle由pandas原生支持，不要求额外安装pyarrow。
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            return pd.read_pickle(cache_file)
        except (OSError, ValueError, EOFError):
            pass
    df = get_daily_data(code, start_date, end_date)
    if df is not None:
        df.to_pickle(cache_file)
    return df
