"""Market-data providers used by the version-3 downloader.

The provider boundary keeps network-specific behaviour out of the cache and
backtest code.  Every provider returns the same Alice daily-data columns.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from data import fetcher as baostock_fetcher


ALICE_COLUMNS = [
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

AKSHARE_WARNING = (
    "AKShare免费行情不提供可靠的历史ST、停牌、PE和PB字段；该缓存只适合数据源验证、"
    "小范围研究或近期补数，不应直接作为正式全市场历史回测结论。"
)


def _number(frame: pd.DataFrame, column: str):
    frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _normalise_akshare(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    clean = frame.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    clean = clean.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pctChg",
            "换手率": "turn",
            "turnover": "turn",
        }
    )
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(clean.columns)
    if missing:
        raise ValueError(f"AKShare返回缺少字段: {', '.join(sorted(missing))}")

    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount", "pctChg", "turn"]:
        if column not in clean:
            clean[column] = np.nan
        _number(clean, column)

    clean = clean.dropna(subset=["date", "close"]).sort_values("date")
    clean["preclose"] = clean["close"].shift(1)
    calculated_change = clean["close"].pct_change(fill_method=None).mul(100)
    clean["pctChg"] = clean["pctChg"].fillna(calculated_change)

    # Tencent uses a decimal turnover ratio while Eastmoney returns percent.
    non_null_turn = clean["turn"].dropna().abs()
    if not non_null_turn.empty and non_null_turn.quantile(0.95) <= 1:
        clean["turn"] = clean["turn"].mul(100)

    # These fields cannot be inferred honestly from an OHLCV series.  NaN is
    # intentional: downstream metadata will mark the cache as research-only.
    for column in ["pe", "pb", "tradestatus", "isST"]:
        clean[column] = np.nan
    return clean[ALICE_COLUMNS].reset_index(drop=True)


class BaoStockProvider:
    name = "baostock"
    warnings: list[str] = []
    supports_security_master = True

    def __init__(self, retries=3, **_kwargs):
        self.retries = max(1, int(retries))

    def security_master(self):
        return baostock_fetcher.get_security_master()

    def stock_list(self, market="main"):
        return baostock_fetcher.get_stock_list(market)

    def liquid_universe(self, top_n=500):
        return baostock_fetcher.get_liquid_universe(top_n)

    def batch(self, codes, start, end, adjust="2", verbose=True):
        return baostock_fetcher.get_batch_data(
            codes,
            start,
            end,
            adjust=adjust,
            verbose=verbose,
            retries=self.retries,
        )

    def index(self, code, start, end):
        return baostock_fetcher.get_index_data(code, start, end)


class AKShareProvider:
    warnings = [AKSHARE_WARNING]
    supports_security_master = False

    def __init__(self, endpoint="tencent", delay=0.8, retries=3):
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("未安装AKShare，请先执行 pip install -r requirements.txt") from exc
        self.ak = ak
        self.endpoint = endpoint
        self.name = f"akshare-{endpoint}"
        self.delay = max(0.0, float(delay))
        self.retries = max(1, int(retries))

    @staticmethod
    def _ak_code(code):
        value = str(code).lower().replace(".", "")
        if value.startswith(("sh", "sz", "bj")):
            return value
        tail = value[-6:]
        return ("sh" if tail.startswith("6") else "sz") + tail

    @staticmethod
    def _compact_date(value):
        return pd.Timestamp(value).strftime("%Y%m%d")

    def _retry(self, operation, label):
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                return operation()
            except Exception as exc:  # Network libraries raise several types.
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(8.0, 1.5**attempt))
        raise RuntimeError(f"{label}连续失败{self.retries}次: {last_error}") from last_error

    def _stock(self, code, start, end, adjust):
        symbol = self._ak_code(code)
        adjust_name = {"1": "hfq", "2": "qfq", "3": ""}[str(adjust)]
        if self.endpoint == "eastmoney":
            raw = self._retry(
                lambda: self.ak.stock_zh_a_hist(
                    symbol=symbol[-6:],
                    period="daily",
                    start_date=self._compact_date(start),
                    end_date=self._compact_date(end),
                    adjust=adjust_name,
                ),
                f"东财 {code}",
            )
        else:
            raw = self._retry(
                lambda: self.ak.stock_zh_a_hist_tx(
                    symbol=symbol,
                    start_date=self._compact_date(start),
                    end_date=self._compact_date(end),
                    adjust=adjust_name,
                ),
                f"腾讯 {code}",
            )
        return _normalise_akshare(raw)

    def security_master(self):
        raise RuntimeError("AKShare免费日线接口不提供可靠的含退市日期证券主数据")

    def stock_list(self, market="main"):
        if market != "main":
            raise RuntimeError("AKShare v4仅支持从文件或历史证券主数据选择股票池")
        spot = self.ak.stock_zh_a_spot_em()
        values = spot.rename(columns={"代码": "code", "名称": "name"})
        values = values[~values["name"].astype(str).str.contains("ST|退", na=False)]
        result = []
        for value in values["code"].astype(str):
            if value.startswith(("600", "601", "603", "605")):
                result.append(f"sh.{value}")
            elif value.startswith(("000", "001", "002", "003")):
                result.append(f"sz.{value}")
        return sorted(set(result))

    def liquid_universe(self, top_n=500):
        return baostock_fetcher.get_liquid_universe(top_n)

    def batch(self, codes, start, end, adjust="2", verbose=True):
        results = {}
        failures = []
        total = len(codes)
        for index, code in enumerate(codes, start=1):
            try:
                frame = self._stock(code, start, end, adjust)
                if frame is not None and not frame.empty:
                    results[code] = frame
                else:
                    failures.append(f"{code}: 无数据")
            except Exception as exc:
                failures.append(f"{code}: {exc}")
            if verbose:
                print(f"    {index}/{total} {code}")
            if self.delay and index < total:
                time.sleep(self.delay)
        if failures:
            print("  本批失败:")
            for message in failures:
                print(f"    - {message}")
        return results

    def index(self, code, start, end):
        symbol = self._ak_code(code)
        raw = self._retry(
            lambda: self.ak.stock_zh_index_daily(symbol=symbol),
            f"指数 {code}",
        )
        clean = _normalise_akshare(raw)
        if clean is None:
            return None
        mask = clean["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        return clean.loc[mask].reset_index(drop=True)


def create_provider(name, *, delay=0.8, retries=3):
    if name == "baostock":
        return BaoStockProvider(retries=retries)
    if name == "akshare-eastmoney":
        return AKShareProvider("eastmoney", delay=delay, retries=retries)
    if name == "akshare-tencent":
        return AKShareProvider("tencent", delay=delay, retries=retries)
    raise ValueError(f"未知数据源: {name}")
