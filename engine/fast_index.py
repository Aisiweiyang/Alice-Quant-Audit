"""按股票的日期快速索引。

背景：回测主循环此前对每个(股票×交易日)做一次布尔掩码/索引查找，
select_candidates 在每个调仓日对全部股票重复构建日期掩码。本模块把每只
股票的日期定位（排序数组+searchsorted+行号字典）、常用列数组、执行门禁
所需字段、MA50等滚动结果在**单次回测内**缓存一次，供主循环与选股复用。

行为等价红线：
- 行查找与 ``engine.backtest._row_on`` 一致（同一日期取最后一行；找不到返回None）；
- ``count_leq`` 与布尔掩码 ``df['date'] <= target`` 的行数一致（要求日期列
  非降序，这正是 cache_store._normalise_frame 与 CSV 导入路径的既定形态）；
- 布尔/数值数组经 pd.to_numeric 逐元素转换，与原逐行 to_numeric 结果一致；
- 帧不满足前提（NaT日期、日期非降序、DatetimeIndex与date列不一致）时
  ``build_stock_index`` 返回 None，调用方必须回退原实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 执行门禁（engine.execution）会访问的行情字段；dict行只包含帧中存在的字段，
# 缺失字段与 Series.get 一样返回 None，默认值语义由调用方保持。
EXECUTION_ROW_COLUMNS = (
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "preclose",
    "pctChg",
    "tradestatus",
    "isST",
    "pe",
    "pb",
    "date",
)


class StockIndex:
    """单只股票行情的日期定位与列数组缓存。"""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.n_rows = len(frame)
        values = pd.to_datetime(frame["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        if np.any(np.isnat(values)):
            raise ValueError("存在NaT日期，无法建立快速索引")
        if values.size > 1 and np.any(values[1:] < values[:-1]):
            raise ValueError("日期非降序，无法建立快速索引")
        if isinstance(frame.index, pd.DatetimeIndex):
            index_values = frame.index.to_numpy(dtype="datetime64[ns]")
            if not np.array_equal(index_values, values):
                raise ValueError("DatetimeIndex与date列不一致，无法建立快速索引")
        self.dates = values
        self._date_ints = values.astype("int64")
        # 同一日期多次出现时保留最后一次出现的行号，与 rows.iloc[-1] 一致。
        position_of = {}
        for position, value in enumerate(self._date_ints.tolist()):
            position_of[value] = position
        self.position_of = position_of
        self._columns: dict[str, np.ndarray] = {}
        self._coerced: dict[str, np.ndarray] = {}
        self._rolling_means: dict[int, np.ndarray] = {}
        self.columns_present = set(frame.columns)

    # ---- 列数组 ----
    def column(self, name: str) -> np.ndarray | None:
        if name not in self.columns_present:
            return None
        arr = self._columns.get(name)
        if arr is None:
            arr = self.frame[name].to_numpy()
            self._columns[name] = arr
        return arr

    def coerced(self, name: str) -> np.ndarray | None:
        """pd.to_numeric(errors='coerce') 后的数组；列不存在返回None。"""
        if name not in self.columns_present:
            return None
        arr = self._coerced.get(name)
        if arr is None:
            arr = pd.to_numeric(self.frame[name], errors="coerce").to_numpy(dtype="float64")
            self._coerced[name] = arr
        return arr

    def coerced_filled(self, name: str, fill_value: float) -> np.ndarray | None:
        if name not in self.columns_present:
            return None
        key = f"{name}#{fill_value}"
        arr = self._coerced.get(key)
        if arr is None:
            arr = (
                pd.to_numeric(self.frame[name], errors="coerce")
                .fillna(fill_value)
                .to_numpy(dtype="float64")
            )
            self._coerced[key] = arr
        return arr

    def rolling_mean(self, column: str, period: int) -> np.ndarray | None:
        """整段序列的一次性滚动均值（因果：位置t只依赖t及之前）。"""
        key = f"{column}#{period}"
        arr = self._rolling_means.get(key)
        if arr is None:
            closes = self.coerced(column)
            if closes is None:
                return None
            arr = pd.Series(closes).rolling(period).mean().to_numpy(dtype="float64")
            self._rolling_means[key] = arr
        return arr

    # ---- 行定位 ----
    def count_leq(self, timestamp_value: int) -> int:
        """日期 <= 给定纳秒时间戳的行数（等价于布尔掩码求和）。"""
        return int(np.searchsorted(self._date_ints, timestamp_value, side="right"))

    def row_at(self, position: int):
        return self.frame.iloc[position]

    def execution_row(self, position: int) -> dict:
        """轻量行情行（仅含帧中存在的字段），供 engine.execution 的 .get 访问。"""
        row = {}
        for name in EXECUTION_ROW_COLUMNS:
            if name not in self.columns_present:
                continue
            row[name] = self.column(name)[position]
        return row

    # ---- 选股前置条件（数组级短路，结果与 passes_strategy_filters 前几条一致）----
    def passes_leading_filters(self, n_hist: int, config) -> bool:
        days = config.consecutive_up_days
        stock_ma_period = getattr(config, "stock_ma_period", 50)
        if n_hist < max(stock_ma_period, 20, days):
            return False
        # 与原实现一致：缺列时让 KeyError 抛出，而不是静默跳过。
        pct = self.coerced("pctChg")
        if pct is None:
            raise KeyError("pctChg")
        window = pct[n_hist - days : n_hist]
        if len(window) < days or np.isnan(window).any():
            return False
        if not ((window > 0) & (window <= config.max_daily_gain_pct)).all():
            return False
        closes = self.coerced("close")
        if closes is None:
            raise KeyError("close")
        ma = self.rolling_mean("close", stock_ma_period)[n_hist - 1]
        if pd.isna(ma) or closes[n_hist - 1] < ma:
            return False
        return True


def build_stock_index(frame: pd.DataFrame) -> StockIndex | None:
    """构造单只股票索引；帧不满足前提时返回None（调用方回退原实现）。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        return None
    if "date" not in frame.columns:
        return None
    try:
        return StockIndex(frame)
    except (ValueError, TypeError, KeyError):
        return None


def build_stock_indexes(data: dict) -> dict:
    """对整个股票池建立索引；值为None的股票由调用方按原实现处理。"""
    indexes = {}
    for code, frame in (data or {}).items():
        try:
            indexes[code] = build_stock_index(frame)
        except (ValueError, TypeError, KeyError):
            indexes[code] = None
    return indexes
