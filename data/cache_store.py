"""Versioned market-data cache helpers.

Pickle files are only safe when they were created locally or come from a
trusted source.  The sidecar JSON makes each dataset inspectable without
opening the pickle file.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
import pickle
from typing import Any

import pandas as pd


SCHEMA_VERSION = 3
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume", "amount"}


def metadata_path(cache_path: str) -> str:
    return f"{os.path.abspath(os.path.expanduser(cache_path))}.metadata.json"


def universe_fingerprint(codes) -> str:
    payload = "\n".join(sorted(str(code) for code in codes)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_cache_key(*, codes, start_date, end_date, adjust, benchmark, universe_mode) -> str:
    """Return a cache key that changes when the actual universe changes."""
    payload = {
        "adjust": str(adjust),
        "benchmark": benchmark,
        "codes": sorted(str(code) for code in codes),
        "end": str(end_date),
        "start": str(start_date),
        "universe_mode": universe_mode,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _normalise_frame(code: str, frame: pd.DataFrame, required_columns=None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{code} 的行情不是DataFrame")
    required = set(required_columns or REQUIRED_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{code} 缺少字段: {', '.join(sorted(missing))}")

    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    clean = clean.dropna(subset=["date"]).sort_values("date")
    clean = clean.drop_duplicates(subset=["date"], keep="last")
    clean.set_index("date", drop=False, inplace=True)
    clean.index.name = "_trade_date"
    return clean


def validate_market_data(data: dict[str, pd.DataFrame], benchmark: str | None = None):
    if not isinstance(data, dict) or not data:
        raise ValueError("行情缓存必须是非空的股票代码/DataFrame映射")
    clean = {}
    for code, frame in data.items():
        code = str(code)
        required = {"date", "close"} if benchmark and code == benchmark else REQUIRED_COLUMNS
        clean[code] = _normalise_frame(code, frame, required)
    if benchmark and benchmark not in clean:
        raise ValueError(f"缓存缺少基准指数 {benchmark}")
    return clean


def merge_market_data(existing, updates, benchmark: str | None = None):
    """Merge incremental rows while keeping the latest row for each date."""
    result = dict(existing or {})
    for code, frame in (updates or {}).items():
        if code in result:
            combined = pd.concat([result[code], frame], ignore_index=True)
        else:
            combined = frame
        required = {"date", "close"} if benchmark and code == benchmark else REQUIRED_COLUMNS
        result[code] = _normalise_frame(code, combined, required)
    return result


def _date_bounds(data):
    dates = []
    rows = 0
    for frame in data.values():
        if frame.empty:
            continue
        dates.extend([frame["date"].min(), frame["date"].max()])
        rows += len(frame)
    if not dates:
        return None, None, rows
    return min(dates).date().isoformat(), max(dates).date().isoformat(), rows


def has_historical_trade_status(data, benchmark=None):
    """Return whether tradable stock rows have complete execution-status fields."""
    for code, frame in data.items():
        if code == benchmark:
            continue
        required = {"tradestatus", "isST", "preclose"}
        if not required.issubset(frame.columns):
            return False
        trade_status = pd.to_numeric(frame["tradestatus"], errors="coerce")
        is_st = pd.to_numeric(frame["isST"], errors="coerce")
        preclose = pd.to_numeric(frame["preclose"], errors="coerce")
        if trade_status.isna().any() or is_st.isna().any():
            return False
        if preclose[trade_status.ne(0)].isna().any():
            return False
    return True


def build_metadata(data, *, benchmark, adjust, universe_mode, source, warnings=None):
    start, end, row_count = _date_bounds(data)
    stock_codes = sorted(code for code in data if code != benchmark)
    stocks = [frame for code, frame in data.items() if code != benchmark]
    valuation_cells = sum(len(frame) * 2 for frame in stocks)
    valuation_values = sum(
        frame[[column for column in ("pe", "pb") if column in frame]].notna().sum().sum()
        for frame in stocks
    )
    valuation_coverage = valuation_values / valuation_cells if valuation_cells else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "period": {"start": start, "end": end},
        "adjust": str(adjust),
        "benchmark": benchmark,
        "universe": {
            "mode": universe_mode,
            "stock_count": len(stock_codes),
            "codes_sha256": universe_fingerprint(stock_codes),
        },
        "row_count": row_count,
        "capabilities": {
            "historical_trade_status": has_historical_trade_status(data, benchmark),
            "historical_valuation_fields": bool(valuation_coverage >= 0.25),
            "valuation_coverage": round(float(valuation_coverage), 6),
        },
        "warnings": list(warnings or []),
    }


def save_market_cache(cache_path: str, data, metadata: dict[str, Any]):
    """Atomically write the pickle and its human-readable metadata."""
    resolved = os.path.abspath(os.path.expanduser(cache_path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    clean = validate_market_data(data, metadata.get("benchmark"))

    pickle_tmp = f"{resolved}.tmp"
    metadata_file = metadata_path(resolved)
    metadata_tmp = f"{metadata_file}.tmp"
    try:
        with open(pickle_tmp, "wb") as handle:
            pickle.dump(clean, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(metadata_tmp, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(pickle_tmp, resolved)
        os.replace(metadata_tmp, metadata_file)
    finally:
        for path in (pickle_tmp, metadata_tmp):
            if os.path.exists(path):
                os.remove(path)
    return resolved


def load_market_cache(cache_path: str, benchmark: str | None = None):
    """Load a trusted local cache and return ``(data, metadata)``."""
    resolved = os.path.abspath(os.path.expanduser(cache_path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"行情缓存不存在: {resolved}")
    with open(resolved, "rb") as handle:
        raw = pickle.load(handle)
    data = validate_market_data(raw, benchmark)

    sidecar = metadata_path(resolved)
    metadata = None
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    return data, metadata
