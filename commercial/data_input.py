"""商业分析数据接入：Alice缓存pickle、CSV行情目录、标准化候选名单。

只读取本地文件，不访问网络；不解析任何可执行内容。
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data.cache_store import REQUIRED_COLUMNS, _normalise_frame, load_market_cache

BENCHMARK_CODE = "sh.000300"

PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
COLUMN_ALIASES = {
    "date": "date",
    "交易日": "date",
    "时间": "date",
    "open": "open",
    "开盘": "open",
    "开盘价": "open",
    "high": "high",
    "最高": "high",
    "最高价": "high",
    "low": "low",
    "最低": "low",
    "最低价": "low",
    "close": "close",
    "收盘": "close",
    "收盘价": "close",
    "volume": "volume",
    "成交量": "volume",
    "vol": "volume",
    "amount": "amount",
    "成交额": "amount",
    "trade_date": "date",
    "日期": "date",
    "open_price": "open",
    "close_price": "close",
    "high_price": "high",
    "low_price": "low",
    "preclose": "preclose",
    "前收盘": "preclose",
    "昨收": "preclose",
    "tradestatus": "tradestatus",
    "tradestatus ": "tradestatus",
    "isst": "isST",
    "isST": "isST",
    "st": "isST",
    "pctchg": "pctChg",
    "pctChg": "pctChg",
    "涨跌幅": "pctChg",
    "pe": "pe",
    "pb": "pb",
}


class DataInputError(ValueError):
    """数据输入无法解析或不含可用行情。"""


CSV_ENCODINGS = ("utf-8-sig", "gbk", "gb18030")


def read_csv_any(path: str):
    """按常见中文编码顺序尝试读取CSV；全部失败时抛DataInputError。"""
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DataInputError(f"无法解析文件编码(尝试过{', '.join(CSV_ENCODINGS)}): {path}")


def _parse_date_column(series: pd.Series) -> pd.Series:
    """日期列解析：兼容 2025-01-01 / 2025/1/1 / 20250101 整数等格式。"""
    if pd.api.types.is_integer_dtype(series):
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    text = series.astype(str).str.strip()
    compact = text.str.fullmatch(r"\d{8}")
    if compact.any():
        parsed = text.where(
            ~compact, text.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
        )
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        result[compact] = pd.to_datetime(parsed[compact], format="%Y%m%d", errors="coerce")
        # pandas 2.x 默认按首个元素推断格式；混合格式需显式 format="mixed"。
        try:
            rest = pd.to_datetime(text[~compact], format="mixed", errors="coerce")
        except (ValueError, TypeError):
            rest = pd.to_datetime(text[~compact], errors="coerce")
        result[~compact] = rest
        return result
    try:
        return pd.to_datetime(text, format="mixed", errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(text, errors="coerce")


def normalize_code(raw) -> str | None:
    """把6位代码或带市场前缀的代码规范化为 Alice 的 sh.xxxxxx/sz.xxxxxx。"""
    text = str(raw).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        head = digits[0]
        if head in "69":
            return f"sh.{digits}"
        if head in "03":
            return f"sz.{digits}"
        return None
    if text.startswith(("sh.", "sz.")) and len(digits) == 6:
        return f"{text[:2]}.{digits}"
    return None


def _standardise_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip()
        renamed[column] = COLUMN_ALIASES.get(key, COLUMN_ALIASES.get(key.lower(), key))
    clean = frame.rename(columns=renamed)
    fabricated: list[str] = []
    if "date" not in clean.columns:
        raise DataInputError("行情文件缺少date列")
    clean["date"] = _parse_date_column(clean["date"])
    clean = clean.dropna(subset=["date"])
    if clean.empty:
        raise DataInputError("行情文件没有有效日期")
    for column in PRICE_COLUMNS:
        if column not in clean.columns:
            raise DataInputError(f"行情文件缺少{column}列")
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if "preclose" not in clean.columns:
        # 无前收盘字段时用上一交易日收盘近似，首行缺教会走pctChg限价回退分支。
        clean["preclose"] = clean["close"].shift(1)
        fabricated.append("preclose")
    if "tradestatus" not in clean.columns:
        clean["tradestatus"] = 1
        fabricated.append("tradestatus")
    if "isST" not in clean.columns:
        clean["isST"] = 0
        fabricated.append("isST")
    if "pctChg" not in clean.columns:
        clean["pctChg"] = clean["close"].pct_change() * 100
        fabricated.append("pctChg")
    return clean, fabricated


def load_pickle_cache(path: str, benchmark: str | None = BENCHMARK_CODE):
    """加载Alice格式缓存；benchmark=None时仅要求数据可解析。"""
    return load_market_cache(path, benchmark)


def load_security_master(cache: str):
    """加载缓存旁的证券主数据CSV（含ipoDate/outDate），供数据审计评估退市覆盖。

    Alice正式缓存的旁车文件命名与.metadata.json一致：在缓存全名后追加
    ".security_master.csv"。文件缺失、缺关键列或不可读时返回None，审计侧
    会如实报告"缺少证券主数据"，不伪造结论。只读取，不修改任何文件。
    """
    resolved = os.path.abspath(os.path.expanduser(cache))
    if not os.path.isfile(resolved):
        return None
    master_path = resolved + ".security_master.csv"
    if not os.path.isfile(master_path):
        return None
    try:
        frame = pd.read_csv(master_path, encoding="utf-8-sig", dtype=str)
    except (OSError, ValueError):
        # 含编码错误(UnicodeDecodeError)与解析错误(ParserError)，均为ValueError子类。
        return None
    if not {"code", "ipoDate", "outDate"}.issubset(frame.columns):
        return None
    return frame


def load_csv_source(path: str):
    """读取CSV行情：单文件(含code列)或目录(每股一个CSV)。

    返回 ``(data, warnings)``；data 已做 date 规范化和重复日期去重，
    原始重复行数通过 warnings 报告。
    """
    warnings: list[str] = []
    resolved = os.path.abspath(os.path.expanduser(path))
    frames: dict[str, pd.DataFrame] = {}
    if os.path.isdir(resolved):
        files = sorted(name for name in os.listdir(resolved) if name.lower().endswith(".csv"))
        if not files:
            raise DataInputError(f"目录中没有CSV文件: {resolved}")
        for name in files:
            code = normalize_code(os.path.splitext(name)[0])
            if code is None:
                warnings.append(f"跳过无法识别代码的文件: {name}")
                continue
            frames[code] = read_csv_any(os.path.join(resolved, name))
    else:
        if not resolved.lower().endswith(".csv"):
            raise DataInputError(f"不支持的行情文件类型: {resolved}")
        frame = read_csv_any(resolved)
        code_column = next(
            (c for c in frame.columns if str(c).strip().lower() in {"code", "代码", "stock_code"}),
            None,
        )
        if code_column is None:
            raise DataInputError("单文件CSV必须包含code(代码)列")
        for raw_code, group in frame.groupby(code_column):
            code = normalize_code(raw_code)
            if code is None:
                warnings.append(f"跳过无法识别的代码: {raw_code}")
                continue
            frames[code] = group.drop(columns=[code_column])

    data = {}
    field_provenance: dict[str, list[str]] = {}
    total_duplicates = 0
    for code, frame in frames.items():
        try:
            clean, fabricated = _standardise_frame(frame)
        except DataInputError as exc:
            warnings.append(f"{code}: {exc}")
            continue
        before = len(clean)
        clean = clean.drop_duplicates(subset=["date"], keep="last")
        total_duplicates += before - len(clean)
        clean = clean.sort_values("date").reset_index(drop=True)
        data[code] = clean
        if fabricated:
            field_provenance[code] = fabricated
    if total_duplicates:
        warnings.append(f"共去除{total_duplicates}条重复日期记录")
    if not data:
        raise DataInputError("没有解析出任何可用行情")
    data = _normalize_frames(data, warnings)
    return data, warnings, field_provenance


def _normalize_frames(data: dict, warnings: list[str]) -> dict:
    """CSV来源统一走 cache_store._normalise_frame 规范化。

    规范化后帧为 DatetimeIndex、日期规范化、升序无重复，与pickle缓存同构，
    使回测引擎与选股的快速索引路径可用；缺少必需列的帧保持原样（走慢路径）。
    """
    normalized = {}
    skipped = 0
    for code, frame in data.items():
        if REQUIRED_COLUMNS.issubset(frame.columns):
            try:
                normalized[code] = _normalise_frame(code, frame)
                continue
            except ValueError:
                pass
        skipped += 1
        normalized[code] = frame
    if skipped:
        warnings.append(f"{skipped}只股票缺少标准字段，未做缓存级规范化（按兼容模式回测）")
    return normalized


def load_candidates(path: str) -> list[str]:
    """读取标准化候选名单（integration JSON契约或含code列的CSV）。"""
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise DataInputError(f"候选名单不存在: {resolved}")
    codes: list[str] = []
    if resolved.lower().endswith(".json"):
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("candidates") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise DataInputError("候选JSON格式不正确：需要candidates数组")
        for row in rows:
            code = normalize_code((row or {}).get("code", ""))
            if code:
                codes.append(code)
    else:
        frame = read_csv_any(resolved)
        column = next(
            (c for c in frame.columns if str(c).strip().lower() in {"code", "代码"}),
            None,
        )
        if column is None:
            raise DataInputError("候选CSV必须包含code(代码)列")
        codes = [code for code in (normalize_code(v) for v in frame[column]) if code]
    unique = list(dict.fromkeys(codes))
    if not unique:
        raise DataInputError("候选名单中没有可识别的股票代码")
    return unique


def restrict_to_candidates(data: dict, codes: list[str], benchmark: str | None = BENCHMARK_CODE):
    """把行情限制到候选名单；返回(过滤后数据, 命中数, 缺失代码)。"""
    normalized = [normalize_code(code) or str(code) for code in codes]
    keep = benchmark in data
    missing = [code for code in normalized if code not in data]
    present = [code for code in normalized if code in data]
    filtered = {code: data[code] for code in present}
    if keep and benchmark not in filtered:
        filtered[benchmark] = data[benchmark]
    return filtered, len(present), missing


def load_analysis_data(cache: str, candidates: str | None = None):
    """统一入口：按扩展名加载数据并可选过滤候选名单。

    返回 ``(data, metadata, source_info)``；CSV来源的 metadata 为 None。
    """
    resolved = os.path.abspath(os.path.expanduser(cache))
    if not os.path.exists(resolved):
        raise DataInputError(f"数据路径不存在: {resolved}")
    if os.path.isdir(resolved) or resolved.lower().endswith(".csv"):
        data, warnings, provenance = load_csv_source(resolved)
        metadata = None
        source_info = {
            "kind": "csv",
            "path": resolved,
            "warnings": warnings,
            "field_provenance": provenance,
        }
    else:
        try:
            data, metadata = load_pickle_cache(resolved, benchmark=BENCHMARK_CODE)
        except ValueError as exc:
            if "缺少基准指数" not in str(exc):
                raise
            data, metadata = load_pickle_cache(resolved, benchmark=None)
            source_info = {
                "kind": "pickle",
                "path": resolved,
                "warnings": ["缓存缺少沪深300基准，将使用等权合成基准，评级最高为LIMITED"],
            }
        else:
            source_info = {"kind": "pickle", "path": resolved, "warnings": []}
    candidate_info = None
    if candidates:
        codes = load_candidates(candidates)
        data, hits, missing = restrict_to_candidates(data, codes)
        candidate_info = {
            "requested": len(codes),
            "matched": hits,
            "missing": missing[:50],
            "missing_count": len(missing),
        }
        if hits < 1:
            raise DataInputError("候选名单与行情数据没有任何交集")
        source_info["candidate_info"] = candidate_info
    return data, metadata, source_info


def synthesize_benchmark(data: dict) -> tuple[dict, str]:
    """数据缺少沪深300时，用等权股票日收益合成基准，供 LIMITED 模式使用。

    合成基准以 'synthetic:equal_weight' 为代码；正式报告中必须披露。
    """
    returns = {}
    counts = {}
    for code, frame in data.items():
        closes = pd.to_numeric(frame["close"], errors="coerce")
        dates = pd.to_datetime(frame["date"]).dt.normalize()
        frame_returns = pd.Series(closes.values, index=dates).groupby(level=0).last().pct_change()
        for date, value in frame_returns.dropna().items():
            returns[date] = returns.get(date, 0.0) + value
            counts[date] = counts.get(date, 0) + 1
    if not returns:
        raise DataInputError("无法合成基准：没有可用收益")
    rows = []
    level = 1000.0
    previous = None
    for date in sorted(returns):
        average = returns[date] / counts[date]
        if previous is not None:
            level *= 1.0 + average
        else:
            previous = date
        rows.append({"date": date, "close": level})
    frame = pd.DataFrame(rows)
    frame["preclose"] = frame["close"].shift(1)
    data = dict(data)
    data["synthetic:equal_weight"] = frame
    return data, "synthetic:equal_weight"
