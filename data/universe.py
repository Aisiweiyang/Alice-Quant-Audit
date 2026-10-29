"""Point-in-time security-master helpers for A-share universes."""

from __future__ import annotations

import hashlib
import os

import pandas as pd


SECURITY_MASTER_COLUMNS = {"code", "code_name", "ipoDate", "outDate", "type", "status"}


def is_main_board_code(code: str) -> bool:
    value = str(code).lower()
    tail = value.split(".")[-1]
    if value.startswith("sh."):
        return tail.startswith(("600", "601", "603", "605"))
    if value.startswith("sz."):
        return tail.startswith(("000", "001", "002", "003"))
    return False


def normalise_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    missing = SECURITY_MASTER_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"证券主数据缺少字段: {', '.join(sorted(missing))}")
    clean = frame.copy()
    clean["code"] = clean["code"].astype(str)
    clean["ipoDate"] = pd.to_datetime(clean["ipoDate"], errors="coerce").dt.normalize()
    clean["outDate"] = pd.to_datetime(clean["outDate"], errors="coerce").dt.normalize()
    clean["type"] = clean["type"].astype(str)
    clean["status"] = clean["status"].astype(str)
    clean = clean.drop_duplicates(subset=["code"], keep="last").sort_values("code")
    return clean.reset_index(drop=True)


def historical_main_board_codes(master: pd.DataFrame, start_date, end_date):
    """Return stocks that were listed at any time during the requested period."""
    clean = normalise_security_master(master)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    stocks = clean.loc[clean["type"].eq("1")].copy()
    stocks = stocks.loc[stocks["code"].map(is_main_board_code)]
    overlaps = stocks["ipoDate"].notna() & stocks["ipoDate"].le(end)
    overlaps &= stocks["outDate"].isna() | stocks["outDate"].ge(start)
    return sorted(stocks.loc[overlaps, "code"].tolist())


def active_on(master: pd.DataFrame, code: str, date) -> bool:
    clean = normalise_security_master(master)
    rows = clean.loc[clean["code"].eq(code)]
    if rows.empty:
        return False
    row = rows.iloc[-1]
    target = pd.Timestamp(date).normalize()
    return bool(
        pd.notna(row["ipoDate"])
        and row["ipoDate"] <= target
        and (pd.isna(row["outDate"]) or row["outDate"] >= target)
    )


def security_master_path(cache_path: str) -> str:
    return f"{os.path.abspath(os.path.expanduser(cache_path))}.security_master.csv"


def load_security_master(cache_path: str):
    path = security_master_path(cache_path)
    if not os.path.isfile(path):
        return None
    return normalise_security_master(pd.read_csv(path, encoding="utf-8-sig"))


def save_security_master(cache_path: str, master: pd.DataFrame):
    path = security_master_path(cache_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = normalise_security_master(master)
    temp_path = f"{path}.tmp"
    try:
        clean.to_csv(temp_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": os.path.basename(path),
        "row_count": len(clean),
        "sha256": digest.hexdigest(),
    }
