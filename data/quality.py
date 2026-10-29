"""Market-data completeness and quality checks for Alice caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

import pandas as pd


REQUIRED_STOCK_COLUMNS = {"date", "open", "high", "low", "close", "volume", "amount"}


@dataclass(frozen=True)
class ExpectedPeriod:
    start: pd.Timestamp
    end: pd.Timestamp


def _master_lookup(security_master):
    if isinstance(security_master, dict):
        return security_master
    if security_master is None or security_master.empty:
        return {}
    result = {}
    for row in security_master.itertuples(index=False):
        code = str(getattr(row, "code"))
        result[code] = (
            pd.to_datetime(getattr(row, "ipoDate", None), errors="coerce"),
            pd.to_datetime(getattr(row, "outDate", None), errors="coerce"),
        )
    return result


def expected_period(code, start, end, security_master=None):
    """Clip a requested range to the security's listing lifetime when known."""
    requested_start = pd.Timestamp(start).normalize()
    requested_end = pd.Timestamp(end).normalize()
    listing = _master_lookup(security_master).get(str(code))
    if listing:
        ipo_date, out_date = listing
        if pd.notna(ipo_date):
            requested_start = max(requested_start, ipo_date.normalize())
        if pd.notna(out_date):
            requested_end = min(requested_end, out_date.normalize())
    if requested_start > requested_end:
        return None
    return ExpectedPeriod(requested_start, requested_end)


def boundary_complete(frame, period, grace_days=10):
    """Return True when both requested boundaries are represented.

    Calendar-day grace accommodates weekends and long public holidays. Internal
    session gaps are handled separately by the full quality audit.
    """
    if period is None:
        return True
    if frame is None or frame.empty or "date" not in frame:
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    span_days = max(0, (period.end - period.start).days)
    effective_grace = min(int(grace_days), max(1, span_days // 4))
    return bool(
        dates.min().normalize() <= period.start + pd.Timedelta(days=effective_grace)
        and dates.max().normalize() >= period.end - pd.Timedelta(days=effective_grace)
    )


def _internal_gap_windows(expected_dates, actual_dates, minimum_sessions):
    """Return long missing runs measured on the benchmark trading calendar."""
    expected = pd.DatetimeIndex(expected_dates).normalize().unique().sort_values()
    actual = pd.DatetimeIndex(actual_dates).normalize().unique().sort_values()
    present_mask = expected.isin(actual)
    missing_positions = [position for position, present in enumerate(present_mask) if not present]
    if not missing_positions:
        return []

    windows = []
    run_start = previous = missing_positions[0]
    for position in missing_positions[1:]:
        if position != previous + 1:
            if previous - run_start + 1 >= minimum_sessions:
                windows.append((expected[run_start], expected[previous]))
            run_start = position
        previous = position
    if previous - run_start + 1 >= minimum_sessions:
        windows.append((expected[run_start], expected[previous]))
    return windows


def pending_downloads(
    data,
    codes,
    start,
    end,
    security_master=None,
    grace_days=10,
    benchmark_dates=None,
    internal_gap_sessions=20,
):
    """Return per-code boundary and long internal-gap download windows."""
    jobs = []
    master = _master_lookup(security_master)
    benchmark_calendar = pd.DatetimeIndex([])
    if benchmark_dates is not None:
        values = pd.to_datetime(pd.Series(benchmark_dates), errors="coerce").dropna()
        benchmark_calendar = pd.DatetimeIndex(values).normalize().unique().sort_values()
    for code in codes:
        period = expected_period(code, start, end, master)
        if period is None:
            continue
        frame = (data or {}).get(code)
        if frame is None or frame.empty or "date" not in frame:
            jobs.append((code, period.start, period.end, "full"))
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if dates.empty:
            jobs.append((code, period.start, period.end, "full"))
            continue
        actual_start = dates.min().normalize()
        actual_end = dates.max().normalize()
        span_days = max(0, (period.end - period.start).days)
        effective_grace = min(int(grace_days), max(1, span_days // 4))
        if actual_start > period.start + pd.Timedelta(days=effective_grace):
            head_end = min(period.end, actual_start - pd.Timedelta(days=1))
            jobs.append((code, period.start, head_end, "head"))
        if actual_end < period.end - pd.Timedelta(days=effective_grace):
            tail_start = max(period.start, actual_end + pd.Timedelta(days=1))
            jobs.append((code, tail_start, period.end, "tail"))
        if len(benchmark_calendar):
            internal_expected = benchmark_calendar[
                (benchmark_calendar >= max(period.start, actual_start))
                & (benchmark_calendar <= min(period.end, actual_end))
            ]
            for gap_start, gap_end in _internal_gap_windows(
                internal_expected,
                dates,
                max(1, int(internal_gap_sessions)),
            ):
                jobs.append((code, gap_start, gap_end, "gap"))
    return jobs


def completion_map(data, codes, start, end, security_master=None, grace_days=10):
    master = _master_lookup(security_master)
    return {
        str(code): boundary_complete(
            (data or {}).get(code),
            expected_period(code, start, end, master),
            grace_days=grace_days,
        )
        for code in codes
    }


def _longest_missing_run(expected_dates, actual_dates):
    longest = current = 0
    actual = set(actual_dates)
    for date in expected_dates:
        if date in actual:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def audit_market_data(
    data,
    *,
    benchmark,
    start,
    end,
    requested_codes=None,
    security_master=None,
    minimum_coverage=0.90,
    grace_days=10,
):
    """Build a JSON-serializable, point-in-time quality report."""
    requested = sorted(
        set(str(code) for code in (requested_codes or [code for code in data if code != benchmark]))
    )
    benchmark_frame = data.get(benchmark)
    master = _master_lookup(security_master)
    benchmark_dates = pd.DatetimeIndex([])
    if benchmark_frame is not None and not benchmark_frame.empty:
        values = pd.to_datetime(benchmark_frame["date"], errors="coerce").dropna()
        benchmark_dates = pd.DatetimeIndex(values).normalize().unique().sort_values()

    rows = []
    for code in requested:
        frame = data.get(code)
        period = expected_period(code, start, end, master)
        if period is None:
            continue
        missing_columns = sorted(
            REQUIRED_STOCK_COLUMNS.difference(frame.columns if frame is not None else [])
        )
        if frame is None or frame.empty or missing_columns:
            rows.append(
                {
                    "code": code,
                    "status": "fail",
                    "row_count": 0,
                    "actual_start": None,
                    "actual_end": None,
                    "expected_start": period.start.date().isoformat(),
                    "expected_end": period.end.date().isoformat(),
                    "session_coverage": 0.0,
                    "missing_sessions": None,
                    "longest_missing_run": None,
                    "invalid_ohlc_rows": None,
                    "unpriced_non_trading_rows": None,
                    "missing_columns": missing_columns,
                }
            )
            continue

        clean = frame.copy()
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
        clean = clean.dropna(subset=["date"])
        in_period = clean[clean["date"].between(period.start, period.end)]
        actual_dates = pd.DatetimeIndex(in_period["date"].unique()).sort_values()
        expected_dates = benchmark_dates[
            (benchmark_dates >= period.start) & (benchmark_dates <= period.end)
        ]
        if len(expected_dates):
            present = len(expected_dates.intersection(actual_dates))
            coverage = present / len(expected_dates)
            missing_sessions = len(expected_dates) - present
            longest_missing = _longest_missing_run(expected_dates, actual_dates)
        else:
            coverage = 0.0
            missing_sessions = None
            longest_missing = None

        numeric = in_period[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        trade_status = pd.to_numeric(
            in_period.get("tradestatus", pd.Series(1, index=in_period.index)),
            errors="coerce",
        )
        unpriced_non_trading = trade_status.eq(0) & numeric.isna().all(axis=1)
        invalid = (
            numeric.isna().any(axis=1)
            | (numeric <= 0).any(axis=1)
            | (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        ) & ~unpriced_non_trading
        boundaries_ok = boundary_complete(in_period, period, grace_days)
        status = (
            "pass"
            if (boundaries_ok and coverage >= minimum_coverage and not invalid.any())
            else "fail"
        )
        rows.append(
            {
                "code": code,
                "status": status,
                "row_count": int(len(in_period)),
                "actual_start": actual_dates.min().date().isoformat()
                if len(actual_dates)
                else None,
                "actual_end": actual_dates.max().date().isoformat() if len(actual_dates) else None,
                "expected_start": period.start.date().isoformat(),
                "expected_end": period.end.date().isoformat(),
                "session_coverage": round(float(coverage), 6),
                "missing_sessions": missing_sessions,
                "longest_missing_run": longest_missing,
                "invalid_ohlc_rows": int(invalid.sum()),
                "unpriced_non_trading_rows": int(unpriced_non_trading.sum()),
                "missing_columns": missing_columns,
            }
        )

    passed = sum(row["status"] == "pass" for row in rows)
    failed = len(rows) - passed
    coverage_values = [row["session_coverage"] for row in rows]
    summary = {
        "formal_ready": bool(rows and failed == 0),
        "requested_codes": len(requested),
        "audited_codes": len(rows),
        "passed_codes": passed,
        "failed_codes": failed,
        "minimum_coverage": float(minimum_coverage),
        "median_session_coverage": round(float(pd.Series(coverage_values).median()), 6)
        if coverage_values
        else 0.0,
    }
    return {"summary": summary, "stocks": rows}


def save_quality_report(cache_path, report):
    """Write a compact JSON summary and per-stock CSV next to the cache."""
    resolved = os.path.abspath(os.path.expanduser(cache_path))
    json_path = f"{resolved}.quality.json"
    csv_path = f"{resolved}.quality.csv"
    json_tmp = f"{json_path}.tmp"
    csv_tmp = f"{csv_path}.tmp"
    try:
        with open(json_tmp, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        pd.DataFrame(report.get("stocks") or []).to_csv(csv_tmp, index=False, encoding="utf-8-sig")
        os.replace(json_tmp, json_path)
        os.replace(csv_tmp, csv_path)
    finally:
        for path in (json_tmp, csv_tmp):
            if os.path.exists(path):
                os.remove(path)
    return json_path, csv_path
