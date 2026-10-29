"""Batch collection and rejection audit for external candidate feeds."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from integration.candidate_feed import CandidateFeed, MAX_FEED_BYTES, load_candidate_feed
from integration.observation_pool import (
    load_observation_pool,
    record_candidate_feed,
)


AUDIT_SCHEMA_VERSION = 1
MAX_AUDIT_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class CollectionResult:
    scanned: int
    imported: int
    existing: int
    rejected: int
    known_rejected: int
    archived_new: int
    archived_existing: int
    missing_trade_dates: tuple[str, ...]
    pool_summary: dict[str, Any]


def archive_candidate_feed(feed: CandidateFeed, archive_dir: str) -> tuple[str, bool]:
    """Create an immutable byte-for-byte archive addressed by source hash."""

    source_path = os.path.abspath(feed.source_path)
    with open(source_path, "rb") as handle:
        raw = handle.read(MAX_FEED_BYTES + 1)
    if len(raw) > MAX_FEED_BYTES:
        raise ValueError("候选文件超过5 MB限制")
    if hashlib.sha256(raw).hexdigest() != feed.source_sha256:
        raise ValueError("候选源文件在校验后发生变化，拒绝归档")

    directory = os.path.abspath(os.path.expanduser(archive_dir))
    os.makedirs(directory, exist_ok=True)
    source_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feed.source_system).strip("_.")
    source_name = (source_name or "source")[:48]
    destination = os.path.join(
        directory,
        f"{feed.trade_date}_{source_name}_{feed.source_sha256}.json",
    )
    if os.path.exists(destination):
        with open(destination, "rb") as handle:
            archived = handle.read(MAX_FEED_BYTES + 1)
        if archived != raw:
            raise ValueError("候选原始归档哈希文件内容不一致，请人工复核")
        return destination, False

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".candidate_source_",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            with open(destination, "rb") as handle:
                archived = handle.read(MAX_FEED_BYTES + 1)
            if archived != raw:
                raise ValueError("候选原始归档并发写入内容不一致，请人工复核")
            return destination, False
        return destination, True
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _parse_date(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"{field}必须是YYYY-MM-DD")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field}必须是有效日期") from exc


def _source_fingerprint(path: str) -> tuple[str, str | None, int]:
    absolute_path = os.path.abspath(path)
    size = os.path.getsize(absolute_path)
    if size <= MAX_FEED_BYTES:
        digest = hashlib.sha256()
        with open(absolute_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        return hexdigest, hexdigest, size
    stat = os.stat(absolute_path)
    marker = f"{absolute_path}|{size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(marker).hexdigest(), None, size


def _empty_audit() -> dict[str, Any]:
    return {"schema_version": AUDIT_SCHEMA_VERSION, "rejections": []}


def load_rejection_audit(path: str) -> dict[str, Any]:
    absolute_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(absolute_path):
        return _empty_audit()
    if os.path.getsize(absolute_path) > MAX_AUDIT_BYTES:
        raise ValueError("候选拒绝审计超过10 MB限制")
    try:
        with open(absolute_path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("候选拒绝审计不是有效JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("不支持的候选拒绝审计版本")
    records = data.get("rejections")
    if not isinstance(records, list):
        raise ValueError("候选拒绝审计rejections必须是列表")
    keys = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("候选拒绝记录必须是JSON对象")
        key = str(record.get("source_key") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", key) or key in keys:
            raise ValueError("候选拒绝审计包含无效或重复指纹")
        if not str(record.get("reason") or ""):
            raise ValueError("候选拒绝记录缺少原因")
        keys.add(key)
    return copy.deepcopy(data)


def _atomic_write_audit(path: str, data: dict[str, Any]) -> None:
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(serialized.encode("utf-8")) > MAX_AUDIT_BYTES:
        raise ValueError("候选拒绝审计超过10 MB限制")
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".candidate_collection_audit_",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _record_rejection(
    audit_path: str,
    audit: dict[str, Any],
    *,
    source_path: str,
    source_key: str,
    source_sha256: str | None,
    file_size: int,
    reason: str,
    rejected_at: datetime,
) -> bool:
    records = audit["rejections"]
    if any(record["source_key"] == source_key for record in records):
        return False
    records.append(
        {
            "source_key": source_key,
            "source_sha256": source_sha256,
            "source_path": os.path.abspath(source_path),
            "file_size": file_size,
            "rejected_at": rejected_at.isoformat(timespec="seconds"),
            "reason": reason,
        }
    )
    records.sort(key=lambda item: (item["rejected_at"], item["source_path"]))
    _atomic_write_audit(audit_path, audit)
    return True


def load_trading_dates(path: str) -> tuple[str, ...]:
    """Load explicit exchange sessions from JSON, CSV or plain text."""

    absolute_path = os.path.abspath(os.path.expanduser(path))
    with open(absolute_path, "r", encoding="utf-8-sig") as handle:
        text = handle.read()
    values: list[Any]
    if absolute_path.lower().endswith(".json"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("交易日历不是有效JSON") from exc
        if isinstance(payload, dict):
            payload = payload.get("trade_dates")
        if not isinstance(payload, list):
            raise ValueError("交易日历JSON必须是日期列表或包含trade_dates")
        values = payload
    else:
        values = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        if not values:
            raise ValueError("交易日历文件没有YYYY-MM-DD日期")
    return tuple(sorted({_parse_date(value, "交易日历日期") for value in values}))


def find_missing_trade_dates(
    pool: dict[str, Any],
    trading_dates: tuple[str, ...],
    *,
    start_date: str | None = None,
    through_date: str | None = None,
) -> tuple[str, ...]:
    observed = {record["trade_date"] for record in pool["imports"]}
    expected = tuple(sorted({_parse_date(item, "交易日历日期") for item in trading_dates}))
    if not expected:
        return ()
    start = (
        _parse_date(start_date, "start_date")
        if start_date
        else (min(observed) if observed else expected[0])
    )
    end = (
        _parse_date(through_date, "through_date")
        if through_date
        else (max(observed) if observed else expected[-1])
    )
    if start > end:
        raise ValueError("start_date不能晚于through_date")
    return tuple(item for item in expected if start <= item <= end and item not in observed)


def collect_candidate_directory(
    source_dir: str,
    *,
    pool_path: str,
    audit_path: str,
    pattern: str = "candidates_*.json",
    trading_dates: tuple[str, ...] = (),
    start_date: str | None = None,
    through_date: str | None = None,
    retry_rejections: bool = False,
    archive_dir: str | None = None,
    collected_at: datetime | None = None,
) -> CollectionResult:
    """Collect every matching feed independently and keep failures auditable."""

    timestamp = collected_at or datetime.now().astimezone()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("collected_at必须包含时区")
    directory = Path(os.path.abspath(os.path.expanduser(source_dir)))
    if not directory.is_dir():
        raise ValueError(f"候选目录不存在: {directory}")
    if not pattern or re.search(r"[\\/]", pattern):
        raise ValueError("pattern只能是候选目录内的文件名模式")
    if (start_date or through_date) and not trading_dates:
        raise ValueError("日期缺口范围必须配合显式交易日历")
    paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    pool_absolute = os.path.abspath(os.path.expanduser(pool_path))
    audit_absolute = os.path.abspath(os.path.expanduser(audit_path))
    if pool_absolute == audit_absolute:
        raise ValueError("观察池与拒绝审计不能使用同一个文件")
    load_observation_pool(pool_absolute)
    audit = load_rejection_audit(audit_absolute)
    rejected_keys = {record["source_key"] for record in audit["rejections"]}
    imported = existing = rejected = known_rejected = 0
    archived_new = archived_existing = 0

    for path in paths:
        source_key, source_sha256, file_size = _source_fingerprint(str(path))
        if source_key in rejected_keys and not retry_rejections:
            known_rejected += 1
            continue
        try:
            feed = load_candidate_feed(str(path))
        except (OSError, ValueError) as exc:
            reason = str(exc)
            is_new = _record_rejection(
                audit_absolute,
                audit,
                source_path=str(path),
                source_key=source_key,
                source_sha256=source_sha256,
                file_size=file_size,
                reason=reason,
                rejected_at=timestamp,
            )
            if is_new:
                rejected_keys.add(source_key)
                rejected += 1
            else:
                known_rejected += 1
            continue
        if archive_dir:
            _, archive_created = archive_candidate_feed(feed, archive_dir)
            if archive_created:
                archived_new += 1
            else:
                archived_existing += 1
        try:
            result = record_candidate_feed(
                pool_absolute,
                feed,
                recorded_at=timestamp,
            )
        except (OSError, ValueError) as exc:
            reason = str(exc)
            is_new = _record_rejection(
                audit_absolute,
                audit,
                source_path=str(path),
                source_key=source_key,
                source_sha256=source_sha256,
                file_size=file_size,
                reason=reason,
                rejected_at=timestamp,
            )
            if is_new:
                rejected_keys.add(source_key)
                rejected += 1
            else:
                known_rejected += 1
            continue
        if result.imported:
            imported += 1
        else:
            existing += 1

    pool = load_observation_pool(pool_absolute)
    missing = (
        find_missing_trade_dates(
            pool,
            trading_dates,
            start_date=start_date,
            through_date=through_date,
        )
        if trading_dates
        else ()
    )
    return CollectionResult(
        scanned=len(paths),
        imported=imported,
        existing=existing,
        rejected=rejected,
        known_rejected=known_rejected,
        archived_new=archived_new,
        archived_existing=archived_existing,
        missing_trade_dates=missing,
        pool_summary={
            "import_count": len(pool["imports"]),
            "trade_dates": sorted({record["trade_date"] for record in pool["imports"]}),
        },
    )
