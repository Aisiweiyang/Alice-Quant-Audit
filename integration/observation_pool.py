"""Persist validated external candidates as a local research observation pool."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from integration.candidate_feed import CandidateFeed


POOL_SCHEMA_VERSION = 1
MAX_POOL_BYTES = 50 * 1024 * 1024


class CandidateFeedConflictError(ValueError):
    """The same source/date identity was observed with different content."""


@dataclass(frozen=True)
class ObservationResult:
    imported: bool
    source_sha256: str
    summary: dict[str, Any]


def _empty_pool() -> dict[str, Any]:
    return {"schema_version": POOL_SCHEMA_VERSION, "imports": []}


def _identity(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("source_system") or ""),
        str(record.get("strategy_version") or ""),
        str(record.get("market") or ""),
        str(record.get("trade_date") or ""),
    )


def _validate_pool(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("候选观察池顶层必须是JSON对象")
    if data.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("不支持的候选观察池版本")
    records = data.get("imports")
    if not isinstance(records, list):
        raise ValueError("候选观察池imports必须是列表")

    hashes = set()
    identities = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("候选观察池记录必须是JSON对象")
        digest = str(record.get("source_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("候选观察池包含无效源文件哈希")
        if digest in hashes:
            raise ValueError("候选观察池包含重复源文件哈希")
        identity = _identity(record)
        if any(not value for value in identity):
            raise ValueError("候选观察池记录缺少来源、策略、市场或交易日期")
        if identity in identities:
            raise ValueError("候选观察池包含重复来源日期身份")
        if not str(record.get("generated_at") or ""):
            raise ValueError("候选观察池记录缺少生成时间")
        if not isinstance(record.get("quality"), dict):
            raise ValueError("候选观察池记录缺少质量信息")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("候选观察池记录缺少候选列表")
        candidate_codes = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("候选观察池候选必须是JSON对象")
            code = str(candidate.get("code") or "")
            name = str(candidate.get("name") or "")
            rank = candidate.get("rank")
            score = candidate.get("score")
            if not code or not name:
                raise ValueError("候选观察池候选缺少代码或名称")
            if code in candidate_codes:
                raise ValueError("候选观察池单次记录包含重复股票")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("候选观察池包含无效排名")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("候选观察池包含无效分数")
            if not math.isfinite(float(score)) or not 0 <= float(score) <= 100:
                raise ValueError("候选观察池包含无效分数")
            attributes = candidate.get("attributes", {})
            if not isinstance(attributes, dict):
                raise ValueError("候选观察池候选属性必须是JSON对象")
            for value in attributes.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("候选观察池包含无效候选属性")
                if not math.isfinite(float(value)):
                    raise ValueError("候选观察池包含无效候选属性")
            candidate_codes.add(code)
        hashes.add(digest)
        identities.add(identity)
    return data


def load_observation_pool(path: str) -> dict[str, Any]:
    """Load and validate the local pool without changing it."""

    absolute_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(absolute_path):
        return _empty_pool()
    if os.path.getsize(absolute_path) > MAX_POOL_BYTES:
        raise ValueError("候选观察池超过50 MB限制")
    try:
        with open(absolute_path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("候选观察池不是有效JSON") from exc
    return copy.deepcopy(_validate_pool(data))


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(serialized.encode("utf-8")) > MAX_POOL_BYTES:
        raise ValueError("候选观察池超过50 MB限制")
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".candidate_observations_",
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


def summarise_observation_pool(data: dict[str, Any]) -> dict[str, Any]:
    """Build repeat-appearance statistics without creating signals or orders."""

    validated = _validate_pool(copy.deepcopy(data))
    records = sorted(
        validated["imports"],
        key=lambda item: (
            item["trade_date"],
            item["source_system"],
            item["strategy_version"],
            item["generated_at"],
        ),
    )
    by_code: dict[str, dict[str, Any]] = {}
    for record in records:
        for candidate in record["candidates"]:
            code = str(candidate["code"])
            current = by_code.setdefault(
                code,
                {
                    "code": code,
                    "name": candidate["name"],
                    "appearances": 0,
                    "first_trade_date": record["trade_date"],
                    "last_trade_date": record["trade_date"],
                    "latest_rank": candidate["rank"],
                    "latest_score": candidate["score"],
                    "latest_attributes": dict(candidate.get("attributes") or {}),
                    "score_total": 0.0,
                },
            )
            current["appearances"] += 1
            current["name"] = candidate["name"]
            current["last_trade_date"] = record["trade_date"]
            current["latest_rank"] = candidate["rank"]
            current["latest_score"] = candidate["score"]
            current["latest_attributes"] = dict(candidate.get("attributes") or {})
            current["score_total"] += float(candidate["score"])

    candidates = []
    for current in by_code.values():
        appearances = current.pop("appearances")
        score_total = current.pop("score_total")
        candidates.append(
            {
                **current,
                "appearances": appearances,
                "average_score": round(score_total / appearances, 6),
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["appearances"],
            -item["latest_score"],
            item["code"],
        )
    )
    trade_dates = sorted({record["trade_date"] for record in records})
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "import_count": len(records),
        "unique_candidate_count": len(candidates),
        "first_trade_date": trade_dates[0] if trade_dates else None,
        "last_trade_date": trade_dates[-1] if trade_dates else None,
        "candidates": candidates,
    }


def record_candidate_feed(
    pool_path: str,
    feed: CandidateFeed,
    *,
    recorded_at: datetime | None = None,
) -> ObservationResult:
    """Record one validated feed exactly once using an atomic local update."""

    absolute_path = os.path.abspath(os.path.expanduser(pool_path))
    data = load_observation_pool(absolute_path)
    records = data["imports"]
    for record in records:
        if record["source_sha256"] == feed.source_sha256:
            return ObservationResult(
                imported=False,
                source_sha256=feed.source_sha256,
                summary=summarise_observation_pool(data),
            )

    identity = (
        feed.source_system,
        feed.strategy_version,
        feed.market,
        feed.trade_date,
    )
    for record in records:
        if _identity(record) == identity:
            raise CandidateFeedConflictError(
                "同一来源、策略和交易日期已有不同内容，拒绝静默覆盖，请人工复核"
            )

    timestamp = recorded_at or datetime.now().astimezone()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("recorded_at必须包含时区")
    records.append(
        {
            "source_system": feed.source_system,
            "strategy_version": feed.strategy_version,
            "market": feed.market,
            "trade_date": feed.trade_date,
            "generated_at": feed.generated_at,
            "recorded_at": timestamp.isoformat(timespec="seconds"),
            "source_path": feed.source_path,
            "source_sha256": feed.source_sha256,
            "quality": feed.quality,
            "candidates": [candidate.as_dict() for candidate in feed.candidates],
        }
    )
    records.sort(
        key=lambda item: (
            item["trade_date"],
            item["source_system"],
            item["strategy_version"],
            item["generated_at"],
        )
    )
    _atomic_write(absolute_path, data)
    return ObservationResult(
        imported=True,
        source_sha256=feed.source_sha256,
        summary=summarise_observation_pool(data),
    )
