"""Validate and normalise candidate lists produced by external research tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from data.universe import is_main_board_code


SCHEMA_VERSION = 1
MARKET = "CN_A_SHARE"
MAX_CANDIDATES = 100
MAX_FEED_BYTES = 5 * 1024 * 1024
OPTIONAL_NUMERIC_FIELDS = {
    "price": (0, 1_000_000_000),
    "change_pct": (-100, 100),
    "amount": (0, 10_000_000_000_000_000),
    "pe": (-1_000_000, 1_000_000),
    "pb": (-1_000_000, 1_000_000),
}


@dataclass(frozen=True)
class Candidate:
    rank: int
    code: str
    name: str
    score: float
    risk: str | None
    components: dict[str, float]
    reasons: tuple[str, ...]
    attributes: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "rank": self.rank,
            "code": self.code,
            "name": self.name,
            "score": self.score,
            "risk": self.risk,
            "components": self.components,
            "reasons": list(self.reasons),
        }
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        return payload


@dataclass(frozen=True)
class CandidateFeed:
    schema_version: int
    source_system: str
    strategy_version: str
    market: str
    trade_date: str
    generated_at: str
    quality: dict[str, Any]
    candidates: tuple[Candidate, ...]
    source_path: str
    source_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_system": self.source_system,
            "strategy_version": self.strategy_version,
            "market": self.market,
            "trade_date": self.trade_date,
            "generated_at": self.generated_at,
            "candidate_count": len(self.candidates),
            "quality": self.quality,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field}不能为空")
    return text


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}必须是数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}必须是数字") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{field}必须在{minimum:g}到{maximum:g}之间")
    return result


def normalise_candidate_code(value: Any) -> str:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"(?:(sh|sz)\.?)?(\d{6})", text)
    if not match:
        raise ValueError(f"无效A股代码: {value}")
    exchange, digits = match.groups()
    if exchange is None:
        if digits.startswith("6"):
            exchange = "sh"
        elif digits.startswith("0"):
            exchange = "sz"
        else:
            raise ValueError(f"无法推断交易所: {value}")
    code = f"{exchange}.{digits}"
    if not is_main_board_code(code):
        raise ValueError(f"候选不属于沪深主板: {value}")
    return code


def _parse_trade_date(value: Any) -> str:
    text = _required_text(value, "trade_date")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError("trade_date必须是YYYY-MM-DD")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError("trade_date必须是YYYY-MM-DD") from exc


def _parse_generated_at(value: Any) -> str:
    text = _required_text(value, "generated_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at必须是ISO 8601时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at必须包含时区")
    return text


def _validate_quality(
    value: Any,
    minimum_universe_coverage: float,
    minimum_history_coverage: float,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quality必须是JSON对象")
    if value.get("passed") is not True:
        raise ValueError("候选数据质量未通过，Alice拒绝导入")
    universe = _number(
        value.get("universe_coverage"),
        "quality.universe_coverage",
        0,
        1,
    )
    history = _number(
        value.get("history_coverage"),
        "quality.history_coverage",
        0,
        1,
    )
    if universe < minimum_universe_coverage:
        raise ValueError("股票池覆盖率低于Alice导入门槛")
    if history < minimum_history_coverage:
        raise ValueError("历史因子覆盖率低于Alice导入门槛")
    return {
        **value,
        "passed": True,
        "universe_coverage": universe,
        "history_coverage": history,
    }


def _validate_candidate(value: Any) -> Candidate:
    if not isinstance(value, dict):
        raise ValueError("candidates中的每一项必须是JSON对象")
    rank_value = value.get("rank")
    if isinstance(rank_value, bool) or not isinstance(rank_value, int) or rank_value < 1:
        raise ValueError("candidate.rank必须是正整数")
    rank = rank_value

    components_value = value.get("components", {})
    if not isinstance(components_value, dict):
        raise ValueError("candidate.components必须是JSON对象")
    components = {
        _required_text(name, "component.name"): _number(
            component,
            f"component.{name}",
            0,
            100,
        )
        for name, component in components_value.items()
    }
    reasons_value = value.get("reasons", [])
    if not isinstance(reasons_value, list):
        raise ValueError("candidate.reasons必须是列表")
    reasons = tuple(_required_text(reason, "candidate.reason") for reason in reasons_value)
    risk_value = value.get("risk")
    risk = None if risk_value is None else _required_text(risk_value, "candidate.risk")
    attributes = {
        field: _number(value.get(field), f"candidate.{field}", minimum, maximum)
        for field, (minimum, maximum) in OPTIONAL_NUMERIC_FIELDS.items()
        if value.get(field) is not None
    }
    return Candidate(
        rank=rank,
        code=normalise_candidate_code(value.get("code")),
        name=_required_text(value.get("name"), "candidate.name"),
        score=_number(value.get("score"), "candidate.score", 0, 100),
        risk=risk,
        components=components,
        reasons=reasons,
        attributes=attributes,
    )


def load_candidate_feed(
    path: str,
    *,
    expected_trade_date: str | None = None,
    minimum_universe_coverage: float = 0.80,
    minimum_history_coverage: float = 0.80,
) -> CandidateFeed:
    """Load one candidate feed without changing the source file or Alice state."""

    absolute_path = os.path.abspath(os.path.expanduser(path))
    if os.path.getsize(absolute_path) > MAX_FEED_BYTES:
        raise ValueError("候选文件超过5 MB限制")
    with open(absolute_path, "rb") as handle:
        raw = handle.read()
    if len(raw) > MAX_FEED_BYTES:
        raise ValueError("候选文件超过5 MB限制")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("候选文件必须是有效的UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("候选文件顶层必须是JSON对象")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"仅支持schema_version={SCHEMA_VERSION}")
    if payload.get("market") != MARKET:
        raise ValueError(f"仅支持market={MARKET}")

    trade_date = _parse_trade_date(payload.get("trade_date"))
    if expected_trade_date is not None:
        expected = _parse_trade_date(expected_trade_date)
        if trade_date != expected:
            raise ValueError(f"候选日期{trade_date}与预期日期{expected}不一致")
    generated_at = _parse_generated_at(payload.get("generated_at"))
    universe_threshold = _number(
        minimum_universe_coverage,
        "minimum_universe_coverage",
        0,
        1,
    )
    history_threshold = _number(
        minimum_history_coverage,
        "minimum_history_coverage",
        0,
        1,
    )
    quality = _validate_quality(
        payload.get("quality"),
        universe_threshold,
        history_threshold,
    )

    records = payload.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError("candidates必须是非空列表")
    if len(records) > MAX_CANDIDATES:
        raise ValueError(f"候选数量不能超过{MAX_CANDIDATES}")
    candidates = tuple(
        sorted(
            (_validate_candidate(item) for item in records),
            key=lambda item: item.rank,
        )
    )
    codes = [item.code for item in candidates]
    ranks = [item.rank for item in candidates]
    if len(codes) != len(set(codes)):
        raise ValueError("候选股票代码重复")
    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError("候选排名必须从1开始且连续")

    return CandidateFeed(
        schema_version=SCHEMA_VERSION,
        source_system=_required_text(payload.get("source_system"), "source_system"),
        strategy_version=_required_text(payload.get("strategy_version"), "strategy_version"),
        market=MARKET,
        trade_date=trade_date,
        generated_at=generated_at,
        quality=quality,
        candidates=candidates,
        source_path=absolute_path,
        source_sha256=digest,
    )
