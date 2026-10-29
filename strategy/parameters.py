"""Validated strategy-parameter overrides for isolated research runs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from strategy.config import StrategyConfig


class ParameterError(ValueError):
    """Raised when an experiment tries to use an unsafe or unknown parameter."""


CHOICES = {
    "rebalance_freq": {"daily", "weekly", "monthly"},
    "rebalance_day": {"monday", "tuesday", "wednesday", "thursday", "friday"},
}

RANGES = {
    "top_volume_n": (50, 3_500),
    "consecutive_up_days": (1, 10),
    "max_daily_gain_pct": (0.1, 9.9),
    "min_score": (0.0, 20.0),
    "pe_min": (-1_000.0, 1_000.0),
    "pe_max": (0.1, 2_000.0),
    "pb_min": (-100.0, 100.0),
    "pb_max": (0.1, 500.0),
    "max_positions": (1, 100),
    "max_position_pct": (0.005, 1.0),
    "ma_period": (2, 250),
    "ma_slope_lookback": (0, 250),
    "stock_ma_period": (2, 250),
    "anti_falling_lookback": (1, 60),
    "anti_falling_threshold": (0.001, 0.50),
    "take_profit_pct": (0.005, 2.0),
    "stop_loss_pct": (0.005, 1.0),
    "commission_rate": (0.0, 0.02),
    "stamp_tax_rate": (0.0, 0.02),
    "stamp_tax_rate_before_20230828": (0.0, 0.02),
    "stamp_tax_rate_after_20230828": (0.0, 0.02),
    "slippage": (0.0, 0.05),
    "min_commission": (0.0, 100.0),
}

BLOCKED_PARAMETERS = {
    "benchmark_index": "基准指数暂不允许在普通调参中修改",
    "initial_capital": "初始资金属于回测口径，不是策略参数",
    "emergency_stop": "紧急停止只用于模拟盘风控",
    "max_total_position_pct": "该参数只用于模拟盘风控",
    "max_order_value_pct": "该参数只用于模拟盘风控",
    "max_active_orders": "该参数只用于模拟盘风控",
    "daily_loss_limit_pct": "该参数只用于模拟盘风控",
}


def _snapshot(config: type = StrategyConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        result[name] = deepcopy(value)
    return result


def parse_value(raw: str) -> Any:
    """Parse CLI text with JSON semantics while keeping plain words usable."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_assignment(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ParameterError(f"参数必须写成 名称=值: {text}")
    name, raw = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ParameterError("参数名称不能为空")
    return name, parse_value(raw.strip())


def load_override_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterError(f"无法读取参数文件 {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParameterError("参数文件最外层必须是 JSON 对象")
    if "overrides" in payload:
        payload = payload["overrides"]
    if not isinstance(payload, dict):
        raise ParameterError("overrides 必须是 JSON 对象")
    return payload


def _coerce(name: str, value: Any, expected: Any) -> Any:
    expected_type = type(expected)
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ParameterError(f"{name} 必须是 true 或 false")
    if expected_type is int and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ParameterError(f"{name} 必须是整数")
    if expected_type is float and not isinstance(value, bool):
        if isinstance(value, (int, float)):
            return float(value)
        raise ParameterError(f"{name} 必须是数字")
    if expected_type is str and isinstance(value, str):
        return value.lower() if name in CHOICES else value
    raise ParameterError(f"{name} 的类型不正确，应与默认值 {expected!r} 一致")


def _validate_number(name: str, value: Any) -> None:
    limits = RANGES.get(name)
    if not limits:
        return
    low, high = limits
    if not low <= value <= high:
        raise ParameterError(f"{name} 必须位于 {low} 到 {high} 之间")


def validate_overrides(overrides: dict[str, Any], base: type = StrategyConfig) -> dict[str, Any]:
    """Return normalized overrides without mutating the production config."""
    if not isinstance(overrides, dict):
        raise ParameterError("参数覆盖必须是字典")
    defaults = _snapshot(base)
    normalized: dict[str, Any] = {}
    weights = deepcopy(defaults.get("factor_weights", {}))
    weights_changed = False

    for name, value in overrides.items():
        if name in BLOCKED_PARAMETERS:
            raise ParameterError(f"{name}: {BLOCKED_PARAMETERS[name]}")
        if name == "factor_weights":
            if not isinstance(value, dict):
                raise ParameterError("factor_weights 必须是 JSON 对象")
            for factor_name, factor_value in value.items():
                if factor_name not in weights:
                    raise ParameterError(f"未知因子权重: {factor_name}")
                if isinstance(factor_value, bool) or not isinstance(factor_value, (int, float)):
                    raise ParameterError(f"factor_weights.{factor_name} 必须是数字")
                if not 0.0 <= float(factor_value) <= 1.0:
                    raise ParameterError(f"factor_weights.{factor_name} 必须位于 0 到 1 之间")
                weights[factor_name] = float(factor_value)
                weights_changed = True
            continue
        if name.startswith("factor_weights."):
            factor_name = name.split(".", 1)[1]
            if factor_name not in weights:
                raise ParameterError(f"未知因子权重: {factor_name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParameterError(f"{name} 必须是数字")
            if not 0.0 <= float(value) <= 1.0:
                raise ParameterError(f"{name} 必须位于 0 到 1 之间")
            weights[factor_name] = float(value)
            weights_changed = True
            continue
        if name not in defaults:
            raise ParameterError(f"未知或不可调参数: {name}")
        coerced = _coerce(name, value, defaults[name])
        if name in CHOICES and coerced not in CHOICES[name]:
            choices = "、".join(sorted(CHOICES[name]))
            raise ParameterError(f"{name} 只能是: {choices}")
        _validate_number(name, coerced)
        normalized[name] = coerced

    if weights_changed:
        if sum(weights.values()) <= 0:
            raise ParameterError("因子权重总和必须大于0")
        normalized["factor_weights"] = weights

    effective = {**defaults, **normalized}
    if effective["pe_min"] >= effective["pe_max"]:
        raise ParameterError("pe_min 必须小于 pe_max")
    if effective["pb_min"] >= effective["pb_max"]:
        raise ParameterError("pb_min 必须小于 pb_max")
    return normalized


def apply_overrides(overrides: dict[str, Any], config: type = StrategyConfig) -> dict[str, Any]:
    """Apply validated overrides to the process-local config class."""
    normalized = validate_overrides(overrides, config)
    for name, value in normalized.items():
        setattr(config, name, deepcopy(value))
    return normalized


def combine_overrides(
    file_overrides: dict[str, Any] | None,
    assignments: list[str] | None,
) -> dict[str, Any]:
    combined = deepcopy(file_overrides or {})
    for assignment in assignments or []:
        name, value = parse_assignment(assignment)
        if name.startswith("factor_weights."):
            combined[name] = value
        elif name == "factor_weights" and isinstance(value, dict):
            existing = combined.get("factor_weights")
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(value)
            combined[name] = merged
        else:
            combined[name] = value
    return combined
