"""Compare local Alice state with a read-only paper-broker snapshot."""

from __future__ import annotations

import json
import os
import tempfile


def _issue(issues, category, key, expected, actual):
    issues.append(
        {
            "severity": "BLOCK",
            "category": category,
            "key": str(key),
            "expected": expected,
            "actual": actual,
        }
    )


def _by_id(records, key):
    result = {}
    for record in records:
        record_id = str(record.get(key) or "")
        if not record_id:
            raise ValueError(f"记录缺少{key}")
        if record_id in result:
            raise ValueError(f"记录存在重复{key}: {record_id}")
        result[record_id] = record
    return result


def _compare_positions(expected, actual, issues, price_tolerance):
    expected_positions = expected.holdings
    actual_positions = actual.holdings
    for code in sorted(set(expected_positions) | set(actual_positions)):
        left = expected_positions.get(code)
        right = actual_positions.get(code)
        if left is None or right is None:
            _issue(issues, "POSITION_MISSING", code, left, right)
            continue
        if int(left["shares"]) != int(right["shares"]):
            _issue(
                issues,
                "POSITION_SHARES",
                code,
                int(left["shares"]),
                int(right["shares"]),
            )
        if abs(float(left["avg_cost"]) - float(right["avg_cost"])) > price_tolerance:
            _issue(
                issues,
                "POSITION_AVG_COST",
                code,
                float(left["avg_cost"]),
                float(right["avg_cost"]),
            )


def _local_orders(snapshot):
    records = []
    for order_id, order in snapshot["orders"].items():
        if order.get("status") in {"PLANNED", "REVIEWED"}:
            continue
        records.append(
            {
                "order_id": order_id,
                "status": order["status"],
                "code": order["code"],
                "action": order["action"],
                "ordered_shares": int(order["estimated_shares"]),
                "filled_shares": int(order.get("filled_shares", 0)),
            }
        )
    return records


def _local_fills(snapshot):
    return [
        {key: value for key, value in fill.items() if key != "payload_hash"}
        for fill in snapshot["fills"].values()
    ]


def _compare_records(expected, actual, id_key, fields, category, issues, tolerance=0):
    left_records = _by_id(expected, id_key)
    right_records = _by_id(actual, id_key)
    for record_id in sorted(set(left_records) | set(right_records)):
        left = left_records.get(record_id)
        right = right_records.get(record_id)
        if left is None or right is None:
            _issue(issues, f"{category}_MISSING", record_id, left, right)
            continue
        for field in fields:
            left_value = left.get(field)
            right_value = right.get(field)
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                different = abs(float(left_value) - float(right_value)) > tolerance
            else:
                different = left_value != right_value
            if different:
                _issue(
                    issues,
                    f"{category}_{field.upper()}",
                    record_id,
                    left_value,
                    right_value,
                )


def reconcile_account(
    local_account,
    broker_account,
    ledger_snapshot,
    broker_orders,
    broker_fills,
    cash_tolerance=0.01,
    price_tolerance=0.01,
):
    """Return a blocking reconciliation report without changing either side."""
    issues = []
    if abs(local_account.cash - broker_account.cash) > cash_tolerance:
        _issue(
            issues,
            "CASH",
            "cash",
            local_account.cash,
            broker_account.cash,
        )
    if (
        local_account.total_assets is not None
        and broker_account.total_assets is not None
        and abs(local_account.total_assets - broker_account.total_assets) > cash_tolerance
    ):
        _issue(
            issues,
            "TOTAL_ASSETS",
            "total_assets",
            local_account.total_assets,
            broker_account.total_assets,
        )

    _compare_positions(local_account, broker_account, issues, price_tolerance)
    local_orders = _local_orders(ledger_snapshot)
    local_fills = _local_fills(ledger_snapshot)
    _compare_records(
        local_orders,
        broker_orders,
        "order_id",
        ("status", "code", "action", "ordered_shares", "filled_shares"),
        "ORDER",
        issues,
    )
    _compare_records(
        local_fills,
        broker_fills,
        "fill_id",
        (
            "order_id",
            "trade_date",
            "code",
            "action",
            "shares",
            "price",
            "commission",
            "stamp_tax",
            "other_fee",
        ),
        "FILL",
        issues,
        tolerance=price_tolerance,
    )
    return {
        "schema_version": 1,
        "passed": not issues,
        "blocking_issue_count": len(issues),
        "summary": {
            "local_positions": len(local_account.holdings),
            "broker_positions": len(broker_account.holdings),
            "local_orders": len(local_orders),
            "broker_orders": len(broker_orders),
            "local_fills": len(local_fills),
            "broker_fills": len(broker_fills),
        },
        "issues": issues,
    }


def write_report(path, report):
    """Write a reconciliation report atomically."""
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".reconciliation_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
