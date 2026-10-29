import json
import tempfile
import unittest
from pathlib import Path

from strategy.config import StrategyConfig
from strategy.parameters import (
    ParameterError,
    combine_overrides,
    load_override_file,
    parse_assignment,
    validate_overrides,
)


class ParameterTests(unittest.TestCase):
    def test_cli_assignment_parses_json_types(self):
        self.assertEqual(parse_assignment("max_positions=12"), ("max_positions", 12))
        self.assertEqual(parse_assignment("use_market_timing=false"), ("use_market_timing", False))
        self.assertEqual(parse_assignment("rebalance_freq=monthly"), ("rebalance_freq", "monthly"))

    def test_overrides_are_validated_without_mutating_baseline(self):
        before = StrategyConfig.factor_weights.copy()
        result = validate_overrides(
            {
                "max_positions": 12,
                "factor_weights.volatility": 0.15,
            }
        )
        self.assertEqual(result["max_positions"], 12)
        self.assertEqual(result["factor_weights"]["volatility"], 0.15)
        self.assertEqual(StrategyConfig.factor_weights, before)

    def test_unknown_blocked_and_out_of_range_parameters_fail(self):
        for overrides in (
            {"does_not_exist": 1},
            {"initial_capital": 1_000_000},
            {"max_position_pct": 2.0},
            {"factor_weights.unknown": 0.1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ParameterError):
                validate_overrides(overrides)

    def test_json_file_and_cli_overrides_combine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps({"max_positions": 10}), encoding="utf-8")
            loaded = load_override_file(path)
        combined = combine_overrides(loaded, ["max_position_pct=0.08"])
        self.assertEqual(combined, {"max_positions": 10, "max_position_pct": 0.08})


if __name__ == "__main__":
    unittest.main()
