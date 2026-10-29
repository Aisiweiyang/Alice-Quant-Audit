import json
import tempfile
import unittest
from pathlib import Path

from research.run_parameter_sweep import load_plan
from strategy.parameters import ParameterError


class ParameterPlanTests(unittest.TestCase):
    def test_plan_merges_base_and_variant_overrides(self):
        payload = {
            "experiment_name": "demo",
            "base_overrides": {"max_positions": 10},
            "variants": [
                {"name": "one", "overrides": {"max_position_pct": 0.09}},
                {"name": "two", "overrides": {"max_position_pct": 0.07}},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            plan = load_plan(path)
        self.assertEqual(plan["variants"][0]["overrides"]["max_positions"], 10)
        self.assertEqual(plan["variants"][1]["overrides"]["max_position_pct"], 0.07)

    def test_plan_rejects_single_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            path.write_text(json.dumps({"variants": [{"name": "only"}]}), encoding="utf-8")
            with self.assertRaises(ParameterError):
                load_plan(path)


if __name__ == "__main__":
    unittest.main()
