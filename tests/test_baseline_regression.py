"""双基线回归对照的测试：锚点文件结构完整性 + 比对核心三条路径（已知答案）。

不跑真实回测；重跑本身由人工/CI按 scripts/baseline_regression.py 执行。
"""

import importlib.util
import json
import os
import unittest

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(sys_path, "tests", "baselines", "real_baselines.json")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "baseline_regression",
        os.path.join(sys_path, "scripts", "baseline_regression.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnchorFileTests(unittest.TestCase):
    def test_real_anchor_file_is_valid_and_complete(self):
        module = _load_module()
        baseline = module.load_baseline(BASELINE_PATH)
        keys = [entry["key"] for entry in baseline["baselines"]]
        self.assertEqual(keys, ["near", "long"])
        # 锚点完整性：确定性核心指标必须全部在列
        required = (
            "total_return_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "total_trades",
            "sharpe_ci_low",
            "sharpe_ci_high",
            "tmc_terminal_return_p5_pct",
            "mc_max_drawdown_pct_p95",
            "data_rating",
        )
        for entry in baseline["baselines"]:
            missing = [k for k in required if k not in entry["anchors"]]
            self.assertEqual(missing, [], f"{entry['key']} 缺锚点: {missing}")
        # 加法不变性锚点：tmc终值p5必须等于已实现收益
        for entry in baseline["baselines"]:
            self.assertEqual(
                entry["anchors"]["tmc_terminal_return_p5_pct"],
                entry["anchors"]["total_return_pct"],
                f"{entry['key']} tmc锚点违背加法不变性",
            )
        # quick分档：near可单独跑，long只在全量跑
        self.assertTrue(baseline["baselines"][0]["quick"])
        self.assertFalse(baseline["baselines"][1]["quick"])


class CompareMetricsTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_identical_passes(self):
        anchors = {"total_return_pct": 45.48, "total_trades": 203, "data_rating": "READY"}
        diffs = self.module.compare_metrics(
            anchors,
            {"total_return_pct": 45.48, "total_trades": 203},
            data_rating="READY",
        )
        self.assertEqual(diffs, [])

    def test_drift_detected(self):
        anchors = {"total_return_pct": 45.48}
        diffs = self.module.compare_metrics(
            anchors,
            {"total_return_pct": 45.49},
        )
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["kind"], "drift")
        self.assertEqual(diffs[0]["expected"], 45.48)
        self.assertEqual(diffs[0]["actual"], 45.49)

    def test_missing_reported(self):
        anchors = {"sharpe_ratio": 1.21}
        diffs = self.module.compare_metrics(anchors, {})
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["kind"], "missing")

    def test_no_silent_tolerance(self):
        # 确定性指标禁止容差：极小漂移也必须红灯。
        diffs = self.module.compare_metrics(
            {"total_return_pct": 45.48},
            {"total_return_pct": 45.4800001},
        )
        self.assertEqual(len(diffs), 1)


class RunRegressionTests(unittest.TestCase):
    def test_quick_mode_skips_long_and_injectable_runner(self):
        module = _load_module()
        baseline = module.load_baseline(BASELINE_PATH)
        near_anchors = baseline["baselines"][0]["anchors"]
        calls = []

        def fake_runner(entry, data_cache):
            calls.append(entry["key"])
            report_dir = os.path.join(temp_dir, f"fake_{entry['key']}")
            os.makedirs(report_dir, exist_ok=True)
            # 假报告=锚点全量复刻（data_rating单列），应逐位一致通过
            metrics = {k: v for k, v in near_anchors.items() if k != "data_rating"}
            with open(os.path.join(report_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with open(
                os.path.join(report_dir, "data_quality.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump({"rating": near_anchors["data_rating"]}, handle)
            return report_dir

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = module.run_regression(
                BASELINE_PATH,
                temp_dir,
                quick=True,
                runner=fake_runner,
            )
            self.assertEqual(calls, ["near"])  # quick只跑near
            self.assertTrue(summary["passed"])
            self.assertTrue(os.path.isfile(summary["report_path"]))

    def test_drift_fails_and_records_diffs(self):
        module = _load_module()

        def failing_runner(entry, data_cache):
            report_dir = os.path.join(temp_dir, f"fake_{entry['key']}")
            os.makedirs(report_dir, exist_ok=True)
            with open(os.path.join(report_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump({"total_return_pct": 45.48, "tmc_terminal_return_p5_pct": 99.0}, handle)
            with open(
                os.path.join(report_dir, "data_quality.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump({"rating": "LIMITED"}, handle)
            return report_dir

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = module.run_regression(
                BASELINE_PATH,
                temp_dir,
                quick=True,
                runner=failing_runner,
            )
            self.assertFalse(summary["passed"])
            diffs = summary["results"][0]["diffs"]
            drifted = {d["key"] for d in diffs}
            self.assertIn("tmc_terminal_return_p5_pct", drifted)
            self.assertIn("data_rating", drifted)


if __name__ == "__main__":
    unittest.main()
