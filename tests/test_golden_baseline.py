"""黄金基线回归测试（参考backtesting.py金文件测试实践）。

固定 demo_cache + 固定区间 + 固定参数跑完整full管线，与 tests/golden/
golden_full.json 逐一比对：浮点字段1e-9容差，其余精确相等。
未来任何回测数值口径的无意改动会在此立即红灯；有意的口径变更须用
scripts/update_golden.py --confirm 重新生成并在提交信息声明
"GOLDEN UPDATE: <原因>"（一次提交只允许一个口径变更）。
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

sys.modules.setdefault("baostock", __import__("types").ModuleType("baostock"))
sys.path.insert(0, ".")

from tests.golden_builder import GOLDEN_PATH, build_golden, diff_golden


def _load_update_golden_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "update_golden.py",
    )
    spec = importlib.util.spec_from_file_location("update_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GoldenBaselineTests(unittest.TestCase):
    def test_golden_file_exists(self):
        self.assertTrue(
            os.path.isfile(GOLDEN_PATH),
            f"黄金基线文件缺失：{GOLDEN_PATH}；请运行 scripts/update_golden.py 首次生成",
        )

    def test_golden_baseline_reproduces_exactly(self):
        """演示数据上完整重跑full管线，核心结果必须与黄金基线一致。"""
        with open(GOLDEN_PATH, "r", encoding="utf-8") as handle:
            golden = json.load(handle)
        with tempfile.TemporaryDirectory() as temp_dir:
            actual = build_golden(temp_dir)
        differences = diff_golden(golden, actual)
        self.assertEqual(
            differences,
            [],
            "回测核心结果与黄金基线不一致——若为无意的口径改动请回溯；"
            "若为有意变更，用 scripts/update_golden.py --confirm 重生成并"
            "在提交信息声明 GOLDEN UPDATE: <原因>",
        )

    def test_golden_contains_required_guard_rails(self):
        """黄金基线必须覆盖提示词要求的核心锚点（评分/收益/Sharpe/CI端点/
        交易数/各图存在性/manifest结构）。"""
        with open(GOLDEN_PATH, "r", encoding="utf-8") as handle:
            golden = json.load(handle)
        self.assertIsNotNone(golden["score"]["score"])
        self.assertIsNotNone(golden["metrics"]["total_return_pct"])
        self.assertIsNotNone(golden["metrics"]["sharpe_ratio"])
        self.assertIsNotNone(golden["metrics"]["sharpe_ci_low"])
        self.assertIsNotNone(golden["metrics"]["sharpe_ci_high"])
        self.assertGreater(golden["metrics"]["total_trades"], 0)
        self.assertGreater(len(golden["charts_written"]), 0)
        self.assertIn("index.html", golden["manifest_output_names"])
        self.assertIn("metrics.json", golden["manifest_output_names"])
        # manifest自身不入清单（自引用哈希无法闭合，2026-09-13口径修正）。
        self.assertNotIn("manifest.json", golden["manifest_output_names"])
        # 明细最深回撤与max_drawdown_pct的等价性锚点也固化在黄金文件里。
        deepest = min(p["depth_pct"] for p in golden["drawdown_periods"])
        self.assertEqual(deepest, golden["metrics"]["max_drawdown_pct"])


class DiffFunctionTests(unittest.TestCase):
    def test_float_within_tolerance_passes(self):
        self.assertEqual(diff_golden({"a": 1.0}, {"a": 1.0 + 5e-10}), [])

    def test_value_changes_are_reported(self):
        diffs = diff_golden({"a": 1.0, "b": {"c": [1, 2]}}, {"a": 2.0, "b": {"c": [1, 3]}})
        self.assertEqual(len(diffs), 2)
        self.assertIn("$.a", diffs[0])

    def test_structure_changes_are_reported(self):
        diffs = diff_golden({"a": 1}, {"a": 1, "b": 2})
        self.assertEqual(len(diffs), 1)
        self.assertIn("黄金基线缺失该键", diffs[0])
        diffs = diff_golden({"a": [1, 2]}, {"a": [1]})
        self.assertEqual(len(diffs), 1)
        self.assertIn("长度不同", diffs[0])

    def test_bool_not_treated_as_float(self):
        diffs = diff_golden({"flag": True}, {"flag": 1})
        self.assertEqual(len(diffs), 1)


class UpdateGoldenScriptTests(unittest.TestCase):
    def _run_main(self, module, golden_path, argv):
        """在受控argv/黄金路径下执行update_golden.main()，返回退出码。"""
        original_path = module.GOLDEN_PATH
        original_argv = sys.argv
        module.GOLDEN_PATH = golden_path
        sys.argv = argv
        try:
            return module.main()
        finally:
            module.GOLDEN_PATH = original_path
            sys.argv = original_argv

    def test_no_silent_overwrite_without_confirm(self):
        """已有黄金文件时，不带--confirm必须拒绝写入（禁止静默覆盖）。"""
        module = _load_update_golden_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = os.path.join(temp_dir, "golden_full.json")
            with open(golden_path, "w", encoding="utf-8") as handle:
                handle.write('{"keep": "old"}')
            module.build_golden = lambda output_root: {"keep": "old", "new": 1}
            exit_code = self._run_main(module, golden_path, ["update_golden.py"])
            self.assertEqual(exit_code, 1, "缺省必须只读不覆盖")
            with open(golden_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"keep": "old"})

    def test_confirm_overwrites_on_real_change(self):
        module = _load_update_golden_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = os.path.join(temp_dir, "golden_full.json")
            with open(golden_path, "w", encoding="utf-8") as handle:
                handle.write('{"keep": "old"}')
            module.build_golden = lambda output_root: {"keep": "old", "new": 1}
            exit_code = self._run_main(module, golden_path, ["update_golden.py", "--confirm"])
            self.assertEqual(exit_code, 0)
            with open(golden_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["new"], 1)

    def test_identical_result_needs_no_confirm(self):
        """与现有黄金完全一致时不覆盖文件且退出码0。"""
        module = _load_update_golden_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = os.path.join(temp_dir, "golden_full.json")
            with open(golden_path, "w", encoding="utf-8") as handle:
                handle.write('{"keep": "old"}')
            module.build_golden = lambda output_root: {"keep": "old"}
            exit_code = self._run_main(module, golden_path, ["update_golden.py"])
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
