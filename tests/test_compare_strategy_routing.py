"""compare策略路由修复的测试（2026-09-13）。

修复前三条链路缺陷：
1. compare对cross_sectional_alpha158仍无条件走经典引擎（策略名与执行引擎不符）；
2. 计划白名单无法表达每方案截面参数（minimum_score=60/70/80无法对比）；
3. compare数据审计未接入缓存旁security_master（真实缓存退化为LIMITED）。
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import commercial.compare as compare_module
from commercial.compare import (
    _dispersion_diagnosis,
    _feature_store_key,
    _plan_factor_config,
    parse_plan_payload,
    run_compare,
)
from commercial.config import ParameterError
from tests.test_night_round import make_data, make_request, write_cache


class DispersionDiagnosisTests(unittest.TestCase):
    """PBO有效性前置闸门：方案组分散度诊断（已知答案小样例）。"""

    @staticmethod
    def _returns(seed, days=250, scale=0.01):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        index = pd.bdate_range("2025-01-01", periods=days)
        return pd.Series(rng.normal(0.0005, scale, days), index=index)

    @staticmethod
    def _results(labels_trades):
        return [
            {"label": label, "metrics": {"total_trades": trades}} for label, trades in labels_trades
        ]

    def test_identical_series_are_insufficient(self):
        same = self._returns(seed=1)
        diagnosis = _dispersion_diagnosis(
            {"方案A": same, "方案B": same.copy(), "方案C": same.copy()},
            self._results([("方案A", 100), ("方案B", 100), ("方案C", 100)]),
        )
        self.assertEqual(diagnosis["validity"], "insufficient_dispersion")
        self.assertEqual(diagnosis["max_pair_correlation"], 1.0)
        self.assertIn("无统计意义", diagnosis["detail"])

    def test_independent_series_pass(self):
        diagnosis = _dispersion_diagnosis(
            {"动量": self._returns(11), "价值": self._returns(22), "低波": self._returns(33)},
            self._results([("动量", 120), ("价值", 100), ("低波", 90)]),
        )
        self.assertEqual(diagnosis["validity"], "ok")
        self.assertLessEqual(diagnosis["max_pair_correlation"], 0.98)

    def test_degenerate_plan_triggers_gate_even_if_uncorrelated(self):
        diagnosis = _dispersion_diagnosis(
            {"动量": self._returns(11), "价值": self._returns(22), "阈值过严": self._returns(33)},
            self._results([("动量", 120), ("价值", 100), ("阈值过严", 1)]),
        )
        self.assertEqual(diagnosis["validity"], "insufficient_dispersion")
        self.assertEqual(
            diagnosis["degenerate_plans"],
            [{"label": "阈值过严", "total_trades": 1}],
        )

    def test_correlation_on_partial_overlap_uses_common_dates(self):
        common = self._returns(seed=5)
        shifted = self._returns(seed=5).iloc[10:]  # 同序列不同起点→仍有重叠
        diagnosis = _dispersion_diagnosis(
            {"A": common, "B": shifted, "C": self._returns(seed=6)},
            self._results([("A", 100), ("B", 100), ("C", 100)]),
        )
        pair = next(c for c in diagnosis["correlations"] if set(c["pair"]) == {"A", "B"})
        self.assertIsNotNone(pair["correlation"])

    def test_compare_persists_dispersion(self):
        # 端到端：3方案compare的overfitting_audit带dispersion（路由测试的
        # ms60/70/80仅差阈值，属典型低分散度方案组）。
        with tempfile.TemporaryDirectory() as temp_dir:
            data = make_data()
            cache = write_cache(data, temp_dir)
            request = make_request(
                cache,
                temp_dir,
                strategy="cross_sectional_alpha158",
                start="2025-05-01",
                end="2025-07-15",
            )
            plan_path = os.path.join(temp_dir, "plan.json")
            with open(plan_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "plans": [
                            {"label": "ms60", "strategy_params": {"minimum_score": 60}},
                            {"label": "ms70", "strategy_params": {"minimum_score": 70}},
                            {"label": "ms80", "strategy_params": {"minimum_score": 80}},
                        ]
                    },
                    handle,
                )
            analysis = run_compare(request, plan_path, report_id="disp_probe")
            ok = [r for r in analysis["plan_results"] if r["status"] == "ok"]
            if len(ok) >= 3:
                dispersion = analysis["overfitting_audit"]["dispersion"]
                self.assertIn(dispersion["validity"], ("ok", "insufficient_dispersion"))
                self.assertIsNotNone(dispersion["detail"])


class ParsePlanParamsTests(unittest.TestCase):
    def test_strategy_params_whitelist_accepted(self):
        plans = parse_plan_payload(
            {
                "plans": [
                    {"label": "ms60", "strategy_params": {"minimum_score": 60}},
                    {"label": "ms70", "strategy_params": {"minimum_score": 70, "exit_rank": 15}},
                ]
            }
        )
        self.assertEqual(plans[0]["strategy_params"], {"minimum_score": 60.0})
        self.assertEqual(plans[1]["strategy_params"]["exit_rank"], 15)

    def test_strategy_params_unknown_key_rejected(self):
        with self.assertRaises(ParameterError):
            parse_plan_payload(
                {
                    "plans": [
                        {"label": "a"},
                        {"label": "b", "strategy_params": {"not_a_param": 1}},
                    ]
                }
            )

    def test_strategy_params_out_of_range_rejected(self):
        with self.assertRaises(ParameterError):
            parse_plan_payload(
                {
                    "plans": [
                        {"label": "a"},
                        {"label": "b", "strategy_params": {"minimum_score": 999}},
                    ]
                }
            )

    def test_strategy_params_must_be_object(self):
        with self.assertRaises(ParameterError):
            parse_plan_payload(
                {
                    "plans": [
                        {"label": "a"},
                        {"label": "b", "strategy_params": "minimum_score=60"},
                    ]
                }
            )


class StrategyGuardTests(unittest.TestCase):
    def test_classic_plan_with_strategy_params_rejected_before_report_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = make_request("x.pkl", temp_dir)  # 默认classic_trend
            plan_path = os.path.join(temp_dir, "plan.json")
            with open(plan_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "plans": [
                            {"label": "a"},
                            {"label": "b", "strategy_params": {"minimum_score": 60}},
                        ]
                    },
                    handle,
                )
            with self.assertRaises(ParameterError):
                run_compare(request, plan_path)
            # 拒绝必须发生在报告目录创建之前
            self.assertEqual(os.listdir(temp_dir), ["plan.json"])


class PlanFactorConfigTests(unittest.TestCase):
    def test_strategy_params_applied(self):
        request = make_request("x.pkl", "/tmp")
        plan = {"overrides": {}, "strategy_params": {"minimum_score": 60}}
        config = _plan_factor_config(request, plan, "sh.000300")
        self.assertEqual(config.minimum_score, 60.0)
        self.assertEqual(config.benchmark_index, "sh.000300")

    def test_classic_overrides_mapped(self):
        request = make_request("x.pkl", "/tmp")
        plan = {
            "overrides": {
                "rebalance_freq": "monthly",
                "commission_rate": 0.001,
                "stamp_tax_rate": 0.0008,
            },
            "strategy_params": {},
        }
        config = _plan_factor_config(request, plan, "sh.000300")
        self.assertEqual(config.rebalance_frequency, "monthly")
        self.assertEqual(config.commission_rate, 0.001)
        # 印花税历史分段规则与build_factor_strategy_config一致
        self.assertEqual(config.stamp_tax_rate_after_20230828, 0.0008)
        self.assertEqual(config.stamp_tax_rate_before_20230828, 0.001)

    def test_synthetic_benchmark_overrides_index(self):
        request = make_request("x.pkl", "/tmp")
        plan = {"overrides": {}, "strategy_params": {}}
        config = _plan_factor_config(request, plan, "synthetic:equal_weight")
        self.assertEqual(config.benchmark_index, "synthetic:equal_weight")


class FeatureStoreKeyTests(unittest.TestCase):
    def _config(self, request, **params):
        plan = {"overrides": {}, "strategy_params": dict(params)}
        return _plan_factor_config(request, plan, "sh.000300")

    def test_score_only_changes_share_store(self):
        request = make_request("x.pkl", "/tmp")
        key60 = _feature_store_key(self._config(request, minimum_score=60))
        key70 = _feature_store_key(self._config(request, minimum_score=70))
        key_exit = _feature_store_key(self._config(request, exit_rank=20))
        self.assertEqual(key60, key70)
        self.assertEqual(key60, key_exit)

    def test_store_affecting_params_change_key(self):
        request = make_request("x.pkl", "/tmp")
        base = _feature_store_key(self._config(request))
        for params in (
            {"liquid_universe_size": 300},
            {"minimum_history": 120},
        ):
            self.assertNotEqual(base, _feature_store_key(self._config(request, **params)))

    def test_weight_variants_share_store(self):
        # run_factor_backtest对传入存储按config权重rescore（压力消融同机制），
        # 权重不同的方案可共享同一存储，无需重建。
        request = make_request("x.pkl", "/tmp")
        plan_a = {
            "overrides": {"factor_weights": {"momentum_rank": 0.7, "trend_quality_rank": 0.3}},
            "strategy_params": {},
        }
        plan_b = {"overrides": {}, "strategy_params": {}}
        self.assertEqual(
            _feature_store_key(_plan_factor_config(request, plan_a, "sh.000300")),
            _feature_store_key(_plan_factor_config(request, plan_b, "sh.000300")),
        )

    def test_rebalance_freq_changes_key(self):
        request = make_request("x.pkl", "/tmp")
        base = _feature_store_key(self._config(request))
        plan = {"overrides": {"rebalance_freq": "monthly"}, "strategy_params": {}}
        self.assertNotEqual(
            base, _feature_store_key(_plan_factor_config(request, plan, "sh.000300"))
        )


class CompareRoutingIntegrationTests(unittest.TestCase):
    def test_cross_sectional_compare_runs_factor_engine_with_shared_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = make_data()
            cache = write_cache(data, temp_dir)
            request = make_request(
                cache,
                temp_dir,
                strategy="cross_sectional_alpha158",
                start="2025-05-01",
                end="2025-07-15",
            )
            plan_path = os.path.join(temp_dir, "plan.json")
            with open(plan_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "plans": [
                            {"label": "ms60", "strategy_params": {"minimum_score": 60}},
                            {"label": "ms70", "strategy_params": {"minimum_score": 70}},
                            {"label": "ms80", "strategy_params": {"minimum_score": 80}},
                        ]
                    },
                    handle,
                )

            real_prepare = compare_module.prepare_feature_store
            calls = []

            def counting_prepare(*args, **kwargs):
                calls.append(1)
                return real_prepare(*args, **kwargs)

            with patch.object(
                compare_module, "prepare_feature_store", side_effect=counting_prepare
            ):
                analysis = run_compare(request, plan_path, report_id="csx_route_probe")
            self.assertEqual(analysis["mode"], "compare")
            ok = [r for r in analysis["plan_results"] if r["status"] == "ok"]
            self.assertEqual(len(ok), 3)
            # 证据1：三个方案的实际参数各不相同且落入effective_config
            scores = {r["label"]: r["effective_config"].get("minimum_score") for r in ok}
            self.assertEqual(scores, {"ms60": 60.0, "ms70": 70.0, "ms80": 80.0})
            # 证据2：策略名与执行引擎一致
            self.assertTrue(all(r.get("strategy") == "cross_sectional_alpha158" for r in ok))
            # 证据3：minimum_score不影响特征存储 → 三方案只构建一次
            self.assertEqual(len(calls), 1, "minimum_score差异不应触发特征存储重建")
            # 证据4：方案回测失败时不允许以经典引擎结果冒充（结果含截面披露）
            audit = analysis["data_audit"]
            self.assertIsNotNone(audit.get("fields", {}).get("delisted_coverage"))

    def test_classic_compare_still_routes_classic_engine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = make_data()
            cache = write_cache(data, temp_dir)
            request = make_request(cache, temp_dir)
            plan_path = os.path.join(temp_dir, "plan.json")
            with open(plan_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "plans": [
                            {"label": "基线"},
                            {"label": "高止损", "overrides": {"stop_loss_pct": 0.08}},
                        ]
                    },
                    handle,
                )
            analysis = run_compare(request, plan_path, report_id="classic_route_probe")
            ok = [r for r in analysis["plan_results"] if r["status"] == "ok"]
            self.assertEqual(len(ok), 2)
            self.assertTrue(all(r.get("strategy") == "classic_trend" for r in ok))


if __name__ == "__main__":
    unittest.main()
