"""第二内置策略 cross_sectional_alpha158 的商业管线接入测试。"""

import os
import tempfile
import unittest

import pandas as pd

from commercial.config import (
    CommercialRequest,
    ParameterError,
    build_factor_strategy_config,
)
from commercial.pipeline import run_full_analysis


def make_data(stocks=8, days=140, seed=11):
    """与tests/test_commercial_mvp同构的主板合成数据（含pe/pb/停牌字段）。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    data = {}
    for index in range(stocks):
        code = f"sh.60{index:04d}" if index % 2 == 0 else f"sz.00{index:04d}"
        steps = rng.normal(0.0003, 0.012, size=days)
        closes = 15 * np.exp(np.cumsum(steps))
        preclose = np.concatenate([[closes[0]], closes[:-1]])
        volume = rng.lognormal(13, 0.3, size=days).round(0)
        data[code] = pd.DataFrame(
            {
                "date": dates,
                "open": (closes * 1.001).round(3),
                "high": (closes * 1.012).round(3),
                "low": (closes * 0.99).round(3),
                "close": closes.round(3),
                "volume": volume,
                "amount": (volume * closes).round(0),
                "preclose": preclose.round(3),
                "pctChg": ((closes / preclose - 1) * 100).round(4),
                "tradestatus": 1,
                "isST": 0,
                "pe": 20.0,
                "pb": 2.0,
            }
        )
    data["sh.000300"] = pd.DataFrame(
        {
            "date": dates,
            "close": (1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=days)))).round(2),
            "open": 1000.0,
        }
    )
    return data


def write_cache(data, directory, name="market.pkl"):
    from data.cache_store import build_metadata, save_market_cache

    metadata = build_metadata(
        data,
        benchmark="sh.000300",
        adjust="2",
        universe_mode="file",
        source="unittest",
    )
    path = os.path.join(directory, name)
    save_market_cache(path, data, metadata)
    return path


def make_request(cache, run_stress=False, run_rolling_oos=False, **overrides):
    options = {"output_root": cache and os.path.dirname(cache)}
    options.update(overrides)
    request = CommercialRequest(
        cache=cache,
        start="2025-02-01",
        end="2025-05-30",
        strategy="cross_sectional_alpha158",
        run_stress=run_stress,
        run_rolling_oos=run_rolling_oos,
        quiet=True,
        **options,
    )
    request.validate()
    return request


class StrategyParamWhitelistTests(unittest.TestCase):
    def test_unknown_param_rejected(self):
        request = make_request("dummy.pkl")
        request.strategy_params = {"hack_code": "x"}
        with self.assertRaises(ParameterError):
            request.validate()

    def test_out_of_range_rejected(self):
        request = make_request("dummy.pkl")
        request.strategy_params = {"minimum_score": "10000"}
        with self.assertRaises(ParameterError):
            request.validate()

    def test_params_only_allowed_for_cross_sectional(self):
        request = make_request("dummy.pkl")
        request.strategy = "classic_trend"
        request.strategy_params = {"minimum_score": "60"}
        with self.assertRaises(ParameterError):
            request.validate()

    def test_valid_params_cast_and_accepted(self):
        request = make_request("dummy.pkl")
        request.strategy_params = {"minimum_score": "60", "exit_rank": "30"}
        request.validate()
        self.assertEqual(request.strategy_params["minimum_score"], 60.0)
        self.assertEqual(request.strategy_params["exit_rank"], 30)

    def test_build_factor_config_maps_shared_params(self):
        request = make_request("dummy.pkl", initial_capital=250_000.0, max_positions=5)
        request.strategy_params = {"minimum_score": "50"}
        request.validate()
        config = build_factor_strategy_config(request)
        self.assertEqual(config.initial_capital, 250_000.0)
        self.assertEqual(config.max_positions, 5)
        self.assertEqual(config.minimum_score, 50.0)
        self.assertEqual(config.market_ma_period, 20)
        self.assertEqual(config.benchmark_index, "sh.000300")
        self.assertTrue(config.stamp_tax_rate_before_20230828 >= request.stamp_tax_rate)


class CrossSectionalEndToEndTests(unittest.TestCase):
    def test_full_analysis_produces_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = write_cache(make_data(), tmp)
            request = make_request(cache)
            analysis = run_full_analysis(request, report_id="csx_e2e_1")
            self.assertFalse(analysis.get("blocked"))
            metrics = (analysis.get("base") or {}).get("metrics") or {}
            self.assertIn("total_return_pct", metrics)
            self.assertGreater(int(metrics.get("sharpe_observations") or 0), 0)
            feature_meta = analysis.get("feature_meta") or {}
            self.assertGreater(int(feature_meta.get("signal_sections") or 0), 0)
            self.assertIsNotNone(analysis.get("score"))
            html_path = os.path.join(analysis["report_dir"], "index.html")
            self.assertTrue(os.path.isfile(html_path))
            with open(html_path, encoding="utf-8") as handle:
                document = handle.read()
            self.assertIn("仅供研究，不构成投资建议", document)
            self.assertIn("cross_sectional_alpha158", document)
            # 执行真实性审计来自因子回测引擎。
            execution = getattr((analysis["base"])["portfolio"], "execution_audit", None)
            self.assertIsNotNone(execution)
            self.assertIn("final_unsold", execution)
            # 分段一致性使用同一策略回测，输出结论或如实数据不足。
            consistency = analysis.get("segment_consistency") or {}
            self.assertIn(consistency.get("status"), {"ok", "insufficient_data"})

    def test_factor_strategy_stress_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = write_cache(make_data(), tmp)
            request = make_request(cache, run_stress=True)
            analysis = run_full_analysis(request, report_id="csx_stress_1")
            self.assertFalse(analysis.get("blocked"))
            stress = analysis.get("stress_summary") or {}
            self.assertEqual(stress.get("status"), "ok", "截面策略应有可用的适配版压力场景")
            ranking = stress.get("ranking") or []
            self.assertTrue(ranking, "应产出与基准的差异排序")
            joined = "\n".join(analysis.get("notes") or [])
            self.assertIn("适配版压力场景", joined)
            score = analysis.get("score") or {}
            self.assertNotIn(
                "cost_sensitivity",
                score.get("insufficient_dimensions") or [],
                "截面压力场景补齐后成本敏感性维度不应再缺失",
            )
            frame = analysis.get("stress_frame")
            self.assertIsNotNone(frame)
            self.assertIn("ablation_no_momentum", set(frame["scenario"]))
            self.assertNotIn("buy_delay_1d", set(frame["scenario"]))

    def test_insufficient_history_blocks_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = write_cache(make_data(days=40), tmp)
            request = make_request(cache)
            analysis = run_full_analysis(request, report_id="csx_e2e_3")
            self.assertTrue(
                analysis.get("blocked"),
                "不足65日历史时截面策略应如实阻断，不产出虚假表现数据",
            )


if __name__ == "__main__":
    unittest.main()
