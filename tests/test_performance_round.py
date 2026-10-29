"""性能优化测试：选股快速路径等价性、执行行等价性、分段开关。

红线：所有性能优化不得改变回测数值口径；选股快速路径的候选代码与评分
必须与原实现完全一致。因子预计算(compute_all_batch)的等价性测试见
tests/test_perf_sprint.py。
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from strategy.config import StrategyConfig
from strategy.factors import compute_all, compute_all_batch
from strategy.selection import select_candidates


def random_frame(rng, rows, with_nan=False):
    dates = pd.bdate_range("2025-01-01", periods=rows)
    steps = rng.normal(0.0002, 0.02, size=rows)
    closes = np.maximum(5.0 * np.exp(np.cumsum(steps)), 0.5)
    volume = rng.lognormal(13, 0.5, size=rows).round(0)
    high = closes * (1 + np.abs(rng.normal(0, 0.01, rows)))
    low = closes * (1 - np.abs(rng.normal(0, 0.01, rows)))
    open_ = closes * (1 + rng.normal(0, 0.005, rows))
    preclose = np.concatenate([[closes[0]], closes[:-1]])
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(3),
            "high": high.round(3),
            "low": low.round(3),
            "close": closes.round(3),
            "volume": volume,
            "amount": (volume * closes).round(0),
            "preclose": preclose.round(3),
            "pctChg": ((closes / preclose - 1) * 100).round(4),
        }
    )
    frame["tradestatus"] = 1
    frame["isST"] = 0
    frame["pe"] = 20.0
    frame["pb"] = 2.0
    if with_nan:
        frame.loc[3, "volume"] = 0.0
        frame.loc[5, "volume"] = np.nan
        frame.loc[7, "preclose"] = np.nan
        frame.loc[11, "pe"] = np.nan
    return frame


def normalise_like_cache(frame):
    """复现 cache_store._normalise_frame 的形态（DatetimeIndex、规范化）。"""
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    clean = clean.dropna(subset=["date"]).sort_values("date")
    clean = clean.drop_duplicates(subset=["date"], keep="last")
    clean.set_index("date", drop=False, inplace=True)
    clean.index.name = "_trade_date"
    return clean


class BatchFactorEdgeTests(unittest.TestCase):
    """对 compute_all_batch 的补充边界测试（与逐信号 compute_all 对照）。"""

    def test_constant_ohlc_adx_absent_everywhere(self):
        """整段常价（TR=0 → ATR=0）时 ADX 应与旧实现一样在所有位置缺席。"""
        rows = 120
        dates = pd.bdate_range("2025-01-01", periods=rows)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1000.0,
                "amount": 10000.0,
                "pctChg": 0.0,
            }
        )
        positions = [59, 60, 80, rows - 1]
        batched = compute_all_batch(frame, positions, StrategyConfig)
        for pos in positions:
            legacy = compute_all(frame.iloc[: pos + 1], StrategyConfig) or {}
            self.assertNotIn("adx", legacy, "常价下旧实现不应有ADX")
            self.assertNotIn("adx", batched[pos] or {}, f"位置{pos}新实现不应有ADX")

    def test_flat_span_with_zero_volume_rows(self):
        rng = np.random.default_rng(77)
        frame = random_frame(rng, 200)
        frame.loc[frame.index[100:110], "close"] = frame.loc[frame.index[100], "close"]
        frame.loc[frame.index[100:110], "high"] = frame.loc[frame.index[100], "high"]
        frame.loc[frame.index[100:110], "low"] = frame.loc[frame.index[100], "low"]
        frame.loc[frame.index[100:110], "volume"] = 0
        for pos in (109, 110, 150, len(frame) - 1):
            legacy = compute_all(frame.iloc[: pos + 1], StrategyConfig) or {}
            batched = compute_all_batch(frame, [pos], StrategyConfig)[pos] or {}
            for name in ("adx", "volatility", "vol_price_corr"):
                self.assertEqual(name in legacy, name in batched, f"{name}@{pos}")
                if name in legacy:
                    self.assertLessEqual(abs(float(legacy[name]) - float(batched[name])), 1e-6)


class FastIndexRowEquivalenceTests(unittest.TestCase):
    """快速索引的行查找与执行行字典必须与 _row_on/原 Series 语义一致。"""

    def test_row_lookup_matches_row_on(self):
        from engine.backtest import _row_on
        from engine.fast_index import build_stock_index

        rng = np.random.default_rng(41)
        frame = normalise_like_cache(random_frame(rng, 200))
        index = build_stock_index(frame)
        self.assertIsNotNone(index)
        for row_no in range(0, len(frame), 17):
            date = frame["date"].iloc[row_no]
            expected = _row_on(frame, date)
            actual = index.row_at(index.position_of[date.value])
            for column in (
                "open",
                "close",
                "volume",
                "preclose",
                "pctChg",
                "tradestatus",
                "isST",
                "high",
                "low",
            ):
                self.assertEqual(
                    float(expected.get(column, float("nan")))
                    if pd.notna(expected.get(column, float("nan")))
                    else expected.get(column),
                    float(actual.get(column))
                    if actual.get(column) is not None and pd.notna(actual.get(column))
                    else actual.get(column),
                    f"{date} 列{column}不一致",
                )
        # 缺失日期返回None
        missing = pd.Timestamp("2030-01-01")
        self.assertIsNone(_row_on(frame, missing))
        self.assertIsNone(index.position_of.get(missing.value))

    def test_execution_row_matches_series_for_order_gate(self):
        from engine.execution import order_is_executable, closing_sale_is_executable
        from engine.fast_index import build_stock_index

        rng = np.random.default_rng(42)
        frame = normalise_like_cache(random_frame(rng, 200, with_nan=True))
        index = build_stock_index(frame)
        for row_no in range(50, len(frame), 23):
            date = frame["date"].iloc[row_no]
            series_row = frame.loc[date]
            dict_row = index.execution_row(index.position_of[date.value])
            code = "sh.600000"
            self.assertEqual(
                order_is_executable(code, series_row, "BUY"),
                order_is_executable(code, dict_row, "BUY"),
                f"{date} BUY门禁不一致",
            )
            self.assertEqual(
                order_is_executable(code, series_row, "SELL"),
                order_is_executable(code, dict_row, "SELL"),
            )
            self.assertEqual(
                closing_sale_is_executable(code, series_row),
                closing_sale_is_executable(code, dict_row),
            )

    def test_count_leq_matches_boolean_mask(self):
        from engine.fast_index import build_stock_index

        rng = np.random.default_rng(43)
        frame = normalise_like_cache(random_frame(rng, 200))
        index = build_stock_index(frame)
        for offset in range(0, len(frame), 13):
            target = frame["date"].iloc[offset]
            expected = int((frame["date"] <= target).sum())
            self.assertEqual(expected, index.count_leq(target.value))
        between = pd.Timestamp("2025-03-08")  # 周末，不在日历
        expected = int((frame["date"] <= between).sum())
        self.assertEqual(expected, index.count_leq(between.value))


class SelectCandidatesFastEquivalenceTests(unittest.TestCase):
    """选股快速路径 vs 原实现：候选代码与评分完全一致。"""

    @staticmethod
    def _build_indexes(data):
        from engine.fast_index import build_stock_indexes

        return build_stock_indexes(data)

    def _compare(self, data, signal_dates, config):
        indexes = self._build_indexes(data)
        self.assertIsNotNone(indexes)
        for signal in signal_dates:
            old = select_candidates(data, signal, (), config)
            new = select_candidates(data, signal, (), config, indexes=indexes)
            self.assertEqual(old, new, f"{signal} 候选不一致")

    def test_equivalence_on_normalised_cache(self):
        rng = np.random.default_rng(31)
        data = {}
        for index in range(20):
            frame = random_frame(rng, 250)
            if index % 5 == 4:
                frame.loc[frame.index[-1], "volume"] = 0.0  # 末日停牌
            data[f"sh.60{index:04d}"] = normalise_like_cache(frame)
        data["sh.000300"] = normalise_like_cache(random_frame(rng, 250))
        self._compare(data, list(pd.bdate_range("2025-06-01", periods=10)), StrategyConfig)

    def test_equivalence_with_range_index_frames(self):
        rng = np.random.default_rng(32)
        data = {}
        for index in range(15):
            data[f"sz.00{index:04d}"] = random_frame(rng, 220)
        data["sh.000300"] = random_frame(rng, 220)
        config = type("Cfg", (StrategyConfig,), {"use_pe_filter": False, "use_pb_filter": False})
        self._compare(data, list(pd.bdate_range("2025-06-01", periods=8)), config)

    def test_equivalence_with_config_variants(self):
        rng = np.random.default_rng(33)
        data = {}
        for index in range(18):
            data[f"sh.60{index:04d}"] = normalise_like_cache(random_frame(rng, 260))
        data["sh.000300"] = normalise_like_cache(random_frame(rng, 260))
        variants = [
            type("Cfg1", (StrategyConfig,), {"consecutive_up_days": 3}),
            type("Cfg2", (StrategyConfig,), {"stock_ma_period": 20}),
            type("Cfg3", (StrategyConfig,), {"stock_ma_crossover": True}),
            type(
                "Cfg4",
                (StrategyConfig,),
                {
                    "anti_falling_knife": True,
                    "anti_falling_lookback": 5,
                },
            ),
            type("Cfg5", (StrategyConfig,), {"use_pe_filter": False, "use_pb_filter": False}),
            type("Cfg6", (StrategyConfig,), {"min_score": 0.0, "top_volume_n": 5}),
        ]
        for config in variants:
            self._compare(data, list(pd.bdate_range("2025-07-01", periods=6)), config)

    def test_held_codes_excluded(self):
        rng = np.random.default_rng(34)
        data = {}
        for index in range(16):
            data[f"sh.60{index:04d}"] = normalise_like_cache(random_frame(rng, 240))
        data["sh.000300"] = normalise_like_cache(random_frame(rng, 240))
        indexes = self._build_indexes(data)
        signal = pd.bdate_range("2025-07-01", periods=1)[0]
        held = [code for code in data if code != "sh.000300"][:4]
        old = select_candidates(data, signal, held, StrategyConfig)
        new = select_candidates(data, signal, held, StrategyConfig, indexes=indexes)
        self.assertEqual(old, new)
        self.assertFalse(any(code in held for code, _score in new))

    def test_unsorted_frame_falls_back_to_legacy_path(self):
        rng = np.random.default_rng(35)
        frame = random_frame(rng, 240)
        shuffled = frame.iloc[np.random.default_rng(36).permutation(len(frame))]
        data = {"sh.600000": shuffled}
        from engine.fast_index import build_stock_index

        self.assertIsNone(build_stock_index(shuffled))
        indexes = self._build_indexes(data)
        # 无可用索引时快速路径必须退回旧实现且结果一致。
        signal = shuffled["date"].iloc[200]
        legacy = select_candidates(data, signal, (), StrategyConfig)
        fast = select_candidates(data, signal, (), StrategyConfig, indexes=indexes)
        self.assertEqual(legacy, fast)


class SegmentConsistencySwitchTests(unittest.TestCase):
    """--no-segment 开关：跳过分段一致性时如实披露、评分降级不伪造。"""

    def _request(self, **overrides):
        import tempfile

        from commercial.config import CommercialRequest

        request = CommercialRequest(
            cache="unused.pkl",
            start="2025-01-15",
            end="2025-06-30",
            run_stress=False,
            run_factor_diagnostics=False,
            run_rolling_oos=False,
            quiet=True,
            output_root=tempfile.mkdtemp(prefix="alice_seg_test_"),
            **overrides,
        )
        request.validate()
        return request

    def test_request_default_keeps_segment_on(self):
        request = self._request()
        self.assertTrue(request.run_segment_consistency)

    def test_scoring_treats_skipped_as_insufficient(self):
        from commercial.reliability_score import compute_reliability_score

        dimension = None
        score = compute_reliability_score(
            data_audit={"rating": "READY", "warnings": [], "blockers": []},
            base_result={"metrics": {"sharpe_reliable": True, "total_trades": 12}},
            segment_consistency={"status": "skipped", "detail": "用户选择跳过分段一致性检查"},
            stress_summary={"status": "ok", "ranking": [], "over_sensitive": False},
            manifest_complete=True,
            metadata_present=True,
        )
        dimensions = {d["key"]: d for d in score["dimensions"]}
        dimension = dimensions.get("out_of_sample")
        self.assertIsNotNone(dimension)
        self.assertEqual(dimension["status"], "insufficient_data")

    def test_pipeline_skips_segment_when_disabled(self):
        import commercial.pipeline as pipeline

        calls = {"segment": 0}

        def fake_base(request, config, cache_path, data=None):
            return {
                "metrics": {"total_return_pct": 5.0},
                "equity": None,
                "trades": None,
                "statistics": {},
                "portfolio": None,
            }

        def fake_segment(request, config, cache_path, data=None):
            calls["segment"] += 1
            return {"status": "ok", "verdict": "consistent"}

        original_base = pipeline.run_base_backtest
        original_segment = pipeline.run_segment_consistency
        original_prepare = pipeline._prepare
        original_cache = pipeline.prepare_cache
        original_bundle = pipeline.write_report_bundle
        original_score = pipeline.compute_reliability_score
        pipeline.run_base_backtest = fake_base
        pipeline.run_segment_consistency = fake_segment
        pipeline.prepare_cache = lambda request, data, benchmark, working_dir: ("unused.pkl", False)
        pipeline._prepare = lambda request, report_dir, emit=None: (
            {"sh.000300": pd.DataFrame({"date": [], "close": []})},
            None,
            {},
            "sh.000300",
            {"rating": "READY", "warnings": [], "blockers": []},
            None,
        )
        pipeline.write_report_bundle = lambda report_dir, analysis: {}
        try:
            request = self._request(run_segment_consistency=False)
            analysis = pipeline.run_full_analysis(request, report_id="seg_off_test")
            self.assertEqual(calls["segment"], 0)
            self.assertEqual(analysis["segment_consistency"]["status"], "skipped")
        finally:
            pipeline.run_base_backtest = original_base
            pipeline.run_segment_consistency = original_segment
            pipeline._prepare = original_prepare
            pipeline.prepare_cache = original_cache
            pipeline.write_report_bundle = original_bundle
            pipeline.compute_reliability_score = original_score


if __name__ == "__main__":
    unittest.main()
