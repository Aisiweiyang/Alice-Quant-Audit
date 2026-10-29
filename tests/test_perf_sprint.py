"""性能冲刺轮测试：批量预计算与逐信号实现的数值等价性（红线≤1e-6）。"""

import unittest

import numpy as np
import pandas as pd

from strategy.config import StrategyConfig
from strategy.factors import compute_all, compute_all_batch

from commercial.factor_diagnostics import DIAGNOSTIC_FACTORS


def _make_frames(seed=11, stocks=8, days=320, flat_spans=True, nan_pctchg=True):
    """随机OHLCV数据：含停牌（量0/价格平台）、NaN pctChg、涨跌，覆盖边界情形。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-06-03", periods=days)
    frames = {}
    for index in range(stocks):
        steps = rng.normal(0.0004, 0.02, size=days)
        closes = (8 + index) * np.exp(np.cumsum(steps))
        close = closes.round(3)
        # 人为制造价格平台（std(dp)==0触发vol_price_corr门禁）
        if flat_spans and index % 2 == 0:
            close[100:110] = close[100]
        open_ = (close * (1 + rng.normal(0, 0.004, days))).round(3)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, days))
        volume = rng.lognormal(13, 0.5, days).round(0)
        if flat_spans:
            volume[200:206] = 0  # 停牌段
        pct = np.concatenate([[0.0], (close[1:] / close[:-1] - 1) * 100])
        pct = pct.round(4)
        if nan_pctchg and index % 3 == 0:
            pct[50:53] = np.nan
        frames[f"sh.60{index:04d}"] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high.round(3),
                "low": low.round(3),
                "close": close,
                "volume": volume,
                "amount": (volume * close).round(0),
                "preclose": np.roll(close, 1),
                "pctChg": pct,
                "tradestatus": 1,
                "isST": 0,
                "pe": rng.uniform(5, 40, days).round(2),
                "pb": rng.uniform(0.5, 8, days).round(2),
            }
        )
    return frames


class FactorBatchEquivalenceTests(unittest.TestCase):
    """compute_all_batch 与逐信号 compute_all 的逐值等价性（≤1e-6）。"""

    def _check_position(self, frame, pos):
        legacy = compute_all(frame.iloc[: pos + 1], StrategyConfig) or {}
        batched = compute_all_batch(frame, [pos], StrategyConfig)[pos] or {}
        for name in DIAGNOSTIC_FACTORS:
            in_legacy, in_batched = name in legacy, name in batched
            self.assertEqual(
                in_legacy,
                in_batched,
                f"因子{name}在位置{pos}的存在性不一致: "
                f"legacy={legacy.get(name)} batched={batched.get(name)}",
            )
            if in_legacy and in_batched:
                diff = abs(float(legacy[name]) - float(batched[name]))
                self.assertLessEqual(
                    diff,
                    1e-6,
                    f"因子{name}在位置{pos}差异{diff}超过1e-6: "
                    f"legacy={legacy[name]} batched={batched[name]}",
                )
        if "macd_golden" in legacy:
            self.assertEqual(legacy["macd_golden"], batched.get("macd_golden"))

    def test_batch_matches_per_signal_compute_all(self):
        for seed in (11, 42):
            for code, frame in _make_frames(seed=seed).items():
                length = len(frame)
                for pos in (59, 60, 100, 109, 205, 210, length // 2, length - 1):
                    self._check_position(frame, pos)

    def test_batch_short_history_returns_none(self):
        frame = _make_frames(stocks=1)["sh.600000"]
        for pos in (0, 5, 28):
            self.assertIsNone(compute_all_batch(frame, [pos], StrategyConfig)[pos])
        self.assertIsNone(compute_all_batch(frame, [-1, 10**6], StrategyConfig)[-1])

    def test_batch_multiposition_equals_single(self):
        frame = _make_frames(stocks=1)["sh.600000"]
        positions = [60, 100, 150, 200, len(frame) - 1]
        multi = compute_all_batch(frame, positions, StrategyConfig)
        for pos in positions:
            single = compute_all_batch(frame, [pos], StrategyConfig)[pos]
            for name in DIAGNOSTIC_FACTORS:
                self.assertEqual(
                    multi[pos].get(name),
                    single.get(name),
                    f"批量与单个计算在位置{pos}因子{name}不一致",
                )

    def test_batch_exactness_on_normal_data(self):
        """常规数据上批量与逐信号应逐位一致（不只是≤1e-6）。"""
        frame = _make_frames(seed=7, stocks=1, flat_spans=False, nan_pctchg=False)["sh.600000"]
        for pos in (60, 120, 240, len(frame) - 1):
            legacy = compute_all(frame.iloc[: pos + 1], StrategyConfig) or {}
            batched = compute_all_batch(frame, [pos], StrategyConfig)[pos] or {}
            for name in DIAGNOSTIC_FACTORS:
                if name in legacy:
                    self.assertEqual(float(legacy[name]), float(batched[name]))


if __name__ == "__main__":
    unittest.main()
