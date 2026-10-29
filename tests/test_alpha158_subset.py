import unittest

import numpy as np
import pandas as pd

from strategy.alpha158_subset import (
    FEATURE_NAMES,
    FEATURE_SET_NAME,
    compute_alpha158_subset,
    feature_manifest,
)


def sample_frame(periods=90):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    close = np.linspace(10.0, 14.0, periods)
    volume = np.linspace(1_000_000.0, 1_300_000.0, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": volume,
            "amount": volume * close,
        },
        index=dates,
    )


class Alpha158SubsetTests(unittest.TestCase):
    def test_manifest_is_complete_and_unique(self):
        manifest = feature_manifest()
        self.assertEqual(manifest["feature_set"], FEATURE_SET_NAME)
        self.assertEqual(manifest["feature_count"], 45)
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(len(manifest["features"]), 45)

    def test_formulas_keep_qlib_orientation(self):
        frame = sample_frame()
        features = compute_alpha158_subset(frame)
        position = len(frame) - 1
        expected_roc20 = frame["close"].iloc[position - 20] / frame["close"].iloc[position]
        self.assertAlmostEqual(features["ROC20"].iloc[position], expected_roc20)
        self.assertAlmostEqual(features["VWAP0"].iloc[position], 1.0)
        self.assertGreater(features["BETA20"].iloc[position], 0.0)
        self.assertAlmostEqual(features["RSQR20"].iloc[position], 1.0, places=10)

    def test_future_rows_do_not_change_prior_features(self):
        frame = sample_frame()
        signal_position = 75
        before = compute_alpha158_subset(frame.iloc[: signal_position + 1]).iloc[-1]
        changed = frame.copy()
        changed.iloc[
            signal_position + 1 :,
            changed.columns.get_indexer(["open", "high", "low", "close", "volume", "amount"]),
        ] = 999999.0
        after = compute_alpha158_subset(changed).iloc[signal_position]
        for name in FEATURE_NAMES:
            left, right = before[name], after[name]
            if pd.isna(left):
                self.assertTrue(pd.isna(right), name)
            else:
                self.assertAlmostEqual(left, right, msg=name)

    def test_invalid_zero_volume_does_not_create_infinity(self):
        frame = sample_frame()
        frame.iloc[-1, frame.columns.get_loc("volume")] = 0.0
        features = compute_alpha158_subset(frame)
        self.assertTrue(pd.isna(features["VWAP0"].iloc[-1]))
        self.assertFalse(np.isinf(features.to_numpy(dtype=float)).any())


if __name__ == "__main__":
    unittest.main()
