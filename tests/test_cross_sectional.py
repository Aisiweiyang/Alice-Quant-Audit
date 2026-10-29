import unittest

import numpy as np
import pandas as pd

from strategy.cross_sectional import factor_row, is_main_board_stock, rank_cross_section
from strategy.factor_config import FactorStrategyConfig


def market_frame(periods=90):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    close = np.linspace(10.0, 13.0, periods)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 1_200_000, periods),
            "amount": np.linspace(10_000_000, 15_000_000, periods),
            "pctChg": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
            "turn": 1.0,
            "pe": 20.0,
            "pb": 2.0,
        }
    )
    frame = frame.set_index("date", drop=False)
    frame.index.name = "_trade_date"
    return frame


class CrossSectionalFactorTests(unittest.TestCase):
    def test_main_board_filter_excludes_indices_and_growth_boards(self):
        self.assertTrue(is_main_board_stock("sh.600000"))
        self.assertTrue(is_main_board_stock("sz.002001"))
        self.assertFalse(is_main_board_stock("sh.000300"))
        self.assertFalse(is_main_board_stock("sz.300001"))

    def test_factor_row_does_not_use_future_observation(self):
        frame = market_frame()
        signal_date = frame.iloc[-2]["date"]
        before = factor_row("sh.600000", frame, signal_date)
        frame.loc[frame.index[-1], ["close", "volume", "amount", "pe", "pb"]] = [99, 9, 9, 99, 9]
        after = factor_row("sh.600000", frame, signal_date)
        for key in ("momentum_60_5", "momentum_20", "volatility_20", "trend_quality"):
            self.assertAlmostEqual(before[key], after[key])

    def test_ranked_score_uses_cross_section_not_raw_scale(self):
        rows = []
        for index in range(3):
            rows.append(
                {
                    "code": f"sh.60000{index}",
                    "amount_20": 100 - index,
                    "above_ma60": True,
                    "momentum_60_5": float(index),
                    "momentum_20": float(index),
                    "short_reversal": float(index),
                    "volatility_20": float(3 - index),
                    "trend_quality": float(index),
                    "volume_confirmation": float(index),
                    "earnings_yield": float(index + 1),
                    "book_to_price": float(index + 1),
                }
            )
        weights = FactorStrategyConfig().factor_weights
        ranked = rank_cross_section(rows, weights)
        self.assertEqual(ranked.iloc[0]["code"], "sh.600002")
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
