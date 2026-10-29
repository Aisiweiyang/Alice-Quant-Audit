import unittest

import numpy as np
import pandas as pd

from research.ridge_model import apply_frozen_ridge_model, build_walk_forward_ridge_scores
from strategy.alpha158_subset import FEATURE_NAMES, prefixed


class RidgeModelTests(unittest.TestCase):
    def test_walk_forward_training_and_frozen_scoring(self):
        dates = pd.bdate_range("2025-01-01", periods=40)
        codes = [f"sh.603{index:03d}" for index in range(60)]
        data = {}
        for index, code in enumerate(codes):
            daily_rate = 0.0002 + index * 0.00002
            data[code] = pd.DataFrame(
                {
                    "date": dates,
                    "close": 10.0 * np.power(1.0 + daily_rate, np.arange(len(dates))),
                }
            )

        rows = []
        for index, code in enumerate(codes):
            row = {"code": code}
            row.update({prefixed(name): float(index) for name in FEATURE_NAMES})
            rows.append(row)
        store = {date: pd.DataFrame(rows) for date in dates[5:31:2]}
        train_end = dates[25]
        enriched, model = build_walk_forward_ridge_scores(
            store,
            data,
            train_end,
            alpha=10.0,
            forward_days=5,
            minimum_training_dates=3,
        )
        self.assertGreater(model["training_dates"], 3)
        self.assertGreater(model["training_rows"], 0)
        self.assertTrue((enriched[dates[5]]["alpha158_ridge_rank"] == 0.5).all())
        out_of_sample = enriched[dates[27]].sort_values("alpha158_ridge_rank")
        self.assertEqual(out_of_sample.iloc[-1]["code"], codes[-1])

        reapplied = apply_frozen_ridge_model({dates[27]: store[dates[27]]}, model)
        self.assertEqual(
            reapplied[dates[27]].sort_values("alpha158_ridge_rank").iloc[-1]["code"],
            codes[-1],
        )


if __name__ == "__main__":
    unittest.main()
