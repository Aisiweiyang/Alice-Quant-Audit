import unittest

import numpy as np
import pandas as pd

from research.factor_diagnostics import diagnose_feature_store
from strategy.alpha158_subset import FEATURE_NAMES, prefixed


class FactorDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_exclude_dates_after_training_cutoff(self):
        dates = pd.bdate_range("2025-01-01", periods=15)
        data = {}
        rows = []
        for index in range(12):
            code = f"sh.600{index:03d}"
            prices = np.linspace(10 + index, 12 + index * 1.2, len(dates))
            data[code] = pd.DataFrame({"date": dates, "close": prices})
            row = {"code": code}
            row.update({prefixed(name): float(index) for name in FEATURE_NAMES})
            rows.append(row)
        store = {
            dates[5]: pd.DataFrame(rows),
            dates[6]: pd.DataFrame(rows),
            pd.Timestamp("2026-01-05"): pd.DataFrame(rows),
        }
        daily, summary = diagnose_feature_store(store, data, "2025-12-31", forward_days=5)
        self.assertEqual(daily["signal_date"].nunique(), 2)
        self.assertEqual(len(summary), len(FEATURE_NAMES))
        self.assertTrue((daily["signal_date"] < "2026-01-01").all())

    def test_forward_labels_cannot_cross_training_cutoff(self):
        dates = pd.bdate_range("2025-12-22", periods=10)
        data = {}
        rows = []
        for index in range(10):
            code = f"sh.601{index:03d}"
            data[code] = pd.DataFrame(
                {"date": dates, "close": np.linspace(10, 11 + index, len(dates))}
            )
            row = {"code": code}
            row.update({prefixed(name): float(index) for name in FEATURE_NAMES})
            rows.append(row)
        daily, _ = diagnose_feature_store(
            {pd.Timestamp("2025-12-29"): pd.DataFrame(rows)},
            data,
            "2025-12-31",
            forward_days=5,
        )
        self.assertTrue(daily["rank_ic"].isna().all())


if __name__ == "__main__":
    unittest.main()
