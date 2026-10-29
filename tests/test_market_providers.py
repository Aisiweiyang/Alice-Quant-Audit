import unittest
from unittest.mock import patch

import pandas as pd

from data.market_providers import _normalise_akshare, create_provider


class MarketProviderTests(unittest.TestCase):
    def test_chinese_akshare_columns_are_normalised_without_inventing_fields(self):
        raw = pd.DataFrame(
            {
                "日期": ["2025-01-02", "2025-01-03"],
                "开盘": [10.0, 10.2],
                "最高": [10.3, 10.4],
                "最低": [9.9, 10.1],
                "收盘": [10.2, 10.3],
                "成交量": [1000, 1100],
                "成交额": [10200, 11330],
                "换手率": [1.2, 1.3],
            }
        )
        frame = _normalise_akshare(raw)
        self.assertEqual(frame.loc[1, "preclose"], 10.2)
        self.assertAlmostEqual(frame.loc[1, "pctChg"], (10.3 / 10.2 - 1) * 100)
        self.assertTrue(frame[["pe", "pb", "tradestatus", "isST"]].isna().all().all())

    def test_decimal_turnover_is_converted_to_percent(self):
        raw = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03"],
                "open": [10.0, 10.2],
                "high": [10.3, 10.4],
                "low": [9.9, 10.1],
                "close": [10.2, 10.3],
                "volume": [1000, 1100],
                "turnover": [0.012, 0.013],
            }
        )
        frame = _normalise_akshare(raw)
        self.assertAlmostEqual(frame.loc[0, "turn"], 1.2)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            create_provider("unknown")

    def test_baostock_provider_forwards_retry_limit(self):
        provider = create_provider("baostock", retries=5)
        with patch(
            "data.market_providers.baostock_fetcher.get_batch_data", return_value={}
        ) as batch:
            provider.batch(["sh.600000"], "2025-01-01", "2025-01-10")
        self.assertEqual(batch.call_args.kwargs["retries"], 5)


if __name__ == "__main__":
    unittest.main()
