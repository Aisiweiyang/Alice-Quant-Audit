import unittest
from unittest.mock import patch

from data.fetcher import (
    DAILY_COLUMNS,
    _adapt_worker_count,
    _daily_frame,
    _split_worker_codes,
    get_batch_data,
    get_stock_list,
)


class FetcherTests(unittest.TestCase):
    def test_single_incremental_row_is_kept(self):
        row = [
            "2026-08-11",
            "10",
            "10.2",
            "9.9",
            "10.1",
            "10",
            "1000",
            "10000",
            "1.0",
            "2.0",
            "20",
            "2",
            "1",
            "0",
        ]
        frame = _daily_frame([row])
        self.assertEqual(list(frame.columns), DAILY_COLUMNS)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["tradestatus"], 1)
        self.assertEqual(frame.iloc[0]["isST"], 0)

    def test_batch_reconnects_after_five_codes(self):
        class FakeResult:
            error_code = "0"
            error_msg = ""

            def __init__(self):
                self._available = True

            def next(self):
                available, self._available = self._available, False
                return available

            def get_row_data(self):
                return [
                    "2026-08-11",
                    "10",
                    "10.2",
                    "9.9",
                    "10.1",
                    "10",
                    "1000",
                    "10000",
                    "1.0",
                    "2.0",
                    "20",
                    "2",
                    "1",
                    "0",
                ]

        codes = [f"sh.60000{i}" for i in range(6)]
        with (
            patch("data.fetcher._login") as login,
            patch("data.fetcher._logout") as logout,
            patch("data.fetcher.time.sleep"),
            patch(
                "data.fetcher.bs.query_history_k_data_plus",
                side_effect=lambda *_args: FakeResult(),
                create=True,
            ),
        ):
            result = get_batch_data(
                codes,
                "2026-08-11",
                "2026-08-11",
                verbose=False,
            )

        self.assertEqual(set(result), set(codes))
        self.assertEqual(login.call_count, 2)
        self.assertEqual(logout.call_count, 2)

    def test_batch_login_failure_is_left_for_checkpoint_retry(self):
        with (
            patch(
                "data.fetcher._login",
                side_effect=UnicodeDecodeError("utf-8", b"\x8b", 0, 1, "bad byte"),
            ),
            patch("data.fetcher._logout"),
            patch("data.fetcher.time.sleep"),
        ):
            result = get_batch_data(
                ["sh.600000"],
                "2026-08-11",
                "2026-08-11",
                verbose=False,
                retries=2,
            )

        self.assertEqual(result, {})

    def test_batch_reconnect_failure_does_not_escape_worker(self):
        with (
            patch(
                "data.fetcher._login",
                side_effect=[None, OSError("login failed"), OSError("login failed")],
            ),
            patch("data.fetcher._logout"),
            patch("data.fetcher.time.sleep"),
            patch(
                "data.fetcher.bs.query_history_k_data_plus",
                side_effect=TimeoutError("timed out"),
                create=True,
            ),
        ):
            result = get_batch_data(
                ["sh.600000"],
                "2026-08-11",
                "2026-08-11",
                verbose=False,
                retries=2,
            )

        self.assertEqual(result, {})

    def test_worker_split_is_complete_and_disjoint(self):
        codes = [f"sh.600{i:03d}" for i in range(11)]
        groups = _split_worker_codes(codes, 2)
        self.assertEqual([len(group) for group in groups], [6, 5])
        self.assertEqual(set(groups[0]).intersection(groups[1]), set())
        self.assertEqual(set(groups[0] + groups[1]), set(codes))

    def test_hs300_list_does_not_fetch_the_full_security_master(self):
        class FakeResult:
            error_code = "0"
            error_msg = ""

            def __init__(self):
                self.rows = iter(
                    [
                        ["2026-09-01", "sh.600519", "A"],
                        ["2026-09-01", "sz.000001", "B"],
                    ]
                )
                self.current = None

            def next(self):
                self.current = next(self.rows, None)
                return self.current is not None

            def get_row_data(self):
                return self.current

        with (
            patch("data.fetcher._login"),
            patch("data.fetcher._logout"),
            patch(
                "data.fetcher.bs.query_hs300_stocks",
                return_value=FakeResult(),
                create=True,
            ),
            patch("data.fetcher.bs.query_stock_basic", create=True) as stock_basic,
        ):
            codes = get_stock_list("hs300")

        self.assertEqual(codes, ["sh.600519", "sz.000001"])
        stock_basic.assert_not_called()

    def test_adaptive_workers_reduce_after_provider_instability(self):
        workers, healthy = _adapt_worker_count(
            3,
            1,
            requested=60,
            retry_errors=12,
            failed_codes=0,
            min_workers=2,
            max_workers=4,
        )
        self.assertEqual((workers, healthy), (2, 0))

    def test_adaptive_workers_grow_after_two_healthy_batches(self):
        workers, healthy = _adapt_worker_count(
            3,
            0,
            requested=100,
            retry_errors=0,
            failed_codes=0,
            min_workers=2,
            max_workers=4,
        )
        self.assertEqual((workers, healthy), (3, 1))
        workers, healthy = _adapt_worker_count(
            workers,
            healthy,
            requested=100,
            retry_errors=0,
            failed_codes=0,
            min_workers=2,
            max_workers=4,
        )
        self.assertEqual((workers, healthy), (4, 0))


if __name__ == "__main__":
    unittest.main()
