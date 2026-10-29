import json
import os
import tempfile
import unittest

from paper_trade.adapters import ManualCsvAdapter


class ManualCsvAdapterTests(unittest.TestCase):
    def test_loads_normalized_read_only_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            account_path = os.path.join(directory, "account.json")
            positions_path = os.path.join(directory, "positions.csv")
            orders_path = os.path.join(directory, "orders.csv")
            fills_path = os.path.join(directory, "fills.csv")
            with open(account_path, "w", encoding="utf-8") as handle:
                json.dump({"cash": 88_000, "total_assets": 100_000}, handle)
            with open(positions_path, "w", encoding="utf-8") as handle:
                handle.write("code,shares,avg_cost,market_price\n")
                handle.write("SH.600000,1000,10.0,12.0\n")
            with open(orders_path, "w", encoding="utf-8") as handle:
                handle.write("order_id,status,code,action,ordered_shares,filled_shares\n")
                handle.write("order-1,ACCEPTED,SH.600000,buy,1000,0\n")
            with open(fills_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "fill_id,order_id,trade_date,code,action,shares,price,"
                    "commission,stamp_tax,other_fee\n"
                )
                handle.write("fill-1,order-1,2025-01-06,SH.600000,buy,100,10,5,0,0\n")

            adapter = ManualCsvAdapter(
                account_path,
                positions_path,
                orders_path,
                fills_path,
            )
            account = adapter.load_account()
            orders = adapter.load_orders()
            fills = adapter.load_fills()

        self.assertEqual(account.cash, 88_000)
        self.assertEqual(account.holdings["sh.600000"]["shares"], 1_000)
        self.assertEqual(orders[0]["status"], "SUBMITTED")
        self.assertEqual(fills[0]["action"], "BUY")
        with self.assertRaises(PermissionError):
            adapter.submit_order({"order_id": "blocked"})
        with self.assertRaises(PermissionError):
            adapter.cancel_order("blocked")


if __name__ == "__main__":
    unittest.main()
