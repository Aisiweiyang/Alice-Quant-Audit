"""证券主数据接入数据审计的测试：退市/晚IPO股票按存续期评估，不再误判覆盖失败。

背景（2026-09-12 真实数据验证轮发现）：正式级真实缓存含正常退市与期中IPO的
股票，但管线未把缓存旁的security_master.csv传入审计，导致这些股票按全窗口
评估被误判为覆盖失败，正式级数据只能得到LIMITED评级。
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from commercial.data_audit import _delisted_coverage, audit_data_credibility
from commercial.data_input import load_security_master


def price_frame(dates, rng):
    closes = 15 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=len(dates))))
    preclose = np.concatenate([[closes[0]], closes[:-1]])
    volume = rng.lognormal(13, 0.3, size=len(dates)).round(0)
    return pd.DataFrame(
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
            "pe": np.full(len(dates), 20.0),
            "pb": np.full(len(dates), 2.0),
        }
    )


def make_audit_data(start="2025-01-01", end="2025-12-31"):
    """基准+4只完整股票；调用方按需截断个股模拟退市/晚IPO。"""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(start, end=end)
    data = {"sh.000300": price_frame(dates, rng)[["date", "close", "preclose"]]}
    for index in range(4):
        code = f"sh.60{index:04d}"
        data[code] = price_frame(dates, rng)
    return data


FULL_MASTER = pd.DataFrame(
    [
        # code, code_name, ipoDate, outDate, type, status
        ["sh.600000", "存续股", "1999-11-10", "", "1", "1"],
        ["sh.600001", "期内退市股", "1999-11-10", "2025-06-30", "1", "0"],
        ["sh.600002", "期中IPO股", "2025-07-01", "", "1", "1"],
        ["sh.600003", "窗口前退市股", "1999-11-10", "2014-05-01", "1", "0"],
        ["sh.999999", "非股票行(指数)", "1991-07-15", "2025-04-01", "2", "0"],
    ],
    columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
)


class LoadSecurityMasterTests(unittest.TestCase):
    def test_reads_sidecar_with_bom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = os.path.join(temp_dir, "demo.pkl")
            open(cache, "wb").close()
            master_path = cache + ".security_master.csv"
            FULL_MASTER.to_csv(master_path, index=False, encoding="utf-8-sig")
            loaded = load_security_master(cache)
            self.assertIsNotNone(loaded)
            self.assertIn("outDate", loaded.columns)
            self.assertEqual(len(loaded), len(FULL_MASTER))

    def test_missing_sidecar_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = os.path.join(temp_dir, "demo.pkl")
            open(cache, "wb").close()
            self.assertIsNone(load_security_master(cache))

    def test_missing_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = os.path.join(temp_dir, "not_exist.pkl")
            self.assertIsNone(load_security_master(cache))

    def test_sidecar_without_required_columns_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = os.path.join(temp_dir, "demo.pkl")
            open(cache, "wb").close()
            master_path = cache + ".security_master.csv"
            pd.DataFrame({"code": ["sh.600000"]}).to_csv(master_path, index=False)
            self.assertIsNone(load_security_master(cache))


class DelistingTailTests(unittest.TestCase):
    """退市尾段缺失细分：透明度字段不改评级口径（2026-09-14）。"""

    def test_tail_gap_near_outdate_is_marked(self):
        # 头部完整、尾部提前终止25天：越过头10天宽限仍是fail，
        # 但outDate在30天标记窗内 → 标记+警告（透明度增强，评级不变）
        data = make_audit_data()
        data["sh.600001"] = data["sh.600001"][
            data["sh.600001"]["date"] <= "2025-08-15"
        ].copy()  # 数据止于2025-08-15，主数据退市日2025-09-09（25天后）
        master = FULL_MASTER.copy()
        master.loc[master["code"] == "sh.600001", "outDate"] = "2025-09-09"
        audit = audit_data_credibility(
            data,
            start="2025-01-01",
            end="2025-12-31",
            security_master=master,
        )
        row = next(r for r in audit["stocks"] if r["code"] == "sh.600001")
        self.assertEqual(row["status"], "fail")  # 评级口径不变
        self.assertTrue(row.get("suspected_delisting_tail"))
        self.assertTrue(any("疑似退市尾段缺失" in w for w in audit["warnings"]))

    def test_gap_far_from_outdate_not_marked(self):
        # outDate远离数据终止日（>30天）：普通覆盖缺口，不标记
        data = make_audit_data()
        data["sh.600001"] = data["sh.600001"][data["sh.600001"]["date"] <= "2025-08-15"].copy()
        master = FULL_MASTER.copy()
        master.loc[master["code"] == "sh.600001", "outDate"] = "2025-12-20"
        audit = audit_data_credibility(
            data,
            start="2025-01-01",
            end="2025-12-31",
            security_master=master,
        )
        row = next(r for r in audit["stocks"] if r["code"] == "sh.600001")
        self.assertEqual(row["status"], "fail")
        self.assertNotIn("suspected_delisting_tail", row)

    def test_head_gap_is_not_delisting_tail(self):
        # 头部就缺（窗口起点无数据）：普通缺口，不标记
        data = make_audit_data()
        data["sh.600001"] = data["sh.600001"][data["sh.600001"]["date"] >= "2025-09-01"].copy()
        master = FULL_MASTER.copy()
        master.loc[master["code"] == "sh.600001", "outDate"] = ""
        audit = audit_data_credibility(
            data,
            start="2025-01-01",
            end="2025-12-31",
            security_master=master,
        )
        row = next(r for r in audit["stocks"] if r["code"] == "sh.600001")
        self.assertNotIn("suspected_delisting_tail", row)
        self.assertFalse(any("疑似退市尾段缺失" in w for w in audit["warnings"]))


class DelistedCoverageTests(unittest.TestCase):
    def test_window_and_type_filter(self):
        data = {
            "sh.600000": pd.DataFrame(),
            "sh.600001": pd.DataFrame(),
            "sh.600003": pd.DataFrame(),
        }
        result = _delisted_coverage(
            data,
            "sh.000300",
            FULL_MASTER,
            start="2025-01-01",
            end="2025-12-31",
        )
        # 分母只含窗口内退市的股票(sh.600001)，排除窗口前退市(sh.600003)
        # 与指数行(sh.999999, type=2)；1/1=100%应为pass。
        self.assertEqual(result["status"], "pass")
        self.assertIn("1只", result["detail"])
        self.assertIn("100.0%", result["detail"])

    def test_no_master_is_insufficient(self):
        result = _delisted_coverage({"sh.600000": pd.DataFrame()}, "sh.000300", None)
        self.assertEqual(result["status"], "insufficient_data")

    def test_low_coverage_is_fail_and_blocks(self):
        master = pd.DataFrame(
            [
                ["sh.600001", "退市A", "1999-01-01", "2025-03-01", "1", "0"],
                ["sh.600002", "退市B", "1999-01-01", "2025-04-01", "1", "0"],
                ["sh.600003", "退市C", "1999-01-01", "2025-05-01", "1", "0"],
                ["sh.600004", "退市D", "1999-01-01", "2025-06-01", "1", "0"],
            ],
            columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
        )
        result = _delisted_coverage(
            {"sh.600001": pd.DataFrame()},
            "sh.000300",
            master,
            start="2025-01-01",
            end="2025-12-31",
        )
        self.assertEqual(result["status"], "fail")
        self.assertLess(float(result["detail"].split("覆盖比例")[1].split("%")[0]), 30)


class AuditWithMasterTests(unittest.TestCase):
    def test_delisted_and_late_ipo_stocks_pass(self):
        data = make_audit_data()
        # 期内退市：sh.600001数据在2025-06-30终止
        data["sh.600001"] = data["sh.600001"][data["sh.600001"]["date"] <= "2025-06-30"].copy()
        # 期中IPO：sh.600002数据从2025-07-01开始
        data["sh.600002"] = data["sh.600002"][data["sh.600002"]["date"] >= "2025-07-01"].copy()
        audit = audit_data_credibility(
            data,
            start="2025-01-01",
            end="2025-12-31",
            security_master=FULL_MASTER,
        )
        by_code = {row["code"]: row for row in audit["stocks"]}
        self.assertEqual(by_code["sh.600001"]["status"], "pass")
        self.assertEqual(by_code["sh.600002"]["status"], "pass")
        self.assertEqual(audit["summary"]["failed_codes"], 0)
        self.assertTrue(audit["fields"]["delisted_coverage"]["status"] != "insufficient_data")
        self.assertEqual(audit["survivorship_bias_risk"]["level"], "low")
        self.assertFalse(audit["blockers"])

    def test_full_window_stock_against_early_outdate_still_fails(self):
        # 反向防护：master声称2025-06-30退市，但数据一直延续到窗口末——
        # 说明数据与主数据矛盾，截断期内未覆盖全部交易日时仍应能暴露问题。
        data = make_audit_data()
        data["sh.600001"] = data["sh.600001"].iloc[:125].copy()
        master = FULL_MASTER.copy()
        master.loc[master["code"] == "sh.600001", "outDate"] = ""
        audit = audit_data_credibility(
            data,
            start="2025-01-01",
            end="2025-12-31",
            security_master=master,
        )
        by_code = {row["code"]: row for row in audit["stocks"]}
        self.assertEqual(by_code["sh.600001"]["status"], "fail")
        self.assertEqual(audit["summary"]["failed_codes"], 1)
        self.assertTrue(any("未通过覆盖率" in w for w in audit["warnings"]))

    def test_without_master_rating_is_limited(self):
        # 未提供主数据时保持既有契约：退市覆盖"数据不足"，评级受限。
        data = make_audit_data()
        data["sh.600001"] = data["sh.600001"][data["sh.600001"]["date"] <= "2025-06-30"].copy()
        audit = audit_data_credibility(
            data,
            start="2025-01-01",
            end="2025-12-31",
        )
        self.assertEqual(audit["rating"], "LIMITED")
        self.assertEqual(audit["fields"]["delisted_coverage"]["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
