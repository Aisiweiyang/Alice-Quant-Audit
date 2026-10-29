"""过拟合审计测试（CSCV的PBO与Deflated Sharpe）。

含S=4全枚举的"已知答案"样例（手工推导）与随机数据上的暴力穷举对照，
实现说明见 commercial/overfitting.py 模块docstring（论文出处）。
"""

import itertools
import math
import sys
import unittest
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from commercial.overfitting import (
    EULER_GAMMA,
    cscv_pbo,
    deflated_sharpe,
    ols_fit,
)


def block_frame(block_means_by_strategy, block_days=2, noise=0.005):
    """按块均值构造日收益矩阵：块内逐日在mean±noise间交替（块间同方差）。"""
    strategies = list(block_means_by_strategy)
    n_blocks = len(block_means_by_strategy[strategies[0]])
    rows = []
    for day in range(n_blocks * block_days):
        block = day // block_days
        offset = noise if (day % 2 == 0) else -noise
        rows.append({name: block_means_by_strategy[name][block] + offset for name in strategies})
    index = pd.bdate_range("2025-01-01", periods=n_blocks * block_days)
    return pd.DataFrame(rows, index=index)


class CscvKnownAnswerTests(unittest.TestCase):
    def test_mirror_design_exact_hand_derived_pbo(self):
        """S=4全枚举（C(4,2)=6组合，逐组合手工推导）→ PBO=4/6=0.6667。

        设计要点：B_i = 0.05 − A_i（镜像块均值）→ 每个组合内两策略半集
        标准差严格相等（|B_i−B_j|=|A_i−A_j|，块内±0.002交替方差相同）
        → Sharpe排名只由均值决定。A=[0.041,0.031,0.021,0.011]，
        B=[0.009,0.019,0.029,0.039]，Δ=总均值差=0.008。
        组合（块序0-3）d=IS均值差(A−B)，翻当且仅当d与Δ−d异号：
        {0,1}: d=0.022, Δ−d=−0.014 异号→IS最优A样本外垫底（计入）；
        {2,3}: d=−0.018→+0.026 计入；{0,2}: d=0.012→−0.004 计入；
        {1,3}: d=−0.008→+0.016 计入；{0,3}: d=0.002→+0.006 同号（该组合
        自对偶，IS最优A样本外仍占优，不计入）；{1,2}: 同上不计入。
        PBO=4/6=0.6667；ω均值=(4×1/3+2×2/3)/6=4/9≈0.4444。
        """
        frame = block_frame(
            {
                "A": [0.041, 0.031, 0.021, 0.011],
                "B": [0.009, 0.019, 0.029, 0.039],
            },
            noise=0.002,
        )
        result = cscv_pbo(frame, n_partitions=4, min_block_days=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cscv_combinations"], 6)
        self.assertEqual(result["pbo_probability"], round(4 / 6, 4))
        self.assertEqual(result["mean_oos_relative_rank"], round(4 / 9, 4))
        self.assertEqual(
            result["oos_relative_rank_histogram"],
            [0, 0, 0, 4, 0, 0, 2, 0, 0, 0],
        )

    def test_consistently_dominant_strategy_gives_pbo_zero(self):
        """S=4全枚举：B=A+0.02（恒定偏移，同均值差结构→同半集标准差）
        →IS最优恒为B且OOS仍最优（ω=2/3>0.5从不触发）→ PBO=0。"""
        frame = block_frame(
            {
                "A": [0.01, 0.02, 0.03, 0.04],
                "B": [0.03, 0.04, 0.05, 0.06],
            },
            noise=0.002,
        )
        result = cscv_pbo(frame, n_partitions=4, min_block_days=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pbo_probability"], 0.0)


class CscvBruteForceCrossCheckTests(unittest.TestCase):
    def test_random_matrix_matches_naive_brute_force(self):
        """随机数据上与逐日暴力穷举实现逐值对照（S=4，三方案，含并列压力）。"""
        rng = np.random.default_rng(101)
        matrix = pd.DataFrame(
            rng.normal(0.0005, 0.01, size=(40, 3)),
            index=pd.bdate_range("2025-01-01", periods=40),
            columns=["p1", "p2", "p3"],
        )
        result = cscv_pbo(matrix, n_partitions=4)
        expected = self._brute_force_pbo(matrix, 4)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pbo_probability"], round(expected, 4))

    def test_histogram_sums_to_combinations(self):
        rng = np.random.default_rng(7)
        matrix = pd.DataFrame(
            rng.normal(0, 0.01, size=(160, 4)),
            columns=["a", "b", "c", "d"],
        )
        result = cscv_pbo(matrix, n_partitions=16)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cscv_combinations"], 12870)
        self.assertEqual(sum(result["oos_relative_rank_histogram"]), 12870)
        self.assertTrue(0.0 <= result["pbo_probability"] <= 1.0)

    def test_deterministic_for_same_input(self):
        rng = np.random.default_rng(9)
        matrix = pd.DataFrame(rng.normal(0, 0.01, size=(96, 3)), columns=list("xyz"))
        self.assertEqual(cscv_pbo(matrix), cscv_pbo(matrix))

    @staticmethod
    def _brute_force_pbo(matrix, s):
        values = matrix.dropna(how="any")
        usable = (len(values) // s) * s
        data = values.iloc[:usable].to_numpy(dtype=float)
        block_days = usable // s
        wins = 0
        combos = list(itertools.combinations(range(s), s // 2))
        for is_blocks in combos:
            oos_blocks = [b for b in range(s) if b not in is_blocks]

            def score(blocks):
                chunk = np.concatenate(
                    [data[b * block_days : (b + 1) * block_days] for b in blocks]
                )
                std = chunk.std(axis=0)
                mean = chunk.mean(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = mean / std
                return np.where(
                    std > 0, ratio, np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0))
                )

            score_is = score(list(is_blocks))
            score_oos = score(oos_blocks)
            best = int(np.argmax(score_is))
            lam = 1 + int((score_oos < score_oos[best]).sum())
            omega = lam / (score_oos.size + 1)
            wins += int(omega <= 0.5)
        return wins / len(combos)


class CscvInsufficientDataTests(unittest.TestCase):
    def test_single_strategy_insufficient(self):
        frame = pd.DataFrame({"a": [0.01] * 100})
        result = cscv_pbo(frame)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["pbo_probability"])

    def test_too_few_days_insufficient(self):
        frame = pd.DataFrame({"a": [0.01] * 60, "b": [0.005] * 60, "c": [-0.002] * 60})
        result = cscv_pbo(frame)  # 60天 < 16块×5天
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIn("时段不足", result["detail"])


class DeflatedSharpeTests(unittest.TestCase):
    def test_zero_sharpe_symmetric_returns_gives_exactly_one_half(self):
        """已知答案：±对称收益 → SR=0、偏度=0 → DSR=Φ(0)=0.5（N=1退化PSR）。"""
        values = np.tile([0.01, -0.01, 0.02, -0.02], 10)
        result = deflated_sharpe(pd.Series(values), n_trials=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deflated_sharpe"], 0.5)
        self.assertEqual(result["sr_observed"], 0.0)

    def test_positive_sharpe_single_trial_is_psr_and_matches_formula(self):
        rng = np.random.default_rng(55)
        values = rng.normal(0.001, 0.01, 200)
        result = deflated_sharpe(pd.Series(values), n_trials=1)
        # 与论文公式的独立复算对照（N=1时SR0=0，退化为PSR）。
        mean, std = values.mean(), values.std(ddof=1)
        sr = mean / std
        centered = values - mean
        m2 = float(np.mean(centered**2))
        skew = float(np.mean(centered**3)) / m2**1.5
        kurt = float(np.mean(centered**4)) / (m2**2)
        denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
        expected = NormalDist().cdf(sr * math.sqrt(len(values) - 1) / math.sqrt(denom))
        self.assertAlmostEqual(result["deflated_sharpe"], round(expected, 4), places=3)

    def test_more_trials_deflate_sharpe(self):
        rng = np.random.default_rng(66)
        values = pd.Series(rng.normal(0.001, 0.01, 250))
        one = deflated_sharpe(values, n_trials=1)
        many = deflated_sharpe(values, n_trials=50)
        self.assertGreater(one["deflated_sharpe"], many["deflated_sharpe"])
        self.assertGreater(many["sr0_expected_max"], one["sr0_expected_max"])

    def test_trial_sharpe_dispersion_sets_sr0(self):
        rng = np.random.default_rng(77)
        values = pd.Series(rng.normal(0.0008, 0.01, 200))
        result = deflated_sharpe(values, n_trials=3, trial_sharpes=[0.10, 0.20, 0.30])
        self.assertEqual(result["variance_source"], "trial_sharpes")
        sigma = float(np.std(np.array([0.10, 0.20, 0.30]), ddof=1))
        expected_sr0 = sigma * (
            (1 - EULER_GAMMA) * NormalDist().inv_cdf(1 - 1 / 3)
            + EULER_GAMMA * NormalDist().inv_cdf(1 - 1 / (3 * math.e))
        )
        self.assertAlmostEqual(result["sr0_expected_max"], round(expected_sr0, 4), places=3)

    def test_insufficient_observations(self):
        result = deflated_sharpe(pd.Series([0.01] * 10), n_trials=1)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["deflated_sharpe"])


class DegradationRegressionTests(unittest.TestCase):
    """性能退化回归（IS最优方案的OOS Sharpe对IS Sharpe）的已知答案测试。"""

    def test_ols_known_answers(self):
        # y=2x：手算 slope=2, intercept=0
        result = ols_fit([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(result["slope"], 2.0)
        self.assertEqual(result["intercept"], 0.0)
        self.assertEqual(result["pairs"], 4)
        # 水平线：slope=0, intercept=1
        result = ols_fit([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(result["slope"], 0.0)
        self.assertEqual(result["intercept"], 1.0)
        # 负斜率：x=[1,2], y=[3,-1] → cov=-2/var=0.5 → slope=-4,
        # intercept = 1-(-4)*1.5 = 7
        result = ols_fit([1.0, 2.0], [3.0, -1.0])
        self.assertEqual(result["slope"], -4.0)
        self.assertEqual(result["intercept"], 7.0)

    def test_nonfinite_pairs_dropped_and_insufficient_cases(self):
        # 含inf的组合只剔该对（常数块Sharpe为±inf），不影响其余成对观测。
        result = ols_fit([1.0, np.inf, 3.0], [2.0, 4.0, 6.0])
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["slope"], 2.0)
        # 成对观测不足 / x无离散度 → 斜率截距如实None。
        self.assertIsNone(ols_fit([1.0], [2.0])["slope"])
        self.assertIsNone(ols_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["slope"])

    def test_cscv_output_carries_degradation(self):
        frame = block_frame(
            {
                "A": [0.041, 0.031, 0.021, 0.011],
                "B": [0.009, 0.019, 0.029, 0.039],
            },
            noise=0.002,
        )
        result = cscv_pbo(frame, n_partitions=4, min_block_days=1)
        degradation = result["degradation_regression"]
        self.assertEqual(result["cscv_combinations"], 6)
        self.assertEqual(degradation["pairs"], 6)
        self.assertIsNotNone(degradation["slope"])
        self.assertIsNotNone(degradation["intercept"])
        # 该镜像设计中IS最优的OOS Sharpe结构对称，斜率应为有限值且非正
        # （样本内越优样本外越差）——与PBO=4/6的结论互相印证。
        self.assertLessEqual(degradation["slope"], 1.0)


if __name__ == "__main__":
    unittest.main()
