"""指标口径对拍测试（Omega / Deflated Sharpe，已知答案全部手算推导）。

配套文档：docs/指标口径对拍.md（表格：输入/手算/实现/偏差）。
与 tests/test_tmc_omega_reports.py、tests/test_pbo_dsr.py 的差别：那两处
侧重公式自洽与随机数据复算；本文件用固定小数据集的精确手算值对拍，
期望值由有理数算术+标准正态表独立推导（推导过程见对拍文档），不经随机数、
不调用被测实现的任何中间函数。

口径出处：
- Omega：empyrical stats.omega_ratio（关键细节：低于阈值的正值收益计入分母）；
- DSR：Bailey & López de Prado, The Deflated Sharpe Ratio, JPM 2014
  （N=1退化为PSR；偏度/峰度为Pearson总体口径，正态=3）。
"""

import math
import sys
import unittest
from statistics import NormalDist

import pandas as pd

sys.path.insert(0, ".")

from commercial.overfitting import deflated_sharpe
from engine.backtest import _omega_ratio


class OmegaHandAnswerTests(unittest.TestCase):
    def test_threshold_zero_hand_answer_nine_tenths(self):
        """8条固定日收益（含一条0）：手算 0.045/0.050 = 0.9。"""
        returns = [0.01, -0.02, 0.005, -0.01, 0.03, -0.005, 0.0, -0.015]
        # 手算：分子=0.01+0.005+0.03=0.045；分母=0.02+0.01+0.005+0.015=0.05
        self.assertAlmostEqual(_omega_ratio(returns), 0.045 / 0.05, places=12)

    def test_below_threshold_positive_goes_into_denominator(self):
        """阈值0.004：+0.002虽为正值但低于阈值，按empyrical口径计入分母。

        超额 = r−L = [+0.006, −0.002, −0.024, +0.004, −0.014, +0.002]
        → ω = 0.012/0.040 = 0.3。若误按收益符号划分（正值全进分子），
        会得到 0.024/0.030 = 0.8——该用例能同时排除这一误实现。
        """
        returns = [0.01, 0.002, -0.02, 0.008, -0.01, 0.006]
        self.assertAlmostEqual(_omega_ratio(returns, threshold=0.004), 0.3, places=12)

    def test_no_below_threshold_observation_returns_none(self):
        """全部超额为正 → 分母无损失 → 与empyrical一致返回None。"""
        self.assertIsNone(_omega_ratio([0.01, 0.02, 0.005], threshold=0.004))


class DsrHandAnswerTests(unittest.TestCase):
    """共用固定数据集：20条日收益 = 10×(+1.0%) + 10×(−0.5%)。

    手算中间量（精确）：
    mean=0.25%；离差=±0.75%各10个；S2=20×0.75%²=0.1125%（平方和）；
    std(ddof=1)=√(0.1125%/19)=0.7694838%；SR=0.25%/0.7694838%=0.3248931
    （精确式 √152000/1200）；两点对称→偏度=0；两点分布Pearson峰度=1；
    分母 = 1−0×SR+((1−1)/4)SR² = 1（精确）。
    """

    RETURNS = [0.01] * 10 + [-0.005] * 10

    def test_single_trial_psr_hand_answer(self):
        """N=1：SR0=0，DSR退化为PSR。

        z = SR×√19 = 0.0475/√0.001125 = 1.4161764（精确式）
        DSR = Φ(1.4161764) = 0.9216381（手算）→ 实现4位舍入 0.9216。
        """
        result = deflated_sharpe(pd.Series(self.RETURNS), n_trials=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deflated_sharpe"], 0.9216)
        self.assertEqual(result["sr_observed"], 0.3249)
        self.assertEqual(result["skewness"], 0.0)
        self.assertEqual(result["kurtosis"], 1.0)
        self.assertEqual(result["sr0_expected_max"], 0.0)
        self.assertEqual(result["variance_source"], "estimator_from_selected_returns")
        # 手算z的精确式与实现输出的一致性（NormalDist为共享正态CDF预言机）。
        z_hand = 0.0475 / math.sqrt(0.001125)
        self.assertAlmostEqual(z_hand, 1.4161763857498668, places=12)
        self.assertAlmostEqual(NormalDist().cdf(z_hand), 0.9216380660274407, places=12)

    def test_four_trials_with_trial_sharpes_hand_answer(self):
        """N=4，trial_sharpes=[1.0,0.5,0.0,−0.5]（均值0.25，手算σ=√(5/12)）。

        σ(SR)=0.6454972；SR0=0.6454972×[(1−γ)Φ⁻¹(0.75)+γΦ⁻¹(1−1/(4e))]
        =0.6791424；z=(0.3248931−0.6791424)×√19=−1.5441365
        DSR=Φ(−1.5441365)=0.0612776（手算）→ 实现4位舍入 0.0613。
        """
        gamma = 0.5772156649015329
        trials = [1.0, 0.5, 0.0, -0.5]
        sigma_hand = math.sqrt(5.0 / 12.0)
        self.assertAlmostEqual(sigma_hand, 0.6454972243679028, places=12)
        sr0_hand = sigma_hand * (
            (1 - gamma) * NormalDist().inv_cdf(0.75)
            + gamma * NormalDist().inv_cdf(1 - 1 / (4 * math.e))
        )
        self.assertAlmostEqual(sr0_hand, 0.6791423627083124, places=12)
        z_hand = (0.32489314482696546 - 0.6791423627083124) * math.sqrt(19)
        self.assertAlmostEqual(NormalDist().cdf(z_hand), 0.06127762903119133, places=12)

        result = deflated_sharpe(
            pd.Series(self.RETURNS),
            n_trials=4,
            trial_sharpes=trials,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deflated_sharpe"], 0.0613)
        self.assertEqual(result["sr0_expected_max"], 0.6791)
        self.assertEqual(result["sigma_sr"], 0.645497)
        self.assertEqual(result["variance_source"], "trial_sharpes")


if __name__ == "__main__":
    unittest.main()
