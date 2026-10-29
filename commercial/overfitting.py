"""过拟合审计：PBO（CSCV法）与 Deflated Sharpe（López de Prado 学派）。

论文口径：
- PBO: Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest
  Overfitting", Journal of Computational Finance 2017（CSCV组合对称交叉验证法）。
- DSR: Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio
  Management 2014（PSR的多重检验扩展）。

全部按论文公式以numpy/标准库自行实现（未参考任何受限许可实现，mlfinlab等
商业许可代码一律未复制）；数据不足时如实返回insufficient_data，不编造数字。
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from engine.backtest import _dated_returns

# CSCV缺省子块数S=16（论文设定）：时段等分成S块，枚举对半组合。
DEFAULT_PARTITIONS = 16
MIN_BLOCK_DAYS = 5
EULER_GAMMA = 0.5772156649015329


def equity_daily_returns(equity_frame: pd.DataFrame) -> pd.Series:
    """报告用净值帧（date/total列）→ 以交易日为索引的日收益序列。

    供compare各方案与full基准统一提取日收益（PBO矩阵列与DSR输入同源）。
    实现委托引擎的``_dated_returns``（同一口径，避免两处实现漂移）。
    """
    return _dated_returns(equity_frame)


def cscv_pbo(
    returns_matrix: pd.DataFrame,
    n_partitions: int = DEFAULT_PARTITIONS,
    min_block_days: int = MIN_BLOCK_DAYS,
) -> dict:
    """组合对称交叉验证（CSCV）估计回测过拟合概率PBO。

    ``returns_matrix``：行=交易日、列=各方案（策略/参数组）的日收益。
    方法：把时段等分成S个子块，枚举全部C(S, S/2)种对半组合（每组S/2块作
    样本内IS、其余作样本外OOS）；每组合内以日收益Sharpe（mean/std，排序
    口径与年化无关）评价各方案，找IS最优方案并记录其OOS相对排名
    ω = λ/(N+1)（λ按OOS表现升序，1=最差）；PBO = ω≤0.5（样本外跌破中位）
    的组合占比。

    论文口径说明：子块数S=16；样本内统计量用Sharpe；相对排名含N+1归一。
    ``min_block_days``为每子块最少交易日门槛（管线用默认5；S=4小样例
    自测可放低到1，仅验证机制，不代表统计可靠）。
    数据不足（方案<2、子块不足、块长<门槛）时返回insufficient_data。
    """
    frame = returns_matrix.dropna(how="any")
    n_strategies = int(frame.shape[1])
    total_days = int(frame.shape[0])
    s = int(n_partitions)
    if s < 4 or s % 2 != 0:
        return _insufficient(f"子块数S必须为≥4的偶数，当前{s}")
    if n_strategies < 2:
        return _insufficient(f"方案仅{n_strategies}个，CSCV需要≥2个（compare管线要求≥3）")
    block_days = total_days // s
    if block_days < min_block_days:
        return _insufficient(
            f"交易日仅{total_days}天，等分{s}块后每块{block_days}天"
            f"（要求≥{min_block_days}天），时段不足，不输出PBO"
        )
    usable_days = block_days * s
    matrix = frame.iloc[:usable_days]  # 末尾多余交易日截去，保证等分
    values = matrix.to_numpy(dtype=float)  # (usable_days, N)
    blocks = values.reshape(s, block_days, n_strategies)
    block_means = blocks.mean(axis=1)
    block_second = (blocks**2).mean(axis=1)

    combinations = np.array(list(itertools.combinations(range(s), s // 2)))
    complements = np.array([[b for b in range(s) if b not in combo] for combo in combinations])

    def scores(block_index: np.ndarray) -> np.ndarray:
        means = block_means[block_index].mean(axis=1)
        variance = np.maximum(block_second[block_index].mean(axis=1) - means**2, 0.0)
        stds = np.sqrt(variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = means / stds
        # 常数块（std=0）：全正记+inf、全负记-inf、全零记0，保证排序确定性。
        return np.where(
            stds > 0,
            ratio,
            np.where(means > 0, np.inf, np.where(means < 0, -np.inf, 0.0)),
        )

    score_is = scores(combinations)
    score_oos = scores(complements)
    best_is = np.argmax(score_is, axis=1)  # 平分时取先出现者，确定性
    best_oos_scores = score_oos[np.arange(len(combinations)), best_is]
    # λ按OOS表现升序（1=最差）：λ = 1 + 严格更差的方案数。
    lambda_ranks = 1 + (score_oos < best_oos_scores[:, None]).sum(axis=1)
    relative_ranks = lambda_ranks / (n_strategies + 1)
    pbo = float((relative_ranks <= 0.5).mean())
    histogram, _edges = np.histogram(relative_ranks, bins=np.linspace(0.0, 1.0, 11))
    degradation = ols_fit(
        score_is[np.arange(len(combinations)), best_is],
        best_oos_scores,
    )
    return {
        "status": "ok",
        "pbo_probability": round(pbo, 4),
        "cscv_combinations": int(len(combinations)),
        "subblock_count": s,
        "subblock_days": int(block_days),
        "strategy_count": n_strategies,
        "observation_count": usable_days,
        "mean_oos_relative_rank": round(float(relative_ranks.mean()), 4),
        "oos_relative_rank_histogram": [int(x) for x in histogram],
        "degradation_regression": degradation,
        "note": (
            "PBO衡量的是这组方案整体的回测过拟合概率（样本内最优在样本外"
            "跌破中位的组合占比），不是对未来收益的预测。"
        ),
        "method": (
            "CSCV（Bailey/Borwein/López de Prado/Zhu, The Probability of "
            "Backtest Overfitting, Journal of Computational Finance 2017）；"
            "IS统计量=日收益Sharpe（mean/std）"
        ),
    }


def ols_fit(x, y) -> dict:
    """一元最小二乘拟合 y = slope*x + intercept（成对剔除非有限值）。

    用于性能退化诊断：x=各组合IS最优方案的样本内Sharpe，y=同方案样本外
    Sharpe；斜率明显小于1（尤其<0）说明样本内优势在样本外大幅缩水。
    成对观测<2或x无离散度时斜率/截距如实None。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    result = {"slope": None, "intercept": None, "pairs": int(finite.sum())}
    if result["pairs"] < 2:
        return result
    xf, yf = x[finite], y[finite]
    x_mean, y_mean = float(xf.mean()), float(yf.mean())
    x_var = float(((xf - x_mean) ** 2).mean())
    if x_var <= 0:
        return result
    slope = float(((xf - x_mean) * (yf - y_mean)).mean() / x_var)
    result["slope"] = round(slope, 4)
    result["intercept"] = round(float(y_mean - slope * x_mean), 4)
    return result


def _insufficient(reason: str) -> dict:
    return {
        "status": "insufficient_data",
        "detail": reason,
        "pbo_probability": None,
        "cscv_combinations": None,
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_NORMAL_DIST = NormalDist()


def _normal_inv_cdf(p: float) -> float:
    return _NORMAL_DIST.inv_cdf(p)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes: list | None = None) -> dict:
    """Deflated Sharpe Ratio（Bailey & López de Prado, JPM 2014）。

    以“本轮尝试的方案数N”为试验次数，先估计N次试验下期望最大Sharpe SR0，
    再输出观测Sharpe扣除多重检验后仍为真的概率：
        SR0 = σ(SR) × [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]，γ为欧拉常数；
        DSR = Φ[ (SR−SR0)·√(T−1) / √(1−γ3·SR+((γ4−1)/4)·SR²) ]，
    其中γ3/γ4为日收益总体偏度/峰度（Pearson口径，正态=3）。
    σ(SR)优先取各试验Sharpe的样本标准差（compare场景可得）；只有单一Sharpe时
    按论文用估计量 σ²(SR) = (1−γ3·SR+((γ4−1)/4)·SR²)/(T−1)。
    N≤1时SR0=0，DSR退化为PSR（基准=0），与块自助CI互为印证。
    """
    values = pd.Series(returns).dropna().to_numpy(dtype=float)
    t = len(values)
    n_trials = int(n_trials)
    if t < 20:
        return {
            "status": "insufficient_data",
            "detail": f"日收益观测仅{t}条（要求≥20），不输出Deflated Sharpe",
            "deflated_sharpe": None,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std <= 0:
        return {
            "status": "insufficient_data",
            "detail": "日收益标准差为0，Sharpe不可定义，不输出Deflated Sharpe",
            "deflated_sharpe": None,
        }
    sr = mean / std
    centered = values - mean
    m2 = float(np.mean(centered**2))
    if m2 <= 0:
        # 全部收益相同（m2=0）时标准差为0，上方已拦截；此处为防御性兜底。
        return {
            "status": "insufficient_data",
            "detail": "日收益无离散度，不输出Deflated Sharpe",
            "deflated_sharpe": None,
        }
    skew = float(np.mean(centered**3)) / m2**1.5
    kurt = float(np.mean(centered**4)) / (m2**2)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0 or not math.isfinite(denom):
        return {
            "status": "insufficient_data",
            "detail": "Sharpe方差估计量非正，Deflated Sharpe不可靠",
            "deflated_sharpe": None,
        }
    trials = [float(s) for s in (trial_sharpes or []) if np.isfinite(s)]
    if n_trials > 1 and len(trials) >= 2:
        variance_source = "trial_sharpes"
        sigma_sr = float(np.std(np.array(trials), ddof=1))
    else:
        variance_source = "estimator_from_selected_returns"
        sigma_sr = math.sqrt(denom / (t - 1))
    if n_trials > 1:
        sr0 = sigma_sr * (
            (1.0 - EULER_GAMMA) * _normal_inv_cdf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * _normal_inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        )
    else:
        sr0 = 0.0
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)
    probability = _normal_cdf(z)
    return {
        "status": "ok",
        "deflated_sharpe": round(float(probability), 4),
        "n_trials": n_trials,
        "sr_observed": round(float(sr), 4),
        "sr0_expected_max": round(float(sr0), 4),
        "sigma_sr": round(float(sigma_sr), 6),
        "variance_source": variance_source,
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "observations": t,
        "note": (
            "DSR为观测Sharpe在扣除N次试验多重检验后仍显著的概率（0-1）；"
            "是统计证据强度度量，不是对未来收益的预测。"
        ),
        "method": (
            "Bailey & López de Prado, The Deflated Sharpe Ratio, "
            "Journal of Portfolio Management 2014；N=1时退化为PSR"
        ),
    }
