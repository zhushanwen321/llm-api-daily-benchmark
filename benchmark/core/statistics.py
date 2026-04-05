"""统计计算模块：均值、标准差、置信区间（纯 Python，无 scipy 依赖）."""
from __future__ import annotations

import math
from statistics import mean, stdev


def _norm_ppf(p: float) -> float:
    """标准正态分布的分位数函数（使用 rational 近似，精度约 1e-9）。"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if abs(p - 0.5) < 1e-16:
        return 0.0

    # 使用 Abramowitz & Stegun 26.2.23 的有理近似
    # 该公式计算的是 p < 0.5 时的负分位数
    # 对于 p > 0.5，使用对称性 Phi^{-1}(p) = -Phi^{-1}(1-p)
    q = p if p < 0.5 else 1.0 - p
    t = math.sqrt(-2.0 * math.log(q))

    # 有理近似的系数
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t

    # 该公式返回负值（适用于 p < 0.5）
    result = -(t - numerator / denominator)

    return result if p < 0.5 else -result


def _t_ppf(p: float, df: int) -> float:
    """t 分布的分位数函数。

    - df=1 时使用 Cauchy 分布的精确解
    - df >= 2 时使用 Cornish-Fisher 三阶展开，df >= 15 时误差 < 0.001
    """
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # df=1 是 Cauchy 分布，有闭式解：tan(π × (p - 0.5))
    if df == 1:
        return math.tan(math.pi * (p - 0.5))

    z = _norm_ppf(p)
    z3, z5, z7 = z ** 3, z ** 5, z ** 7
    df_f = float(df)
    df2, df3 = df_f * df_f, df_f * df_f * df_f

    # Cornish-Fisher 三阶展开
    term1 = z
    term2 = (z3 + z) / (4.0 * df_f)
    term3 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df2)
    term4 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df3)

    return term1 + term2 + term3 + term4


def calculate_mean(scores: list[float]) -> float:
    if not scores:
        raise ValueError("Cannot calculate mean of empty list")
    return mean(scores)


def calculate_std(scores: list[float]) -> float:
    if len(scores) < 2:
        raise ValueError("Cannot calculate std with less than 2 samples")
    return stdev(scores)


def calculate_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    if len(scores) < 2:
        raise ValueError("Cannot calculate CI with less than 2 samples")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1 (exclusive)")

    n = len(scores)
    sample_mean = mean(scores)
    sample_std = stdev(scores)
    standard_error = sample_std / (n ** 0.5)
    t_critical = _t_ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * standard_error
    return (sample_mean - margin_of_error, sample_mean + margin_of_error)
