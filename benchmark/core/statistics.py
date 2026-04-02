"""统计计算模块：均值、标准差、置信区间."""

from __future__ import annotations

from statistics import mean, stdev

import scipy.stats


def calculate_mean(scores: list[float]) -> float:
    """计算分数的均值.

    Args:
        scores: 分数列表.

    Returns:
        均值.

    Raises:
        ValueError: 列表为空时.
    """
    if not scores:
        raise ValueError("Cannot calculate mean of empty list")
    return mean(scores)


def calculate_std(scores: list[float]) -> float:
    """计算分数的样本标准差.

    Args:
        scores: 分数列表.

    Returns:
        样本标准差.

    Raises:
        ValueError: 列表为空或只有一个元素时.
    """
    if len(scores) < 2:
        raise ValueError("Cannot calculate std with less than 2 samples")
    return stdev(scores)


def calculate_confidence_interval(
    scores: list[float],
    confidence: float = 0.95
) -> tuple[float, float]:
    """计算均值的置信区间（单次计算，非Bootstrap）.

    使用 t-distribution 计算置信区间。

    Args:
        scores: 分数列表.
        confidence: 置信水平（默认0.95），必须在(0, 1)范围内.

    Returns:
        (lower_bound, upper_bound).

    Raises:
        ValueError: 列表为空或只有一个元素时，或confidence不在(0, 1)范围内.
    """
    if len(scores) < 2:
        raise ValueError("Cannot calculate CI with less than 2 samples")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1 (exclusive)")

    n = len(scores)
    sample_mean = mean(scores)
    sample_std = stdev(scores)
    standard_error = sample_std / (n ** 0.5)

    # 使用 t-distribution
    t_critical = scipy.stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * standard_error

    return (sample_mean - margin_of_error, sample_mean + margin_of_error)
