"""高级统计分析模块：Bootstrap 置信区间 + t-test 显著性检验（纯 Python）."""
from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Any

from benchmark.core.statistics import _t_ppf


def _percentile(sorted_data: list[float], q: float) -> float:
    """排序数据的第 q 百分位数（线性插值法，与 numpy 默认一致）。"""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    rank = q / 100.0 * (n - 1)
    lower = int(math.floor(rank))
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    frac = rank - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


def _t_cdf_bisect(t_stat: float, df: float) -> float:
    """通过二分法 + _t_ppf 求 t 分布的 CDF。"""
    if abs(t_stat) < 1e-15:
        return 0.5
    lo, hi = (0.5, 1.0) if t_stat > 0 else (0.0, 0.5)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        try:
            val = _t_ppf(mid, df=int(round(df)))
        except ValueError:
            hi = mid
            continue
        if val < t_stat:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _ttest_ind_p_value(mean_a, mean_b, var_a, var_b, n_a, n_b):
    """Welch's t-test 双侧 p 值。"""
    var_sum = var_a / n_a + var_b / n_b
    if var_sum == 0:
        return 1.0
    se = math.sqrt(var_sum)
    t_stat = (mean_a - mean_b) / se

    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if denom == 0:
        return 1.0
    df = num / denom

    p_one_side = _t_cdf_bisect(t_stat, df)
    return 2.0 * min(p_one_side, 1.0 - p_one_side)


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    rng = random.Random(42)
    n = len(scores)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choices(scores, k=n)
        bootstrap_means.append(statistics.mean(sample))
    bootstrap_means.sort()
    lower = _percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = _percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    return (lower, upper)


def ttest_significance(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    if len(scores_a) < 2 or len(scores_b) < 2:
        raise ValueError("Each group needs at least 2 samples for t-test")

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = statistics.mean(scores_a), statistics.mean(scores_b)
    var_a, var_b = statistics.variance(scores_a), statistics.variance(scores_b)

    p_value = _ttest_ind_p_value(mean_a, mean_b, var_a, var_b, n_a, n_b)

    pooled_std = math.sqrt((var_a + var_b) / 2.0)
    effect_size = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0
    is_significant = bool(p_value < alpha)

    if is_significant:
        if mean_a > mean_b:
            conclusion = f"model_a 显著优于 model_b (p={p_value:.4f}, d={effect_size:.2f})"
        else:
            conclusion = f"model_b 显著优于 model_a (p={p_value:.4f}, d={effect_size:.2f})"
    else:
        conclusion = f"无显著差异 (p={p_value:.4f})"

    return {
        "p_value": float(p_value),
        "is_significant": is_significant,
        "effect_size": effect_size,
        "conclusion": conclusion,
    }


def pairwise_comparison(
    model_scores: dict[str, list[float]],
    alpha: float = 0.05,
) -> list[dict]:
    models = list(model_scores.keys())
    results = []
    for model_a, model_b in itertools.combinations(models, 2):
        test_result = ttest_significance(
            model_scores[model_a], model_scores[model_b], alpha
        )
        results.append({"model_a": model_a, "model_b": model_b, **test_result})
    return results
