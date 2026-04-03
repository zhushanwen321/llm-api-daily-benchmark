"""高级统计分析模块：Bootstrap 置信区间 + t-test 显著性检验."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import scipy.stats


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Bootstrap 重采样置信区间.

    通过有放回抽样计算均值置信区间，
    对样本量小（如 15 题）的情况更稳健。
    """
    arr = np.array(scores)
    rng = np.random.default_rng(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = float(np.percentile(bootstrap_means, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(bootstrap_means, (1 + confidence) / 2 * 100))
    return (lower, upper)


def ttest_significance(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """两模型 t-test 显著性检验."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        raise ValueError("Each group needs at least 2 samples for t-test")

    t_stat, p_value = scipy.stats.ttest_ind(scores_a, scores_b)

    # Cohen's d
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    pooled_std = np.sqrt(
        (np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2
    )
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
    """多模型两两 t-test 比较."""
    models = list(model_scores.keys())
    results = []
    for model_a, model_b in itertools.combinations(models, 2):
        test_result = ttest_significance(
            model_scores[model_a], model_scores[model_b], alpha
        )
        results.append({
            "model_a": model_a,
            "model_b": model_b,
            **test_result,
        })
    return results
