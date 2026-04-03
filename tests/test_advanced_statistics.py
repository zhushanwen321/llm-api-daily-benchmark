import pytest
from benchmark.core.advanced_statistics import (
    bootstrap_confidence_interval,
    ttest_significance,
    pairwise_comparison,
)


def test_bootstrap_ci_basic():
    scores = [80, 90, 100, 70, 60, 85, 95, 75, 80, 90]
    lower, upper = bootstrap_confidence_interval(scores, confidence=0.95)
    assert lower < upper
    assert lower < 82.5 < upper  # 均值应在区间内


def test_bootstrap_ci_single_value():
    """所有值相同时，CI 退化为该值."""
    scores = [50.0] * 10
    lower, upper = bootstrap_confidence_interval(scores)
    assert abs(lower - 50.0) < 1.0
    assert abs(upper - 50.0) < 1.0


def test_ttest_significant():
    a = [80, 90, 100, 70, 60, 85, 95, 75, 80, 90]
    b = [60, 70, 80, 50, 40, 65, 75, 55, 60, 70]
    result = ttest_significance(a, b)
    assert result["is_significant"] is True
    assert result["p_value"] < 0.05
    assert result["effect_size"] > 0  # Cohen's d > 0


def test_ttest_not_significant():
    a = [80, 82, 78, 81, 79]
    b = [80, 81, 79, 82, 78]
    result = ttest_significance(a, b)
    assert result["p_value"] > 0.05


def test_ttest_too_few_samples():
    with pytest.raises(ValueError):
        ttest_significance([1], [2])


def test_pairwise_comparison():
    model_scores = {
        "model_a": [80, 85, 90, 75, 95],
        "model_b": [60, 65, 70, 55, 75],
        "model_c": [78, 82, 88, 72, 90],
    }
    results = pairwise_comparison(model_scores)
    assert len(results) == 3  # C(3,2) = 3 pairs
    # model_a vs model_b 应该显著
    ab = [r for r in results if set([r["model_a"], r["model_b"]]) == {"model_a", "model_b"}]
    assert len(ab) == 1
    assert ab[0]["is_significant"] is True
