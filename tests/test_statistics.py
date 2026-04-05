# tests/test_statistics.py
import pytest
from benchmark.core.statistics import calculate_mean, calculate_std, calculate_confidence_interval

def test_calculate_mean():
    """计算均值."""
    scores = [80.0, 85.0, 90.0, 75.0, 95.0]
    result = calculate_mean(scores)
    assert result == 85.0

def test_calculate_std():
    """计算标准差."""
    scores = [80.0, 85.0, 90.0, 75.0, 95.0]
    result = calculate_std(scores)
    assert abs(result - 7.91) < 0.1  # 样本标准差

def test_calculate_confidence_interval():
    """计算95%置信区间."""
    scores = [80.0, 85.0, 90.0, 75.0, 95.0]
    lower, upper = calculate_confidence_interval(scores, confidence=0.95)
    assert lower < upper
    assert lower < 85.0 < upper

def test_empty_list_raises():
    """空列表应抛出异常."""
    with pytest.raises(ValueError):
        calculate_mean([])

import math
from benchmark.core.statistics import _t_ppf

def test_t_ppf_known_values():
    """验证 _t_ppf 在已知小 df 下的基本正确性."""
    val = _t_ppf(0.975, df=1)
    assert abs(val - 12.706) < 1.0, f"df=1 p=0.975: got {val}, expected ~12.706"

    val = _t_ppf(0.975, df=5)
    assert abs(val - 2.571) < 0.05, f"df=5 p=0.975: got {val}, expected ~2.571"

    val = _t_ppf(0.95, df=10)
    assert abs(val - 1.812) < 0.05, f"df=10 p=0.95: got {val}, expected ~1.812"

def test_t_ppf_symmetry():
    """t 分布关于 0 对称: ppf(p) = -ppf(1-p)."""
    for df in [2, 5, 15, 30]:
        for p in [0.8, 0.9, 0.95, 0.975, 0.99]:
            pos = _t_ppf(p, df=df)
            neg = _t_ppf(1 - p, df=df)
            assert abs(pos + neg) < 1e-10, f"symmetry broken: df={df}, p={p}"

def test_t_ppf_monotonic():
    """ppf 应随 p 单调递增."""
    df = 20
    prev = _t_ppf(0.5, df=df)
    for p in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
        cur = _t_ppf(p, df=df)
        assert cur > prev, f"not monotonic: p={p}, prev={prev}, cur={cur}"
        prev = cur

def test_t_ppf_accuracy_df_ge_15():
    """df >= 15 时误差 < 0.001。"""
    scipy_reference = {
        (0.95, 15): 1.753050, (0.975, 15): 2.131450, (0.99, 15): 2.602480,
        (0.95, 20): 1.724718, (0.975, 20): 2.085963, (0.99, 20): 2.527977,
        (0.95, 30): 1.697261, (0.975, 30): 2.042272, (0.99, 30): 2.457262,
        (0.95, 50): 1.675905, (0.975, 50): 2.008559, (0.99, 50): 2.403272,
        (0.95, 100): 1.660234, (0.975, 100): 1.983972, (0.99, 100): 2.364217,
    }
    for (p, df), expected in scipy_reference.items():
        got = _t_ppf(p, df=df)
        assert abs(got - expected) < 0.001, f"df={df}, p={p}: got {got}, expected {expected}"

def test_t_ppf_zero():
    """ppf(0.5) 应接近 0."""
    for df in [1, 2, 5, 10, 30, 100]:
        val = _t_ppf(0.5, df=df)
        assert abs(val) < 1e-10, f"ppf(0.5, df={df}) = {val}"
