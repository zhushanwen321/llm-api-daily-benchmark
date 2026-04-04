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
