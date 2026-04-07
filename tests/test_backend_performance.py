"""PerformanceScorer 单元测试。"""

import pytest

from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.performance import PerformanceScorer


def _make_ctx(code: str, canonical: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="test_perf",
            dimension="backend-dev",
            dataset="bigcodebench",
            prompt="test",
            expected_output="",
            metadata={"test": "", "entry_point": "", "canonical_solution": canonical},
        ),
    )


def test_no_canonical_solution():
    """没有 canonical_solution 时应返回 100 分。"""
    scorer = PerformanceScorer()
    result = scorer.score(_make_ctx("def f(): return 1", ""))
    assert result.score == 100.0


def test_similar_performance():
    """两段代码性能接近，应该得 100 分。"""
    code1 = "def f(): return sum(range(10))"
    code2 = "def f(): return sum(range(10))"
    scorer = PerformanceScorer()
    result = scorer.score(_make_ctx(code1, code2))
    assert result.score == 100.0


def test_execution_error_graceful():
    """生成代码有语法错误，不应崩溃。"""
    scorer = PerformanceScorer()
    result = scorer.score(_make_ctx("def f(:", "def f(): return 1"))
    assert result.score == 100.0  # graceful skip


def test_canonical_execution_error_graceful():
    """canonical_solution 执行出错。"""
    scorer = PerformanceScorer()
    result = scorer.score(_make_ctx("def f(): return 1", "def f(:"))
    assert result.score == 100.0  # graceful skip


def test_get_metric_name():
    """验证 metric name。"""
    assert PerformanceScorer().get_metric_name() == "performance"
