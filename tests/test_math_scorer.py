import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.math_scorer import MathScorer


def _make_ctx(predicted: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=predicted,
        raw_output=predicted,
        expected=expected,
        task=TaskDefinition(
            task_id="math_test",
            dimension="reasoning",
            dataset="math",
            prompt="test",
            expected_output=expected,
        ),
    )


def test_pure_number_match():
    scorer = MathScorer()
    assert scorer.score(_make_ctx("42", "42")).passed is True


def test_numeric_close():
    scorer = MathScorer()
    assert scorer.score(_make_ctx("42.0", "42")).passed is True


def test_fraction_vs_decimal():
    scorer = MathScorer()
    result = scorer.score(_make_ctx(r"\frac{14}{3}", r"\frac{14}{3}"))
    assert result.passed is True


def test_fraction_equivalent():
    scorer = MathScorer()
    result = scorer.score(_make_ctx("14/3", r"\frac{14}{3}"))
    assert result.passed is True


def test_degree_match():
    scorer = MathScorer()
    result = scorer.score(_make_ctx(r"90^\circ", r"90^\circ"))
    assert result.passed is True


def test_algebraic_expression_match():
    scorer = MathScorer()
    result = scorer.score(_make_ctx("p - q", "p - q"))
    assert result.passed is True


def test_wrong_answer():
    scorer = MathScorer()
    result = scorer.score(_make_ctx("99", "42"))
    assert result.passed is False
    assert result.score == 0.0


def test_get_metric_name():
    scorer = MathScorer()
    assert scorer.get_metric_name() == "math_match"
