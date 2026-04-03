import pytest

from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.choice_match_scorer import ChoiceMatchScorer


def _make_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="mmlu_test_1",
        dimension="system-architecture",
        dataset="mmlu",
        prompt="Test",
        expected_output="B",
        metadata={},
    )


def _make_ctx(model_answer: str, expected: str = "B") -> ScoringContext:
    return ScoringContext(
        model_answer=model_answer,
        raw_output=model_answer,
        expected=expected,
        task=_make_task(),
    )


def test_correct_choice():
    scorer = ChoiceMatchScorer()
    ctx = _make_ctx("B", "B")
    result = scorer.score(ctx)
    assert result.passed is True
    assert result.score == 100.0


def test_correct_choice_with_explanation():
    scorer = ChoiceMatchScorer()
    ctx = _make_ctx("The answer is B because...", "B")
    result = scorer.score(ctx)
    assert result.passed is True


def test_case_insensitive():
    scorer = ChoiceMatchScorer()
    ctx = _make_ctx("b", "B")
    result = scorer.score(ctx)
    assert result.passed is True


def test_wrong_choice():
    scorer = ChoiceMatchScorer()
    ctx = _make_ctx("A", "B")
    result = scorer.score(ctx)
    assert result.passed is False
    assert result.score == 0.0


def test_no_choice_letter():
    scorer = ChoiceMatchScorer()
    ctx = _make_ctx("maybe 42", "B")
    result = scorer.score(ctx)
    assert result.passed is False
    assert "No choice letter" in result.details["error"]


def test_get_metric_name():
    scorer = ChoiceMatchScorer()
    assert scorer.get_metric_name() == "choice_match"
