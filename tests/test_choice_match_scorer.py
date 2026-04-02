# tests/test_choice_match_scorer.py
import pytest

from benchmark.models.schemas import TaskDefinition
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


def test_correct_choice():
    scorer = ChoiceMatchScorer()
    task = _make_task()
    result = scorer.score("B", "B", task)
    assert result.passed is True
    assert result.score == 100.0


def test_correct_choice_with_explanation():
    scorer = ChoiceMatchScorer()
    task = _make_task()
    result = scorer.score("The answer is B because...", "B", task)
    assert result.passed is True


def test_case_insensitive():
    scorer = ChoiceMatchScorer()
    task = _make_task()
    result = scorer.score("b", "B", task)
    assert result.passed is True


def test_wrong_choice():
    scorer = ChoiceMatchScorer()
    task = _make_task()
    result = scorer.score("A", "B", task)
    assert result.passed is False
    assert result.score == 0.0


def test_no_choice_letter():
    scorer = ChoiceMatchScorer()
    task = _make_task()
    result = scorer.score("maybe 42", "B", task)
    assert result.passed is False
    assert "No choice letter" in result.details["error"]


def test_get_metric_name():
    scorer = ChoiceMatchScorer()
    assert scorer.get_metric_name() == "choice_match"
