from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.exact_match_scorer import ExactMatchScorer


def _make_ctx(answer: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=answer,
        raw_output=answer,
        expected=expected,
        task=TaskDefinition(
            task_id="test",
            dimension="reasoning",
            dataset="gsm8k",
            prompt="test",
            expected_output=expected,
        ),
    )


def test_exact_match_correct():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("42", "42"))
    assert result.passed is True
    assert result.score == 100.0


def test_exact_match_numeric_close():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("42.0", "42"))
    assert result.passed is True


def test_exact_match_wrong():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("99", "42"))
    assert result.passed is False
    assert result.score == 0.0


def test_exact_match_no_number():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("no number here", "42"))
    assert result.passed is False
    assert "No number" in result.details["error"]
