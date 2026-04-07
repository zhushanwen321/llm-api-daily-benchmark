import asyncio

from benchmark.models.schemas import ScoringContext, ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class DummyScorer(BaseScorer):
    """用于测试 BaseScorer 接口的 mock 评分器."""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        return ScoreResult(
            score=100.0 if ctx.model_answer == ctx.expected else 0.0,
            passed=ctx.model_answer == ctx.expected,
        )

    def get_metric_name(self) -> str:
        return "dummy"


def _make_ctx(answer: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=answer,
        raw_output=answer,
        expected=expected,
        task=TaskDefinition(
            task_id="test",
            dimension="reasoning",
            dataset="test",
            prompt="test",
            expected_output=expected,
        ),
    )


def test_base_scorer_correct_answer():
    scorer = DummyScorer()
    result = scorer.score(_make_ctx("42", "42"))
    assert result.passed is True
    assert result.score == 100.0


def test_base_scorer_wrong_answer():
    scorer = DummyScorer()
    result = scorer.score(_make_ctx("99", "42"))
    assert result.passed is False
    assert result.score == 0.0


def test_base_scorer_ascore_delegates_to_score():
    """ascore() 默认实现应委托给 score()。"""
    scorer = DummyScorer()
    ctx = _make_ctx("42", "42")
    result = asyncio.run(scorer.ascore(ctx))
    assert result.passed is True
    assert result.score == 100.0
