import asyncio
import pytest
from benchmark.models.schemas import ScoreResult, ScoringContext, TaskDefinition
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.composite import CompositeScorer


class _FixedScorer(BaseScorer):
    def __init__(self, score_val: float, name: str = "fixed"):
        self._score = score_val
        self._name = name

    def score(self, ctx: ScoringContext) -> ScoreResult:
        return ScoreResult(score=self._score, passed=self._score >= 60)

    def get_metric_name(self) -> str:
        return self._name


def _make_ctx() -> ScoringContext:
    return ScoringContext(
        model_answer="test", raw_output="test", expected="",
        task=TaskDefinition(task_id="t1", dimension="test", dataset="test",
            prompt="test", expected_output=""),
    )


class TestCompositeScorerInit:
    def test_valid_weights(self):
        scorer = CompositeScorer([(0.4, _FixedScorer(100)), (0.6, _FixedScorer(50))])
        assert scorer.get_metric_name() == "composite"

    def test_weights_not_sum_to_one(self):
        with pytest.raises(ValueError, match="权重之和必须等于 1.0"):
            CompositeScorer([(0.5, _FixedScorer(100)), (0.6, _FixedScorer(50))])

    def test_empty_scorers(self):
        with pytest.raises(ValueError, match="至少需要一个子评分器"):
            CompositeScorer([])

    def test_single_scorer_weight_one(self):
        scorer = CompositeScorer([(1.0, _FixedScorer(80))])
        result = scorer.score(_make_ctx())
        assert result.score == 80.0
        assert result.passed is True


class TestCompositeScorerScore:
    def test_weighted_score(self):
        scorer = CompositeScorer([(0.4, _FixedScorer(100, "a")), (0.6, _FixedScorer(50, "b"))])
        result = scorer.score(_make_ctx())
        assert result.score == pytest.approx(70.0)

    def test_passed_threshold_60(self):
        scorer = CompositeScorer([(1.0, _FixedScorer(59, "x"))])
        result = scorer.score(_make_ctx())
        assert result.passed is False
        scorer2 = CompositeScorer([(1.0, _FixedScorer(60, "x"))])
        result2 = scorer2.score(_make_ctx())
        assert result2.passed is True

    def test_details_contains_weights_and_scores(self):
        scorer = CompositeScorer([(0.3, _FixedScorer(100, "a")), (0.7, _FixedScorer(0, "b"))])
        result = scorer.score(_make_ctx())
        assert result.details["composite.weights"] == {"a": 0.3, "b": 0.7}
        assert result.details["composite.scores"] == {"a": 100.0, "b": 0.0}

    def test_sub_scorer_exception_defaults_to_100(self):
        class _BrokenScorer(BaseScorer):
            def score(self, ctx: ScoringContext) -> ScoreResult:
                raise RuntimeError("boom")
            def get_metric_name(self) -> str:
                return "broken"

        scorer = CompositeScorer([(0.5, _BrokenScorer()), (0.5, _FixedScorer(0, "ok"))])
        result = scorer.score(_make_ctx())
        assert result.score == pytest.approx(50.0)
        assert result.details["composite.scores"]["broken"] == 100.0
        assert result.details["composite.errors"]["broken"] == "RuntimeError: boom"


class TestCompositeScorerAsync:
    def test_ascore_parallel_execution(self):
        import time
        class _SlowScorer(BaseScorer):
            def __init__(self, score_val: float, name: str, delay: float):
                self._score = score_val
                self._name = name
                self._delay = delay
            def score(self, ctx: ScoringContext) -> ScoreResult:
                time.sleep(self._delay)
                return ScoreResult(score=self._score, passed=self._score >= 60)
            def get_metric_name(self) -> str:
                return self._name

        scorer = CompositeScorer([
            (0.5, _SlowScorer(100, "a", 0.2)),
            (0.5, _SlowScorer(0, "b", 0.2)),
        ])
        t0 = time.monotonic()
        result = asyncio.run(scorer.ascore(_make_ctx()))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.35
        assert result.score == pytest.approx(50.0)
