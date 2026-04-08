import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from benchmark.models.schemas import ScoreResult, ScoringContext, TaskDefinition, GenerateResponse
from benchmark.scorers.composite import CompositeScorer


def make_reasoning_ctx(
    predicted: str = "42",
    expected: str = "42",
    reasoning: str = "因为 x=42，所以答案为 42。首先计算 x+1=43，其次验证 x-1=41，最后得出 x=42。",
    level: int = 3,
    subject: str = "Algebra",
) -> ScoringContext:
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
            metadata={"level": level, "subject": subject, "source": "test"},
        ),
        reasoning_content=reasoning,
    )


class TestAnswerCorrectnessScorer:
    def test_exact_match(self):
        from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("42", "42"))
        assert r.passed is True and r.score == 100.0

    def test_numeric_match(self):
        from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("42.0", "42"))
        assert r.passed is True

    def test_fraction_match(self):
        from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx(r"\frac{14}{3}", r"\frac{14}{3}"))
        assert r.passed is True

    def test_wrong_answer(self):
        from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("99", "42"))
        assert r.passed is False and r.score == 0.0

    def test_empty_reasoning_still_scores(self):
        from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("42", "42", reasoning=""))
        assert r.passed is True

    def test_get_metric_name(self):
        from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
        assert AnswerCorrectnessScorer().get_metric_name() == "answer_correctness"


class TestReasoningCompletenessScorer:
    def test_empty_reasoning_gives_100(self):
        from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
        r = ReasoningCompletenessScorer().score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_long_reasoning_high_score(self):
        from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
        long_reasoning = "因为。所以。因此。由于。故。因为。所以。因此。由于。故。" * 50
        r = ReasoningCompletenessScorer().score(make_reasoning_ctx(reasoning=long_reasoning))
        assert r.score > 50.0

    def test_short_reasoning_low_score(self):
        from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
        r = ReasoningCompletenessScorer().score(make_reasoning_ctx(reasoning="答案", level=5))
        assert r.score < 50.0

    def test_structural_markers_boost(self):
        from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
        reasoning = "首先。其次。最后。因为。所以。因此。" * 100
        r = ReasoningCompletenessScorer().score(make_reasoning_ctx(reasoning=reasoning, level=3))
        assert r.score >= 60.0

    def test_get_metric_name(self):
        from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
        assert ReasoningCompletenessScorer().get_metric_name() == "reasoning_completeness"


class TestReasoningValidityScorer:
    def test_score_raises_not_implemented(self):
        from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
        scorer = ReasoningValidityScorer(llm=MagicMock())
        with pytest.raises(NotImplementedError):
            scorer.score(make_reasoning_ctx(reasoning=""))

    def test_empty_reasoning_async(self):
        from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
        scorer = ReasoningValidityScorer(llm=MagicMock())
        r = asyncio.run(scorer.ascore(make_reasoning_ctx(reasoning="")))
        assert r.score == 100.0

    def test_llm_judge_call(self):
        from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
        mock_llm = MagicMock()
        mock_llm.agenerate = AsyncMock(return_value=GenerateResponse(
            content='{"logical_consistency": 40, "math_facts": 40, "computation": 20}',
        ))
        scorer = ReasoningValidityScorer(llm=mock_llm)
        r = asyncio.run(
            scorer.ascore(make_reasoning_ctx(reasoning="因为 2+2=4，所以答案为 4。"))
        )
        assert r.score == 100.0
        mock_llm.agenerate.assert_called_once()

    def test_cache_hit(self):
        from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
        mock_llm = MagicMock()
        mock_llm.agenerate = AsyncMock(return_value=GenerateResponse(
            content='{"logical_consistency": 40, "math_facts": 40, "computation": 20}',
        ))
        scorer = ReasoningValidityScorer(llm=mock_llm)
        ctx = make_reasoning_ctx(reasoning="test reasoning content")
        asyncio.run(scorer.ascore(ctx))
        asyncio.run(scorer.ascore(ctx))
        assert mock_llm.agenerate.call_count == 1

    def test_malformed_json_graceful(self):
        from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
        mock_llm = MagicMock()
        mock_llm.agenerate = AsyncMock(return_value=GenerateResponse(content="not json at all"))
        scorer = ReasoningValidityScorer(llm=mock_llm)
        r = asyncio.run(
            scorer.ascore(make_reasoning_ctx(reasoning="some reasoning"))
        )
        assert r.score == 50.0

    def test_get_metric_name(self):
        from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
        assert ReasoningValidityScorer(llm=MagicMock()).get_metric_name() == "reasoning_validity"


class TestMethodEleganceScorer:
    def test_empty_reasoning_gives_100(self):
        from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
        r = MethodEleganceScorer().score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_keyword_match(self):
        from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
        r = MethodEleganceScorer().score(make_reasoning_ctx(
            reasoning="使用因式分解和换元法解决问题，利用对称性简化", subject="Algebra",
        ))
        assert r.score >= 30.0

    def test_redundancy_penalty(self):
        from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
        long_reasoning = "x" * 4000
        r = MethodEleganceScorer().score(make_reasoning_ctx(reasoning=long_reasoning, level=3))
        assert r.score >= 0.0

    def test_redundancy_penalty_high_level(self):
        from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
        long_reasoning = "x" * 8000
        r = MethodEleganceScorer().score(make_reasoning_ctx(reasoning=long_reasoning, level=3))
        assert r.details.get("redundancy_penalty", 0) > 0

    def test_unknown_subject(self):
        from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
        r = MethodEleganceScorer().score(make_reasoning_ctx(reasoning="some reasoning", subject="Unknown"))
        assert r.score == 0.0

    def test_get_metric_name(self):
        from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
        assert MethodEleganceScorer().get_metric_name() == "method_elegance"


class TestDifficultyAdaptationScorer:
    def test_empty_reasoning_gives_100(self):
        from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer
        r = DifficultyAdaptationScorer().score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_exact_depth_match(self):
        from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer
        reasoning = "首先，因为 x=2，所以 y=4，因此答案为 4。"
        r = DifficultyAdaptationScorer().score(make_reasoning_ctx(reasoning=reasoning, level=3))
        assert r.score >= 60.0

    def test_too_shallow(self):
        from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer
        r = DifficultyAdaptationScorer().score(make_reasoning_ctx(reasoning="答案是 42。", level=5))
        assert r.score < 50.0

    def test_get_metric_name(self):
        from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer
        assert DifficultyAdaptationScorer().get_metric_name() == "difficulty_adaptation"


class TestReasoningComposite:
    def test_composite_integration(self):
        from benchmark.scorers.reasoning import create_reasoning_composite
        subs = create_reasoning_composite()
        scorer = CompositeScorer(subs)
        ctx = make_reasoning_ctx("42", "42")
        r = scorer.score(ctx)
        assert r.score > 50.0

    def test_composite_wrong_answer(self):
        from benchmark.scorers.reasoning import create_reasoning_composite
        subs = create_reasoning_composite()
        scorer = CompositeScorer(subs)
        ctx = make_reasoning_ctx("99", "42")
        r = scorer.score(ctx)
        assert r.score < 60.0
