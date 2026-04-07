"""测试 System-Architecture 评分器."""

from __future__ import annotations

import pytest

from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.system_architecture.answer_correctness import (
    AnswerCorrectnessScorer,
)
from benchmark.scorers.system_architecture.option_analysis import OptionAnalysisScorer
from benchmark.scorers.system_architecture.reasoning_completeness import (
    ReasoningCompletenessScorer,
)
from benchmark.scorers.system_architecture.reasoning_confidence import (
    ReasoningConfidenceScorer,
)
from benchmark.scorers.system_architecture.subject_adaptation import (
    SubjectAdaptationScorer,
)


def _make_ctx(
    answer: str = "B",
    expected: str = "B",
    reasoning: str = "",
    category: str = "computer science",
    num_options: int = 4,
) -> ScoringContext:
    """创建测试用的 ScoringContext."""
    return ScoringContext(
        model_answer=answer,
        raw_output=answer,
        expected=expected,
        task=TaskDefinition(
            task_id="test",
            dimension="system-architecture",
            dataset="mmlu-pro",
            prompt="test",
            expected_output=expected,
            metadata={"category": category, "num_options": num_options},
        ),
        reasoning_content=reasoning,
    )


class TestAnswerCorrectnessScorer:
    """测试 AnswerCorrectnessScorer."""

    def test_empty_reasoning(self):
        """空推理内容应返回 100 分."""
        scorer = AnswerCorrectnessScorer()
        ctx = _make_ctx(answer="B", expected="B", reasoning="")
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True
        assert result.details["reason"] == "empty_reasoning"

    def test_correct_answer(self):
        """正确答案应返回 100 分."""
        scorer = AnswerCorrectnessScorer()
        ctx = _make_ctx(
            answer="B", expected="B", reasoning="Therefore, the answer is B."
        )
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True
        assert result.details["predicted"] == "B"

    def test_incorrect_answer(self):
        """错误答案应返回 0 分."""
        scorer = AnswerCorrectnessScorer()
        ctx = _make_ctx(
            answer="A", expected="B", reasoning="Therefore, the answer is A."
        )
        result = scorer.score(ctx)
        assert result.score == 0.0
        assert result.passed is False
        assert result.details["predicted"] == "A"
        assert result.details["expected"] == "B"

    def test_last_option_chosen(self):
        """应选择最后一个匹配的选项."""
        scorer = AnswerCorrectnessScorer()
        ctx = _make_ctx(
            answer="C",
            expected="C",
            reasoning="First, A is incorrect. Then B is also wrong. Finally, C is correct.",
        )
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.details["predicted"] == "C"

    def test_no_option_found(self):
        """未找到选项应返回 0 分."""
        scorer = AnswerCorrectnessScorer()
        ctx = _make_ctx(
            answer="42", expected="B", reasoning="The answer is forty-two."
        )
        result = scorer.score(ctx)
        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.details


class TestReasoningCompletenessScorer:
    """测试 ReasoningCompletenessScorer."""

    def test_empty_reasoning(self):
        """空推理内容应返回 100 分."""
        scorer = ReasoningCompletenessScorer()
        ctx = _make_ctx(reasoning="")
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True

    def test_short_reasoning_penalty(self):
        """短推理内容应扣分."""
        scorer = ReasoningCompletenessScorer()
        ctx = _make_ctx(reasoning="Short.")  # 6 字符
        result = scorer.score(ctx)
        assert result.score < 100.0
        assert result.details["length_penalty"] > 0

    def test_option_coverage_bonus(self):
        """提到选项字母应加分."""
        scorer = ReasoningCompletenessScorer()
        ctx = _make_ctx(
            reasoning="A is wrong. B is better. Therefore B is correct."
        )
        result = scorer.score(ctx)
        assert result.details["option_bonus"] == 20.0

    def test_step_keywords_bonus(self):
        """推理步骤关键词应加分."""
        scorer = ReasoningCompletenessScorer()
        ctx = _make_ctx(
            reasoning="First, A is wrong. Because B is better, therefore B is correct."
        )
        result = scorer.score(ctx)
        assert result.details["step_count"] >= 3
        assert result.details["step_bonus"] > 0

    def test_complete_reasoning(self):
        """完整推理应得高分."""
        scorer = ReasoningCompletenessScorer()
        ctx = _make_ctx(
            reasoning="First, A is wrong because it lacks key features. Second, B is better. Therefore B is correct. Since B meets all requirements, B is the answer."
        )
        result = scorer.score(ctx)
        assert result.score >= 80.0


class TestOptionAnalysisScorer:
    """测试 OptionAnalysisScorer."""

    def test_empty_reasoning(self):
        """空推理内容应返回 100 分."""
        scorer = OptionAnalysisScorer()
        ctx = _make_ctx(reasoning="")
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True

    def test_elimination_bonus(self):
        """排除法关键词应加分."""
        scorer = OptionAnalysisScorer()
        ctx = _make_ctx(
            reasoning="We can eliminate A. B is incorrect. C is wrong. D is not correct."
        )
        result = scorer.score(ctx)
        assert result.details["elimination_count"] >= 3
        assert result.details["elimination_bonus"] > 0

    def test_comparison_bonus(self):
        """对比分析关键词应加分."""
        scorer = OptionAnalysisScorer()
        ctx = _make_ctx(
            reasoning="Compared to A, B is better. While A has flaws, B is superior. However, C is also good."
        )
        result = scorer.score(ctx)
        assert result.details["comparison_count"] >= 2
        assert result.details["comparison_bonus"] > 0

    def test_unique_options_bonus(self):
        """不同选项字母提及应加分."""
        scorer = OptionAnalysisScorer()
        ctx = _make_ctx(reasoning="A is good. B is better. C is best. D is excellent.")
        result = scorer.score(ctx)
        assert result.details["unique_options"] == 4
        assert result.details["option_bonus"] == 8.0

    def test_comprehensive_analysis(self):
        """全面分析应得高分."""
        scorer = OptionAnalysisScorer()
        ctx = _make_ctx(
            reasoning="We must eliminate A because it's wrong. Compared to B, C is better while D is incorrect. Unlike A, B has merit. However, C is the best choice. We rule out D."
        )
        result = scorer.score(ctx)
        assert result.score >= 70.0


class TestReasoningConfidenceScorer:
    """测试 ReasoningConfidenceScorer."""

    def test_empty_reasoning(self):
        """空推理内容应返回 100 分."""
        scorer = ReasoningConfidenceScorer()
        ctx = _make_ctx(reasoning="")
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True

    def test_certainty_bonus(self):
        """确定性关键词应加分."""
        scorer = ReasoningConfidenceScorer()
        ctx = _make_ctx(reasoning="Clearly, B is correct. B must be the answer. Certainly, B is best.")
        result = scorer.score(ctx)
        assert result.details["certainty_count"] >= 2
        assert result.details["certainty_bonus"] > 0

    def test_uncertainty_penalty(self):
        """不确定性关键词应扣分."""
        scorer = ReasoningConfidenceScorer()
        ctx = _make_ctx(
            reasoning="I think B is correct. Maybe B is the answer. Probably B is best. It seems B is good."
        )
        result = scorer.score(ctx)
        assert result.details["uncertainty_count"] >= 3
        assert result.details["uncertainty_penalty"] > 0
        assert result.score < 60.0

    def test_mixed_confidence(self):
        """混合确定性和不确定性应平衡."""
        scorer = ReasoningConfidenceScorer()
        ctx = _make_ctx(
            reasoning="Clearly, B is correct. I think C might be wrong. B must be the answer."
        )
        result = scorer.score(ctx)
        assert 50.0 <= result.score <= 100.0

    def test_high_confidence(self):
        """高确定性应得高分."""
        scorer = ReasoningConfidenceScorer()
        ctx = _make_ctx(
            reasoning="Clearly, B is correct. Certainly, B must be the answer. Undoubtedly, B is best. Obviously, B is the choice."
        )
        result = scorer.score(ctx)
        assert result.score >= 80.0


class TestSubjectAdaptationScorer:
    """测试 SubjectAdaptationScorer."""

    def test_empty_reasoning(self):
        """空推理内容应返回 100 分."""
        scorer = SubjectAdaptationScorer()
        ctx = _make_ctx(reasoning="")
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True

    def test_computer_science_in_range(self):
        """CS 学科长度在范围内应得 100 分."""
        scorer = SubjectAdaptationScorer()
        ctx = _make_ctx(
            category="computer science",
            reasoning="A" * 250,  # 100-500 范围内
        )
        result = scorer.score(ctx)
        assert result.score == 100.0
        assert result.passed is True

    def test_math_in_range(self):
        """数学学科长度在范围内应得 100 分."""
        scorer = SubjectAdaptationScorer()
        ctx = _make_ctx(
            category="mathematics", reasoning="A" * 200  # 80-400 范围内
        )
        result = scorer.score(ctx)
        assert result.score == 100.0

    def test_too_short(self):
        """过短应扣分."""
        scorer = SubjectAdaptationScorer()
        ctx = _make_ctx(
            category="computer science", reasoning="A" * 50  # 低于 100
        )
        result = scorer.score(ctx)
        assert result.score < 100.0
        assert result.details["ratio"] < 1.0

    def test_too_long(self):
        """过长应扣分."""
        scorer = SubjectAdaptationScorer()
        ctx = _make_ctx(
            category="computer science", reasoning="A" * 600  # 高于 500
        )
        result = scorer.score(ctx)
        assert result.score < 100.0
        assert result.details["excess"] > 0

    def test_unknown_category_default(self):
        """未知学科使用默认范围."""
        scorer = SubjectAdaptationScorer()
        ctx = _make_ctx(
            category="unknown subject", reasoning="A" * 300  # 50-600 范围内
        )
        result = scorer.score(ctx)
        assert result.score == 100.0
