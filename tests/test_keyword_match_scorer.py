"""关键词匹配评分器测试."""

import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.keyword_match_scorer import KeywordMatchScorer


def _make_ctx(model_answer: str, keywords: list[str] | None = None) -> ScoringContext:
    """构造 ScoringContext，keywords 放在 task.metadata 中."""
    return ScoringContext(
        model_answer=model_answer,
        raw_output=model_answer,
        expected="",
        task=TaskDefinition(
            task_id="test",
            dimension="frontend-dev",
            dataset="frontcode",
            prompt="",
            expected_output="",
            metadata={"keywords": keywords or []},
        ),
    )


def test_score_with_all_keywords():
    """包含所有关键词得满分."""
    scorer = KeywordMatchScorer()
    ctx = _make_ctx('<div class="button">Click</div>', keywords=["div", "class", "button"])
    result = scorer.score(ctx)
    assert result.score == 100.0
    assert result.passed is True


def test_score_with_partial_keywords():
    """包含部分关键词按比例得分."""
    scorer = KeywordMatchScorer()
    ctx = _make_ctx(
        '<div class="button">Click</div>',
        keywords=["div", "class", "button", "onclick", "addEventListener"],
    )
    result = scorer.score(ctx)
    assert result.score == 60.0  # 3/5 = 60%


def test_score_with_regex_patterns():
    """支持正则表达式匹配."""
    scorer = KeywordMatchScorer(use_regex=True)
    ctx = _make_ctx(
        'function handleClick() { const x = 1; }',
        keywords=[r"function\s+\w+", r"const\s+\w+\s*="],
    )
    result = scorer.score(ctx)
    assert result.score == 100.0


def test_score_with_no_match():
    """没有匹配返回0分."""
    scorer = KeywordMatchScorer()
    ctx = _make_ctx('p paragraph text', keywords=["div", "span"])
    result = scorer.score(ctx)
    assert result.score == 0.0
    assert result.passed is False


def test_no_keywords_configured():
    """keywords 为空列表时返回 error."""
    scorer = KeywordMatchScorer()
    ctx = _make_ctx("some output", keywords=[])
    result = scorer.score(ctx)
    assert result.score == 0.0
    assert result.passed is False
    assert "error" in result.details
    assert result.details["error"] == "No keywords configured"


def test_get_metric_name():
    """get_metric_name 返回正确指标名."""
    scorer = KeywordMatchScorer()
    assert scorer.get_metric_name() == "keyword_match"
