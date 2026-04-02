"""关键词匹配评分器测试."""

import pytest
from benchmark.models.schemas import TaskDefinition, ScoreResult
from benchmark.scorers.keyword_match_scorer import KeywordMatchScorer


def test_score_with_all_keywords():
    """包含所有关键词得满分."""
    scorer = KeywordMatchScorer(keywords=["div", "class", "button"])
    task = TaskDefinition(
        task_id="test_1",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="Create a div with class button",
        expected_output="",
        metadata={}
    )
    result = scorer.score('<div class="button">Click</div>', "", task)
    assert result.score == 100.0
    assert result.passed is True


def test_score_with_partial_keywords():
    """包含部分关键词按比例得分."""
    scorer = KeywordMatchScorer(keywords=["div", "class", "button", "onclick", "addEventListener"])
    task = TaskDefinition(
        task_id="test_2",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="",
        expected_output="",
        metadata={}
    )
    result = scorer.score('<div class="button">Click</div>', "", task)
    assert result.score == 60.0  # 3/5 = 60%


def test_score_with_regex_patterns():
    """支持正则表达式匹配."""
    scorer = KeywordMatchScorer(
        keywords=[r"function\s+\w+", r"const\s+\w+\s*="],
        use_regex=True
    )
    task = TaskDefinition(
        task_id="test_3",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="",
        expected_output="",
        metadata={}
    )
    result = scorer.score('function handleClick() { const x = 1; }', "", task)
    assert result.score == 100.0


def test_score_with_no_match():
    """没有匹配返回0分."""
    scorer = KeywordMatchScorer(keywords=["div", "span"])
    task = TaskDefinition(
        task_id="test_4",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="",
        expected_output="",
        metadata={}
    )
    result = scorer.score('p paragraph text', "", task)
    assert result.score == 0.0
    assert result.passed is False
