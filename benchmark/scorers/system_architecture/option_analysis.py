"""选项分析评分器 - 评估模型对选项的分析深度."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class OptionAnalysisScorer(BaseScorer):
    """选项分析评分器 (20%) - 评估模型对选项的分析深度.

    空推理 → 100 分
    排除法关键词 +8 分/个 (上限 40)
    对比分析关键词 +5 分/个 (上限 20)
    选项字母提及数 +2 分/个 (上限 10)
    基础分 30 + 各项加分
    """

    # 选项字母正则
    _OPTION_RE = re.compile(r"\b([A-Z])\b")

    # 排除法关键词
    _ELIMINATION_KEYWORDS = [
        "eliminate",
        "rule out",
        "incorrect",
        "wrong",
        "not correct",
        "exclude",
        "reject",
    ]

    # 对比分析关键词
    _COMPARISON_KEYWORDS = [
        "compared to",
        "versus",
        "while",
        "whereas",
        "however",
        "although",
        "unlike",
        "instead of",
    ]

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content or ""

        # 空推理内容时返回 100 分
        if not reasoning.strip():
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"reason": "empty_reasoning"},
                reasoning="Empty reasoning_content, default to full score",
            )

        # 基础分
        base_score = 30.0

        # 排除法加分：关键词 +8 分/个 (上限 40)
        reasoning_lower = reasoning.lower()
        elimination_count = sum(
            1 for kw in self._ELIMINATION_KEYWORDS if kw in reasoning_lower
        )
        elimination_bonus = min(elimination_count * 8.0, 40.0)

        # 对比分析加分：关键词 +5 分/个 (上限 20)
        comparison_count = sum(
            1 for kw in self._COMPARISON_KEYWORDS if kw in reasoning_lower
        )
        comparison_bonus = min(comparison_count * 5.0, 20.0)

        # 选项字母提及加分：+2 分/个 (上限 10)
        option_matches = self._OPTION_RE.findall(reasoning)
        unique_options = len(set(option_matches)) if option_matches else 0
        option_bonus = min(unique_options * 2.0, 10.0)

        # 计算最终分数
        final_score = base_score + elimination_bonus + comparison_bonus + option_bonus
        final_score = min(100.0, final_score)

        return ScoreResult(
            score=final_score,
            passed=final_score >= 50.0,
            details={
                "elimination_count": elimination_count,
                "elimination_bonus": elimination_bonus,
                "comparison_count": comparison_count,
                "comparison_bonus": comparison_bonus,
                "unique_options": unique_options,
                "option_bonus": option_bonus,
            },
            reasoning=(
                f"Elimination: {elimination_count}, bonus: {elimination_bonus:.1f}; "
                f"Comparison: {comparison_count}, bonus: {comparison_bonus:.1f}; "
                f"Options: {unique_options}, bonus: {option_bonus:.1f}"
            ),
        )

    def get_metric_name(self) -> str:
        return "option_analysis"
