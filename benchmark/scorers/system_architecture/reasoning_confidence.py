"""推理置信度评分器 - 评估推理过程的确定性."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ReasoningConfidenceScorer(BaseScorer):
    """推理置信度评分器 (15%) - 评估推理过程的确定性.

    空推理 → 100 分
    确定性关键词 +5 分/个 (上限 40)
    不确定性关键词 -8 分/个 (上限 50)
    基础分 60 + 确定性 - 不确定性
    """

    # 确定性关键词
    _CERTAINTY_KEYWORDS = [
        "clearly",
        "definitely",
        "certainly",
        "must be",
        "undoubtedly",
        "absolutely",
        "unquestionably",
        "obviously",
    ]

    # 不确定性关键词
    _UNCERTAINTY_KEYWORDS = [
        "maybe",
        "perhaps",
        "not sure",
        "could be",
        "might be",
        "I think",
        "probably",
        "possibly",
        "likely",
        "seems",
        "appears",
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
        base_score = 60.0

        # 确定性加分：关键词 +5 分/个 (上限 40)
        reasoning_lower = reasoning.lower()
        certainty_count = sum(
            1 for kw in self._CERTAINTY_KEYWORDS if kw in reasoning_lower
        )
        certainty_bonus = min(certainty_count * 5.0, 40.0)

        # 不确定性减分：关键词 -8 分/个 (上限 50)
        uncertainty_count = sum(
            1 for kw in self._UNCERTAINTY_KEYWORDS if kw in reasoning_lower
        )
        uncertainty_penalty = min(uncertainty_count * 8.0, 50.0)

        # 计算最终分数
        final_score = base_score + certainty_bonus - uncertainty_penalty
        final_score = max(0.0, min(100.0, final_score))

        return ScoreResult(
            score=final_score,
            passed=final_score >= 50.0,
            details={
                "certainty_count": certainty_count,
                "certainty_bonus": certainty_bonus,
                "uncertainty_count": uncertainty_count,
                "uncertainty_penalty": uncertainty_penalty,
            },
            reasoning=(
                f"Certainty: {certainty_count}, bonus: {certainty_bonus:.1f}; "
                f"Uncertainty: {uncertainty_count}, penalty: {uncertainty_penalty:.1f}"
            ),
        )

    def get_metric_name(self) -> str:
        return "reasoning_confidence"
