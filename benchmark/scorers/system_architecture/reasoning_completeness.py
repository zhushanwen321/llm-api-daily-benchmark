"""推理完整性评分器 - 评估推理过程的完整性."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ReasoningCompletenessScorer(BaseScorer):
    """推理完整性评分器 (25%) - 评估推理过程的完整性.

    空推理 → 100 分
    长度 < 50 字符扣分
    提到选项字母 +20 分 (option_coverage)
    推理步骤关键词 +3 分/个 (上限 15)
    基础分 100 - 长度罚分 + 选项加分 + 步骤加分
    """

    # 选项字母正则
    _OPTION_RE = re.compile(r"\b([A-Z])\b")

    # 推理步骤关键词
    _STEP_KEYWORDS = [
        "because",
        "therefore",
        "since",
        "however",
        "first",
        "second",
        "third",
        "finally",
        "thus",
        "hence",
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
        base_score = 100.0

        # 长度罚分：长度 < 50 字符扣分
        length = len(reasoning.strip())
        length_penalty = 0.0
        if length < 50:
            length_penalty = (50 - length) * 2  # 每少 1 字符扣 2 分

        # 选项覆盖加分：提到选项字母 +20 分
        option_matches = self._OPTION_RE.findall(reasoning)
        option_bonus = 20.0 if option_matches else 0.0

        # 推理步骤加分：关键词 +3 分/个 (上限 15)
        reasoning_lower = reasoning.lower()
        step_count = sum(1 for kw in self._STEP_KEYWORDS if kw in reasoning_lower)
        step_bonus = min(step_count * 3.0, 15.0)

        # 计算最终分数
        final_score = base_score - length_penalty + option_bonus + step_bonus
        final_score = max(0.0, min(100.0, final_score))

        return ScoreResult(
            score=final_score,
            passed=final_score >= 60.0,
            details={
                "length": length,
                "length_penalty": length_penalty,
                "option_matches": option_matches,
                "option_bonus": option_bonus,
                "step_count": step_count,
                "step_bonus": step_bonus,
            },
            reasoning=(
                f"Length: {length}, penalty: {length_penalty:.1f}; "
                f"Option coverage: {option_bonus:.1f}; "
                f"Step keywords: {step_count}, bonus: {step_bonus:.1f}"
            ),
        )

    def get_metric_name(self) -> str:
        return "reasoning_completeness"
