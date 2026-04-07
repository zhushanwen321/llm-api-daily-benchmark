"""答案正确性评分器 - 基于 reasoning_content 提取选项并比较."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class AnswerCorrectnessScorer(BaseScorer):
    """答案正确性评分器 (30%) - 复用 ChoiceMatchScorer 逻辑.

    从 reasoning_content 中提取最后的选项字母（A/B/C/D/...），
    与 expected 比较不区分大小写。
    匹配成功 score=100，失败 score=0。
    reasoning_content 为空时返回 100 分。
    """

    _CHOICE_RE = re.compile(r"\b([A-Z])\b")

    def score(self, ctx: ScoringContext) -> ScoreResult:
        # 空推理内容时返回 100 分
        if not ctx.reasoning_content or not ctx.reasoning_content.strip():
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"reason": "empty_reasoning"},
                reasoning="Empty reasoning_content, default to full score",
            )

        expected_letter = ctx.expected.strip().upper()

        # 从推理内容中提取所有选项字母
        matches = self._CHOICE_RE.findall(ctx.reasoning_content)

        if not matches:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={
                    "error": "No choice letter found in reasoning_content",
                    "reasoning_preview": ctx.reasoning_content[:200],
                },
                reasoning="Reasoning content contains no choice letter",
            )

        # 取最后一个匹配的字母
        predicted = matches[-1].upper()

        passed = predicted == expected_letter
        score = 100.0 if passed else 0.0
        return ScoreResult(
            score=score,
            passed=passed,
            details={"predicted": predicted, "expected": expected_letter},
            reasoning=(
                f"Correct: predicted={predicted}"
                if passed
                else f"Incorrect: predicted={predicted}, expected={expected_letter}"
            ),
        )

    def get_metric_name(self) -> str:
        return "answer_correctness"
