"""选择题评分器。从模型输出中提取选项字母，与期望答案字母比较."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ChoiceMatchScorer(BaseScorer):
    """选择题字母匹配评分器，用于 MMLU 选择题。

    从模型输出中提取最后的选项字母（A/B/C/D/...），
    与 expected_output（正确选项字母）进行不区分大小写的比较。
    匹配成功 score=100，失败 score=0。
    """

    # 匹配独立的选项字母（前后不是字母字符）
    _CHOICE_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected_letter = ctx.expected.strip().upper()

        # 从输出中提取所有选项字母
        matches = self._CHOICE_RE.findall(ctx.model_answer)

        if not matches:
            return ScoreResult(
                score=0,
                passed=False,
                details={
                    "error": "No choice letter found in output",
                    "raw_output": ctx.model_answer[:200],
                },
                reasoning="Model output contains no choice letter",
            )

        # 取最后一个匹配的字母（模型可能在推理过程中提到多个，最终答案通常在最后）
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
        return "choice_match"
