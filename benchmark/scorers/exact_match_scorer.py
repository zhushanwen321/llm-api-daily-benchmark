"""精确匹配评分器。从模型输出中提取数字，与期望答案数值比较."""

from __future__ import annotations

import math
import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """精确匹配评分器，用于 reasoning 维度（GSM8K）。"""

    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def score(self, ctx: ScoringContext) -> ScoreResult:
        model_output = ctx.model_answer
        expected = ctx.expected

        numbers = self._NUMBER_RE.findall(model_output)
        if not numbers:
            return ScoreResult(
                score=0,
                passed=False,
                details={
                    "error": "No number found in output",
                    "raw_output": model_output[:200],
                },
                reasoning="Model output contains no numeric answer",
            )

        predicted_str = numbers[-1]
        expected_str = expected.strip()

        if predicted_str == expected_str:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted_str, "expected": expected_str},
                reasoning=f"Correct: predicted={predicted_str}",
            )

        try:
            predicted_val = float(predicted_str)
            expected_val = float(expected_str)
            passed = math.isclose(predicted_val, expected_val, rel_tol=1e-9)
        except ValueError:
            passed = False

        score = 100.0 if passed else 0.0
        return ScoreResult(
            score=score,
            passed=passed,
            details={"predicted": predicted_str, "expected": expected_str},
            reasoning=(
                f"Correct: predicted={predicted_str}"
                if passed
                else f"Incorrect: predicted={predicted_str}, expected={expected_str}"
            ),
        )

    def get_metric_name(self) -> str:
        return "exact_match"
