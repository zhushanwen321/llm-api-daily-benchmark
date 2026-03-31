"""精确匹配评分器。从模型输出中提取数字，与期望答案精确比较."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """精确匹配评分器，用于 reasoning 维度（GSM8K）。
    从模型输出中提取最后一个数字作为预测答案，与 expected_output 中的期望答案进行字符串精确比较。
    匹配成功 score=100，失败 score=0。
    """

    def score(
        self,
        model_output: str,
        expected: str,
        task: TaskDefinition,  # noqa: ARG002 — 基类接口要求
    ) -> ScoreResult:
        numbers = re.findall(r"-?\d+\.?\d*", model_output)
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
        predicted = numbers[-1]
        passed = predicted == expected.strip()
        score = 100.0 if passed else 0.0
        return ScoreResult(
            score=score,
            passed=passed,
            details={"predicted": predicted, "expected": expected.strip()},
            reasoning=(
                f"Correct: predicted={predicted}"
                if passed
                else f"Incorrect: predicted={predicted}, expected={expected.strip()}"
            ),
        )

    def get_metric_name(self) -> str:
        return "exact_match"
