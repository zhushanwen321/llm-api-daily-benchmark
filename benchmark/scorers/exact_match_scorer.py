"""精确匹配评分器。从模型输出中提取数字，与期望答案数值比较."""

from __future__ import annotations

import math
import re

from benchmark.models.schemas import ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """精确匹配评分器，用于 reasoning 维度（GSM8K）。

    从模型输出中提取最后一个数字作为预测答案，
    与 expected_output 中的期望答案进行数值比较。
    使用 math.isclose 容忍浮点精度差异。
    匹配成功 score=100，失败 score=0。
    """

    # 匹配整数或小数，不匹配单独的减号
    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def score(
        self,
        model_output: str,
        expected: str,
        task: TaskDefinition,  # noqa: ARG002 — 基类接口要求
    ) -> ScoreResult:
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

        # 先尝试字符串精确匹配（快速路径）
        if predicted_str == expected_str:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted_str, "expected": expected_str},
                reasoning=f"Correct: predicted={predicted_str}",
            )

        # 字符串不匹配时，尝试数值比较（处理 "42" vs "42.00" 等）
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
