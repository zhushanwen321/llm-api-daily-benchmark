"""学科适应性评分器 - 评估推理长度是否适合学科特点."""

from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class SubjectAdaptationScorer(BaseScorer):
    """学科适应性评分器 (10%) - 评估推理长度是否适合学科特点.

    空推理 → 100 分
    学科期望长度: CS (100-500), math (80-400), physics (100-500), 未知 (50-600)
    范围内 → 100 分
    过短 → 线性缩放
    过长 → 线性扣分
    """

    # 学期望长度范围 (min, max)
    _SUBJECT_RANGES = {
        "computer science": (100, 500),
        "mathematics": (80, 400),
        "math": (80, 400),
        "physics": (100, 500),
        "default": (50, 600),
    }

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

        # 获取学科信息
        category = ctx.task.metadata.get("category", "").lower()
        min_len, max_len = self._SUBJECT_RANGES.get(
            category, self._SUBJECT_RANGES["default"]
        )

        length = len(reasoning.strip())

        # 在范围内 → 100 分
        if min_len <= length <= max_len:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"length": length, "range": (min_len, max_len), "category": category},
                reasoning=f"Length {length} within expected range [{min_len}, {max_len}]",
            )

        # 过短 → 线性缩放
        if length < min_len:
            ratio = length / min_len
            score = ratio * 100.0
            return ScoreResult(
                score=score,
                passed=score >= 60.0,
                details={
                    "length": length,
                    "min_expected": min_len,
                    "max_expected": max_len,
                    "category": category,
                    "ratio": ratio,
                },
                reasoning=f"Length {length} too short (expected ≥{min_len}), score: {score:.1f}",
            )

        # 过长 → 线性扣分
        excess = length - max_len
        penalty_ratio = excess / max_len  # 超出比例
        score = 100.0 - penalty_ratio * 100.0
        score = max(0.0, score)

        return ScoreResult(
            score=score,
            passed=score >= 60.0,
            details={
                "length": length,
                "min_expected": min_len,
                "max_expected": max_len,
                "category": category,
                "excess": excess,
            },
            reasoning=f"Length {length} too long (expected ≤{max_len}), score: {score:.1f}",
        )

    def get_metric_name(self) -> str:
        return "subject_adaptation"
