from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_DEPTH_MARKERS = [
    "因为", "所以", "因此", "由于", "故",
    "because", "therefore", "thus", "since",
    "步骤", "首先", "其次", "最后",
    "step", "first", "second", "finally",
]
_EXPECTED_DEPTH = {3: 3, 4: 5, 5: 7}


class DifficultyAdaptationScorer(BaseScorer):
    """基于推理深度与难度等级的匹配度评估。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(
                score=100.0, passed=True,
                details={"reason": "empty_reasoning"},
                reasoning="No reasoning content, default 100",
            )

        reasoning_lower = reasoning.lower()
        depth = sum(1 for m in _DEPTH_MARKERS if m in reasoning_lower)

        level = ctx.task.metadata.get("level", 3)
        expected = _EXPECTED_DEPTH.get(level, 3)
        depth_match = 1 - abs(depth - expected) / expected
        score = max(0.0, depth_match * 100)

        return ScoreResult(
            score=round(score, 1), passed=score >= 50.0,
            details={
                "actual_depth": depth,
                "expected_depth": expected,
                "depth_match": round(depth_match, 2),
            },
            reasoning=f"Depth: actual={depth}, expected={expected}, match={depth_match:.2f}",
        )

    def get_metric_name(self) -> str:
        return "difficulty_adaptation"
