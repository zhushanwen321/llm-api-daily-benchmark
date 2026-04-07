from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_CONNECTORS = ["因为", "所以", "因此", "由于", "故", "because", "therefore", "thus", "since"]
_STRUCTURAL = ["步骤", "首先", "其次", "最后", "step", "first", "second", "finally"]
_MIN_TOKENS = {3: 200, 4: 400, 5: 600}


class ReasoningCompletenessScorer(BaseScorer):
    """基于推理内容长度、连接词和结构标记评估推理完整性。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(
                score=100.0, passed=True,
                details={"reason": "empty_reasoning"},
                reasoning="No reasoning content, default 100",
            )

        est_tokens = len(reasoning) / 4
        level = ctx.task.metadata.get("level", 3)
        min_tokens = _MIN_TOKENS.get(level, 200)

        length_score = min(100.0, est_tokens / min_tokens * 100)
        conn_count = sum(1 for c in _CONNECTORS if c in reasoning.lower())
        conn_score = min(100.0, conn_count * 15)
        struct_count = sum(1 for s in _STRUCTURAL if s in reasoning.lower())
        struct_score = min(100.0, struct_count * 20)

        total = length_score * 0.4 + conn_score * 0.3 + struct_score * 0.3
        return ScoreResult(
            score=round(total, 1), passed=total >= 60.0,
            details={
                "est_tokens": round(est_tokens),
                "length_score": round(length_score, 1),
                "connector_count": conn_count,
                "connector_score": round(conn_score, 1),
                "structural_count": struct_count,
                "structural_score": round(struct_score, 1),
            },
            reasoning=f"Completeness: length={length_score:.0f}, connectors={conn_score:.0f}, structure={struct_score:.0f}",
        )

    def get_metric_name(self) -> str:
        return "reasoning_completeness"
