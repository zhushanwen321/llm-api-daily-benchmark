from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_SUBJECT_KEYWORDS = {
    "Algebra": ["因式分解", "对称性", "换元", "韦达定理", "factorization", "symmetry", "substitution"],
    "Geometry": ["辅助线", "相似", "勾股定理", "auxiliary line", "similar", "Pythagorean"],
    "Number Theory": ["模运算", "整除", "同余", "欧拉函数", "modular", "divisibility", "Euler"],
    "Precalculus": ["三角恒等变换", "复数", "向量", "trigonometric", "complex number", "vector"],
    "Counting & Probability": ["排列", "组合", "概率", "permutation", "combination", "probability"],
    "Intermediate Algebra": ["二次方程", "不等式", "函数", "quadratic", "inequality", "function"],
    "Prealgebra": ["分数", "小数", "百分数", "fraction", "decimal", "percentage"],
}


class MethodEleganceScorer(BaseScorer):
    """基于学科关键词匹配和冗余惩罚评估方法优雅性。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(
                score=100.0, passed=True,
                details={"reason": "empty_reasoning"},
                reasoning="No reasoning content, default 100",
            )

        subject = ctx.task.metadata.get("subject", "")
        keywords = _SUBJECT_KEYWORDS.get(subject, [])
        reasoning_lower = reasoning.lower()
        matched = [kw for kw in keywords if kw.lower() in reasoning_lower]

        keyword_score = min(50.0, len(matched) * 10)

        est_tokens = len(reasoning) / 4
        level = ctx.task.metadata.get("level", 3)
        threshold = level * 500
        excess = est_tokens - threshold
        penalty = min(30.0, max(0.0, (excess / 100) * 5)) if excess > 0 else 0.0

        total = max(0.0, keyword_score - penalty)
        return ScoreResult(
            score=round(total, 1), passed=total >= 30.0,
            details={
                "matched_keywords": matched,
                "keyword_score": round(keyword_score, 1),
                "est_tokens": round(est_tokens),
                "redundancy_penalty": round(penalty, 1),
            },
            reasoning=f"Elegance: keywords={keyword_score:.0f}, penalty={penalty:.0f}",
        )

    def get_metric_name(self) -> str:
        return "method_elegance"
