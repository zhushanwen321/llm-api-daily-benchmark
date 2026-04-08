from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.math_scorer import MathScorer


class AnswerCorrectnessScorer(BaseScorer):
    """委托 MathScorer 进行答案正确性评分。"""

    def __init__(self) -> None:
        self._math = MathScorer()

    def score(self, ctx: ScoringContext) -> ScoreResult:
        return self._math.score(ctx)

    def get_metric_name(self) -> str:
        return "answer_correctness"
