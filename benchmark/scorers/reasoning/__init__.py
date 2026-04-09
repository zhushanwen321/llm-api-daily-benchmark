from __future__ import annotations

import os
from typing import Any

from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
from benchmark.scorers.reasoning.reasoning_completeness import (
    ReasoningCompletenessScorer,
)
from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer


def _is_weighted_scoring_enabled() -> bool:
    value = os.getenv("WEIGHTED_SCORING", "false").lower()
    return value in ("true", "1", "yes", "on")


def create_reasoning_composite() -> list[tuple[float, Any]]:
    if _is_weighted_scoring_enabled():
        return [
            (0.50, AnswerCorrectnessScorer()),
            (0.30, ReasoningCompletenessScorer()),
            (0.15, MethodEleganceScorer()),
            (0.05, DifficultyAdaptationScorer()),
        ]
    return [
        (1.00, AnswerCorrectnessScorer()),
    ]
