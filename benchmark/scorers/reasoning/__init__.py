from __future__ import annotations

from typing import Any

from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer


def create_reasoning_composite(llm: Any = None) -> list[tuple[float, Any]]:
    if llm is None:
        from unittest.mock import MagicMock
        llm = MagicMock()
    return [
        (0.40, AnswerCorrectnessScorer()),
        (0.25, ReasoningCompletenessScorer()),
        (0.20, ReasoningValidityScorer(llm=llm)),
        (0.10, MethodEleganceScorer()),
        (0.05, DifficultyAdaptationScorer()),
    ]
