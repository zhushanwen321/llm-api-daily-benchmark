"""集成测试: 验证 CompositeScorer 与 DIMENSION_REGISTRY 的配合。"""

import pytest

from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.composite import CompositeScorer


def _make_backend_ctx(code: str, test: str = "", canonical: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="test",
            dimension="backend-dev",
            dataset="bigcodebench",
            prompt="test",
            expected_output="",
            metadata={"test": test, "entry_point": "", "canonical_solution": canonical},
        ),
    )


def _make_sysarch_ctx(
    answer: str = "B", expected: str = "B", reasoning: str = ""
) -> ScoringContext:
    return ScoringContext(
        model_answer=answer,
        raw_output=answer,
        expected=expected,
        task=TaskDefinition(
            task_id="test",
            dimension="system-architecture",
            dataset="mmlu-pro",
            prompt="test",
            expected_output=expected,
            metadata={"category": "computer science", "num_options": 4},
        ),
        reasoning_content=reasoning,
    )


class TestBackendCompositeIntegration:
    def test_creates_backend_composite(self):
        from benchmark.scorers.backend import create_backend_composite

        scorers = create_backend_composite()
        assert isinstance(scorers, list)
        scorer = CompositeScorer(scorers)
        code = "def add(a, b):\n    return a + b\n"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
        result = scorer.score(_make_backend_ctx(code, test))
        assert result.score > 0
        assert "composite.weights" in result.details
        assert len(result.details["composite.weights"]) == 7

    def test_backend_passed_threshold(self):
        from benchmark.scorers.backend import create_backend_composite

        scorer = CompositeScorer(create_backend_composite())
        code = "def add(a, b):\n    return a + b\n"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
        result = scorer.score(_make_backend_ctx(code, test))
        assert result.passed is True


class TestSysArchCompositeIntegration:
    pytestmark = pytest.mark.skip(reason="system-architecture dimension removed")

    def test_creates_sysarch_composite(self):
        from benchmark.scorers.system_architecture import create_sysarch_composite

        scorers = create_sysarch_composite()
        assert isinstance(scorers, list)
        scorer = CompositeScorer(scorers)
        reasoning = (
            "Let me analyze each option:\n"
            "A is incorrect because it violates the principle.\n"
            "C can be ruled out.\n"
            "D is wrong.\n"
            "Therefore, B is clearly the correct answer."
        )
        result = scorer.score(_make_sysarch_ctx("B", "B", reasoning))
        assert result.score > 0
        assert len(result.details["composite.weights"]) == 5

    def test_sysarch_correct_answer_with_reasoning(self):
        from benchmark.scorers.system_architecture import create_sysarch_composite

        scorer = CompositeScorer(create_sysarch_composite())
        reasoning = (
            "Let me analyze each option:\n"
            "A is incorrect because it violates the principle.\n"
            "C can be ruled out since it doesn't apply.\n"
            "D is wrong because it contradicts the premise.\n"
            "Therefore, B is clearly the correct answer."
        )
        result = scorer.score(_make_sysarch_ctx("B", "B", reasoning))
        assert result.passed is True
