from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.execution_scorer import ExecutionScorer


def _make_ctx(code: str, test_code: str = "", entry_point: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="test_exec",
            dimension="backend-dev",
            dataset="bigcodebench",
            prompt="test",
            expected_output="",
            metadata={"test": test_code, "entry_point": entry_point},
        ),
    )


def test_execution_correct_code():
    code = "def add(a, b):\n    return a + b"
    test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
    scorer = ExecutionScorer(timeout=10)
    result = scorer.score(_make_ctx(code, test, "add"))
    assert result.passed is True
    assert result.score == 100.0


def test_execution_wrong_code():
    code = "def add(a, b):\n    return a - b"
    test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
    scorer = ExecutionScorer(timeout=10)
    result = scorer.score(_make_ctx(code, test, "add"))
    assert result.passed is False
    assert result.score == 0.0


def test_execution_empty_output():
    scorer = ExecutionScorer()
    result = scorer.score(_make_ctx(""))
    assert result.passed is False
    assert result.details["error"] == "Empty model output"
