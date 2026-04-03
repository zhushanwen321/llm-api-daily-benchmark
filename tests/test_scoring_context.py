# tests/test_scoring_context.py
from benchmark.models.schemas import ScoringContext, TaskDefinition


def test_scoring_context_basic():
    """基本字段赋值和默认值."""
    task = TaskDefinition(
        task_id="test_1",
        dimension="reasoning",
        dataset="math",
        prompt="What is 2+2?",
        expected_output="4",
    )
    ctx = ScoringContext(
        model_answer="4",
        raw_output=r"The answer is \boxed{4}",
        expected="4",
        task=task,
    )
    assert ctx.model_answer == "4"
    assert ctx.raw_output == r"The answer is \boxed{4}"
    assert ctx.expected == "4"
    assert ctx.task.task_id == "test_1"
    assert ctx.reasoning_content == ""
    assert ctx.gen_metrics is None
    assert ctx.execution_trace is None
    assert ctx.execution_metrics is None


def test_scoring_context_with_all_fields():
    """所有字段赋值."""
    task = TaskDefinition(
        task_id="test_2",
        dimension="backend-dev",
        dataset="bigcodebench",
        prompt="Write a function",
        expected_output="",
    )
    ctx = ScoringContext(
        model_answer="def foo(): pass",
        raw_output="```python\ndef foo(): pass\n```",
        expected="",
        task=task,
        reasoning_content="Let me think...",
        gen_metrics={"prompt_tokens": 10, "completion_tokens": 50, "duration": 1.5},
        execution_trace=[{"tool": "exec", "result": "ok"}],
        execution_metrics={"time": 1.5},
    )
    assert ctx.reasoning_content == "Let me think..."
    assert ctx.gen_metrics["prompt_tokens"] == 10
    assert ctx.execution_trace == [{"tool": "exec", "result": "ok"}]
    assert ctx.execution_metrics == {"time": 1.5}
