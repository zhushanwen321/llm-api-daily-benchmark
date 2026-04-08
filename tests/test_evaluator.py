import asyncio
import pytest
from unittest.mock import AsyncMock

from benchmark.core.evaluator import SingleTurnEvaluator
from benchmark.models.schemas import GenerateResponse, TaskDefinition


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.agenerate.return_value = GenerateResponse(
        content='{"answer": "42"}',
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_content="thinking...",
        duration=1.0,
        ttft=0.2,
        ttft_content=0.3,
    )
    return llm


def test_single_turn_evaluator_reasoning(mock_llm):
    """reasoning 维度: 从 JSON 提取 answer 字段."""

    async def _run():
        task = TaskDefinition(
            task_id="gsm8k_1",
            dimension="reasoning",
            dataset="gsm8k",
            prompt="What is 6 * 7?",
            expected_output="42",
        )
        evaluator = SingleTurnEvaluator()
        ctx = await evaluator.evaluate(task, "test/model", mock_llm)

        assert ctx.model_answer == "42"
        assert ctx.raw_output == '{"answer": "42"}'
        assert ctx.expected == "42"
        assert ctx.task.task_id == "gsm8k_1"
        assert ctx.reasoning_content == "thinking..."
        assert ctx.gen_metrics is not None
        assert ctx.gen_metrics["prompt_tokens"] == 10
        assert ctx.gen_metrics["completion_tokens"] == 5

    asyncio.run(_run())


def test_single_turn_evaluator_backend_dev(mock_llm):
    """backend-dev 维度: 从 JSON 提取 code 字段."""

    async def _run():
        mock_llm.agenerate.return_value = GenerateResponse(
            content='{"code": "def foo(): pass"}',
            prompt_tokens=10,
            completion_tokens=5,
        )
        task = TaskDefinition(
            task_id="bigcodebench_1",
            dimension="backend-dev",
            dataset="bigcodebench",
            prompt="Write a function",
            expected_output="",
        )
        evaluator = SingleTurnEvaluator()
        ctx = await evaluator.evaluate(task, "test/model", mock_llm)

        assert ctx.model_answer == "def foo(): pass"
        assert ctx.expected == ""

    asyncio.run(_run())
