"""评测编排器。将"如何调用模型"与"评分逻辑"分离."""
from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.response_parser import parse_response
from benchmark.models.schemas import ScoringContext, TaskDefinition


class BaseEvaluator(ABC):
    """评测编排器基类."""

    @abstractmethod
    async def evaluate(
        self,
        task: TaskDefinition,
        model: str,
        llm: LLMEvalAdapter,
    ) -> ScoringContext:
        """执行评测，返回评分上下文."""


class SingleTurnEvaluator(BaseEvaluator):
    """单轮生成：prompt -> generate -> parse."""

    async def evaluate(
        self,
        task: TaskDefinition,
        model: str,
        llm: LLMEvalAdapter,
        system_message: str | None = None,
    ) -> ScoringContext:
        response = await llm.agenerate(task.prompt, model=model, system_message=system_message)
        parsed = parse_response(response.content, task.dimension)

        # 保留 API 指标
        duration = response.duration
        completion_tokens = response.completion_tokens
        gen_metrics = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "duration": duration,
            "tokens_per_second": (
                completion_tokens / duration if duration > 0 and completion_tokens > 0 else 0.0
            ),
            "ttft": response.ttft,
            "ttft_content": response.ttft_content,
            "truncated": response.truncated,
            "finish_reason": response.finish_reason,
        }

        return ScoringContext(
            model_answer=parsed.answer,
            raw_output=response.content,
            expected=task.expected_output,
            task=task,
            reasoning_content=response.reasoning_content,
            gen_metrics=gen_metrics,
        )
