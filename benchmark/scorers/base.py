"""评分器基类。

Stage 3 重构: score() 接收 ScoringContext 替代原来的 3 个参数。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from benchmark.models.schemas import ScoreResult, ScoringContext


class BaseScorer(ABC):
    """评分器抽象基类."""

    @abstractmethod
    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行评分.

        Args:
            ctx: 统一评分上下文，包含 model_answer/raw_output/expected/task.

        Returns:
            ScoreResult 包含分数、是否通过、详情、理由。
        """

    @abstractmethod
    def get_metric_name(self) -> str:
        """返回此评分器的指标名称（如 exact_match, execution）。"""

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """异步评分。默认用线程池包装同步 score()，子类可重写。"""
        return await asyncio.to_thread(self.score, ctx)
