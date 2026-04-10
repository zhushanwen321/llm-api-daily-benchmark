"""评分后端抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.models.schemas import ScoreResult, ScoringContext


class LLMScorerBackend(ABC):
    """LLM 评分后端抽象基类。

    子类需要实现 score() 和 health_check() 方法。
    score() 接收评分上下文和维度列表，返回每个维度的评分结果。
    """

    @abstractmethod
    async def score(
        self,
        context: ScoringContext,
        dimensions: list[str],
    ) -> dict[str, ScoreResult]:
        """对给定上下文进行多维度评分。

        Args:
            context: 评分上下文（题目、答案、推理过程等）
            dimensions: 需要计算的评分维度列表

        Returns:
            维度名 -> ScoreResult 的映射
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """检查后端是否可用。"""
