"""探针基类定义."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter


class BaseProbe(ABC):
    """探针基类."""

    @property
    @abstractmethod
    def frequency(self) -> str:
        """探针频率: fast | medium | slow."""
        pass

    @abstractmethod
    def load_probes(self) -> list[TaskDefinition]:
        """加载探针定义."""
        pass

    @abstractmethod
    async def execute_probe(
        self,
        probe: TaskDefinition,
        model: str,
        adapter: LLMEvalAdapter,
    ) -> EvalResult:
        """执行探针."""
        pass

    @abstractmethod
    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """提取探针特征."""
        pass
