"""异步评分后端抽象层。

提供统一的评分后端接口，支持多种后端实现（qwen CLI、LLM API等）。
通过工厂函数 create_scorer_backend() 根据配置创建实例。
"""

from benchmark.scorers.llm_scorer.base import LLMScorerBackend
from benchmark.scorers.llm_scorer.qwen_cli_backend import QwenCLIBackend
from benchmark.scorers.llm_scorer.api_backend import LLMAPIScorerBackend
from benchmark.scorers.llm_scorer.factory import create_scorer_backend

__all__ = [
    "LLMScorerBackend",
    "QwenCLIBackend",
    "LLMAPIScorerBackend",
    "create_scorer_backend",
]
