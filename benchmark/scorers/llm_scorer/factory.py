"""评分后端工厂函数。"""

from __future__ import annotations

import os

from benchmark.scorers.llm_scorer.base import LLMScorerBackend
from benchmark.scorers.llm_scorer.qwen_cli_backend import QwenCLIBackend
from benchmark.scorers.llm_scorer.api_backend import LLMAPIScorerBackend


def create_scorer_backend(
    backend_type: str | None = None,
) -> LLMScorerBackend:
    """根据配置创建评分后端实例。

    Args:
        backend_type: 后端类型，"qwen_cli" 或 "llm_api"。
                      为 None 时从环境变量 SCORING_BACKEND_TYPE 读取。

    Returns:
        评分后端实例

    Raises:
        ValueError: 未知的后端类型
    """
    if backend_type is None:
        backend_type = os.getenv("SCORING_BACKEND_TYPE", "qwen_cli").lower()

    if backend_type == "qwen_cli":
        qwen_path = os.getenv("QWEN_CLI_PATH", "qwen")
        timeout = int(os.getenv("QWEN_TIMEOUT", "300"))
        return QwenCLIBackend(qwen_path=qwen_path, timeout=timeout)
    elif backend_type == "llm_api":
        api_key = os.getenv("SCORING_API_KEY", "")
        api_base = os.getenv("SCORING_API_BASE", "https://api.openai.com/v1")
        model = os.getenv("SCORING_MODEL", "gpt-4")
        return LLMAPIScorerBackend(
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
    else:
        raise ValueError(
            f"Unknown scoring backend type: {backend_type}. "
            f"Supported: 'qwen_cli', 'llm_api'"
        )
