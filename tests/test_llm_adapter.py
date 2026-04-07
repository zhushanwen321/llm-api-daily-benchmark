from unittest.mock import patch

from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.concurrency import AsyncConcurrencyLimiter


@patch("benchmark.core.llm_adapter.get_model_config")
def test_async_uses_concurrency_limiter(mock_config):
    """async 路径应使用 AsyncConcurrencyLimiter。"""
    mock_config.return_value = {
        "api_key": "k",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
        "max_concurrency": 3,
    }
    adapter = LLMEvalAdapter()
    limiter = adapter._get_or_create_async_limiter("test/model")
    assert limiter is not None
    assert isinstance(limiter, AsyncConcurrencyLimiter)


@patch("benchmark.core.llm_adapter.get_model_config")
def test_no_max_concurrency_no_limiter(mock_config):
    """max_concurrency 为 None 时不创建 limiter。"""
    mock_config.return_value = {
        "api_key": "k",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
        "max_concurrency": None,
    }
    adapter = LLMEvalAdapter()
    assert adapter._get_or_create_async_limiter("test/model") is None
