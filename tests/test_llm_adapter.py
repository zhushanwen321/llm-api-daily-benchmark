import threading
import json
from unittest.mock import MagicMock, patch

from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.concurrency import AsyncConcurrencyLimiter


def _mock_stream_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """构造模拟的流式 API 响应。"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    # 模拟 SSE 流式响应
    chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n",
        f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}})}\n\n",
        "data: [DONE]\n\n",
    ]

    def iter_lines(decode_unicode=True):
        for chunk in chunks:
            for line in chunk.split('\n'):
                if line:
                    yield line

    mock_resp.iter_lines = iter_lines
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda self, *args: None

    return mock_resp


@patch("benchmark.core.llm_adapter.get_model_config")
@patch("benchmark.core.llm_adapter.requests.post")
def test_generate_returns_generate_response(mock_post, mock_config):
    mock_config.return_value = {
        "api_key": "test-key",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
        "max_concurrency": None,
    }
    mock_post.return_value = _mock_stream_response("hello world", 20, 10)

    adapter = LLMEvalAdapter()
    result = adapter.generate("test prompt", "test/model")

    assert result.content == "hello world"
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 10


@patch("benchmark.core.llm_adapter.get_model_config")
@patch("benchmark.core.llm_adapter.requests.post")
def test_generate_handles_missing_usage(mock_post, mock_config):
    mock_config.return_value = {
        "api_key": "test-key",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
        "max_concurrency": None,
    }

    # 模拟没有 usage 字段的流式响应
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'no usage'}}]})}\n\n",
        f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n",
        "data: [DONE]\n\n",
    ]

    def iter_lines(decode_unicode=True):
        for chunk in chunks:
            for line in chunk.split('\n'):
                if line:
                    yield line

    mock_resp.iter_lines = iter_lines
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda self, *args: None

    mock_post.return_value = mock_resp

    adapter = LLMEvalAdapter()
    result = adapter.generate("test prompt", "test/model")

    assert result.content == "no usage"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


@patch("benchmark.core.llm_adapter.get_model_config")
def test_sync_uses_threading_semaphore(mock_config):
    """sync 路径应使用 threading.Semaphore。"""
    mock_config.return_value = {
        "api_key": "k",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
        "max_concurrency": 2,
    }
    adapter = LLMEvalAdapter()
    sem = adapter._get_or_create_sync_semaphore("test/model")
    assert sem is not None
    assert isinstance(sem, threading.Semaphore)


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
    assert adapter._get_or_create_sync_semaphore("test/model") is None
    assert adapter._get_or_create_async_limiter("test/model") is None
