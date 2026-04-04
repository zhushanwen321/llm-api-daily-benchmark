from unittest.mock import MagicMock, patch

from benchmark.core.llm_adapter import LLMEvalAdapter


def _mock_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """构造模拟的 API 响应。"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return mock_resp


@patch("benchmark.core.llm_adapter.get_model_config")
@patch("benchmark.core.llm_adapter.requests.post")
def test_generate_returns_generate_response(mock_post, mock_config):
    mock_config.return_value = {
        "api_key": "test-key",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
    }
    mock_post.return_value = _mock_response("hello world", 20, 10)

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
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "no usage"}}],
    }
    mock_post.return_value = mock_resp

    adapter = LLMEvalAdapter()
    result = adapter.generate("test prompt", "test/model")

    assert result.content == "no usage"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
