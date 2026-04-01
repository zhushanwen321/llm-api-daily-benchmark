from datetime import datetime

from benchmark.models.schemas import ApiCallMetrics, GenerateResponse


def test_generate_response_defaults():
    resp = GenerateResponse(content="hello")
    assert resp.content == "hello"
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0


def test_generate_response_with_tokens():
    resp = GenerateResponse(content="hello", prompt_tokens=10, completion_tokens=5)
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5


def test_api_call_metrics():
    now = datetime.now()
    metrics = ApiCallMetrics(
        result_id="abc123",
        prompt_tokens=100,
        completion_tokens=50,
        duration=2.5,
        tokens_per_second=20.0,
        created_at=now,
    )
    assert metrics.result_id == "abc123"
    assert metrics.tokens_per_second == 20.0
