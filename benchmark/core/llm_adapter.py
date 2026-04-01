"""LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
)

from benchmark.config import get_model_config
from benchmark.models.schemas import GenerateResponse
from benchmark.core.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

# 当模型配置未指定 max_tokens 时的默认值
_DEFAULT_MAX_TOKENS = 4096


class LLMEvalAdapter:
    """LLM 调用适配器.

    从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
    支持重试（最多 max_retries 次，指数退避）。
    支持 provider 级令牌桶限流。
    """

    # provider -> limiter 实例缓存，同一 provider 的所有模型共享
    _provider_limiters: dict[str, TokenBucketRateLimiter] = {}

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 5,
        timeout: int = 300,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._limiter: TokenBucketRateLimiter | None = None
        if model:
            self._model_cache[model] = get_model_config(model)
            self._limiter = self._get_or_create_limiter(model)

    def _get_or_create_limiter(self, model: str) -> TokenBucketRateLimiter | None:
        """获取或创建 provider 级限流器。"""
        cfg = self._get_model_config(model)
        rate = cfg.get("rate_limit")
        if rate is None:
            return None
        provider = cfg["provider"]
        if provider not in self._provider_limiters:
            self._provider_limiters[provider] = TokenBucketRateLimiter(rate=rate)
        return self._provider_limiters[provider]

    def _get_model_config(self, model: str) -> dict[str, Any]:
        """获取模型配置，带缓存."""
        if model not in self._model_cache:
            self._model_cache[model] = get_model_config(model)
        return self._model_cache[model]

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> GenerateResponse:
        """调用 LLM 生成文本.

        Args:
            prompt: 输入提示.
            model: 模型标识（provider/model 格式）.
            temperature: 温度参数（评测时固定为 0）.
            max_tokens: 最大输出 token 数.

        Returns:
            GenerateResponse 对象，包含生成文本和 token 用量.

        Raises:
            ValueError: 模型未配置.
            ConnectionError: 重试耗尽后仍失败.
        """
        cfg = self._get_model_config(model)
        api_key = cfg["api_key"]
        api_base = cfg["api_base"].rstrip("/")
        model_max_tokens = cfg.get("max_tokens", max_tokens)

        # 限流
        if self._limiter is not None:
            self._limiter.acquire()

        # 兼容 api_base 已包含路径结尾的情况
        if api_base.endswith("/chat/completions") or api_base.endswith("/messages"):
            url = api_base
        else:
            url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": min(max_tokens, model_max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(30, self.timeout),
                    stream=True,
                )
                resp.raise_for_status()

                content_parts: list[str] = []
                usage: dict[str, Any] = {}
                t_start = time.monotonic()
                t_first_token: float | None = None
                t_last_chunk = t_start
                got_done = False

                with resp:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]

                        if data_str == "[DONE]":
                            got_done = True
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                f"Failed to parse SSE chunk: {exc}, data: {data_str}"
                            )
                            continue

                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            delta_content = delta.get("content")
                            if delta_content:
                                content_parts.append(delta_content)
                                t_last_chunk = time.monotonic()
                                if t_first_token is None:
                                    t_first_token = t_last_chunk

                        chunk_usage = chunk.get("usage")
                        if chunk_usage:
                            usage = chunk_usage

                if not got_done:
                    raise ConnectionError("Stream ended without [DONE] marker")

                full_content = "".join(content_parts)
                duration = t_last_chunk - t_start
                ttft = t_first_token - t_start if t_first_token else 0.0

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                tokens_per_second = 0.0
                if duration > 0 and completion_tokens > 0:
                    tokens_per_second = completion_tokens / duration

                return GenerateResponse(
                    content=full_content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration=duration,
                    tokens_per_second=tokens_per_second,
                    ttft=ttft,
                )

            except (
                requests.exceptions.RequestException,
                RequestsConnectionError,
                ChunkedEncodingError,
            ) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    # 429 限频用更长退避，其他错误用标准退避
                    is_rate_limited = (
                        isinstance(exc, requests.exceptions.HTTPError)
                        and exc.response is not None
                        and exc.response.status_code == 429
                    )
                    base = 10 if is_rate_limited else 2
                    wait = min(base * (2**attempt), 120)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed for model '{model}': "
                        f"{exc}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)

        raise ConnectionError(
            f"Failed after {self.max_retries} retries for model '{model}': {last_error}"
        ) from last_error
