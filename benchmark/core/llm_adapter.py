"""LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from benchmark.config import get_model_config
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
    ) -> str:
        """调用 LLM 生成文本.

        Args:
            prompt: 输入提示.
            model: 模型标识（provider/model 格式）.
            temperature: 温度参数（评测时固定为 0）.
            max_tokens: 最大输出 token 数.

        Returns:
            模型生成的文本.

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
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    # 429 限频用更长退避，其他错误用标准退避
                    is_rate_limited = (
                        isinstance(exc, requests.exceptions.HTTPError)
                        and exc.response is not None
                        and exc.response.status_code == 429
                    )
                    base = 10 if is_rate_limited else 2
                    wait = min(base * 2**attempt, 120)
                    time.sleep(wait)

        raise ConnectionError(
            f"Failed after {self.max_retries} retries for model '{model}': {last_error}"
        ) from last_error
