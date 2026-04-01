"""LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

from __future__ import annotations

import time
from typing import Any

import requests

from benchmark.config import get_model_config

# 当模型配置未指定 max_tokens 时的默认值
_DEFAULT_MAX_TOKENS = 4096


class LLMEvalAdapter:
    """LLM 调用适配器.

    从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
    支持重试（最多 max_retries 次，指数退避）。
    如果初始化时传入 model，配置只加载一次；否则在 generate() 时按需加载。
    """

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 5,
        timeout: int = 300,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self._model_cache: dict[str, dict[str, Any]] = {}
        # 初始化时预加载模型配置（如果指定了 model）
        if model:
            self._model_cache[model] = get_model_config(model)

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
            model: 模型名称（需在 models.yaml 中配置）.
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

        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
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
