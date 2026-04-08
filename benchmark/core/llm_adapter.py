"""LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

from benchmark.config import get_model_config
from benchmark.models.schemas import GenerateResponse
from benchmark.core.concurrency import AsyncConcurrencyLimiter

logger = logging.getLogger(__name__)


class _NonRetryableError(Exception):
    """4xx 客户端错误（非 429），不重试。"""


# 当模型配置未指定 max_tokens 时的默认值
_DEFAULT_MAX_TOKENS = 4096
# 两个有效 SSE chunk 之间的最大空闲间隔（秒），防止服务端挂起导致无限等待
_CHUNK_IDLE_TIMEOUT = 60


class LLMEvalAdapter:
    """LLM 调用适配器.

    从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
    支持重试（最多 max_retries 次，指数退避）。
    支持 provider 级令牌桶限流。
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
        if model:
            self._model_cache[model] = get_model_config(model)

    def _get_or_create_async_limiter(self, model: str) -> AsyncConcurrencyLimiter | None:
        """获取或创建 provider 级异步并发控制器，用于 agenerate。"""
        cfg = self._get_model_config(model)
        max_conc = cfg.get("max_concurrency")
        if max_conc is None:
            return None
        provider = cfg["provider"]
        return AsyncConcurrencyLimiter.get_or_create(provider, max_conc)

    def _get_model_config(self, model: str) -> dict[str, Any]:
        """获取模型配置，带缓存."""
        if model not in self._model_cache:
            self._model_cache[model] = get_model_config(model)
        return self._model_cache[model]

    async def agenerate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system_message: str | None = None,
        disable_thinking: bool = False,
    ) -> GenerateResponse:
        """异步调用 LLM 生成文本。每次 attempt 独立 acquire/release semaphore。"""
        cfg = self._get_model_config(model)
        api_key = cfg["api_key"]
        api_base = cfg["api_base"].rstrip("/")
        model_max_tokens = cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)
        effective_max_tokens = (
            min(max_tokens, model_max_tokens)
            if max_tokens is not None
            else model_max_tokens
        )

        if api_base.endswith("/chat/completions") or api_base.endswith("/messages"):
            url = api_base
        else:
            url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": os.getenv("LLM_USER_AGENT", "claude-code/1.0.0"),
        }
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        thinking_cfg = cfg.get("thinking", {})
        if not disable_thinking and thinking_cfg.get("enabled") and thinking_cfg.get("request_params"):
            payload.update(thinking_cfg["request_params"])

        limiter = self._get_or_create_async_limiter(model)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if limiter is not None:
                    await limiter.acquire()
                try:
                    return await self._single_api_request(
                        url, headers, payload, model, thinking_cfg
                    )
                finally:
                    if limiter is not None:
                        limiter.release()
            except _NonRetryableError as exc:
                raise ConnectionError(str(exc)) from exc
            except (TimeoutError, httpx.HTTPStatusError, httpx.StreamError,
                    httpx.ConnectError, httpx.TimeoutException, ConnectionError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = self._calc_backoff(exc, attempt)
                    # 429 时设置全局退避，阻止其他 task 立刻涌入
                    if (limiter is not None
                            and isinstance(exc, httpx.HTTPStatusError)
                            and exc.response.status_code == 429):
                        loop = asyncio.get_running_loop()
                        limiter.set_rate_limited(loop.time() + wait)
                    logger.warning(
                        f"[{model}] async attempt {attempt + 1}/{self.max_retries} "
                        f"失败 ({type(exc).__name__}): {exc}. {wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)  # semaphore 已释放，不空占

        raise ConnectionError(
            f"[{model}] 重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error

    async def _single_api_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model: str,
        thinking_cfg: dict[str, Any],
    ) -> GenerateResponse:
        """执行单次 API 请求（无重试），包含 SSE 流式解析。"""
        logger.debug(f"[{model}] 开始请求 URL: {url}")
        logger.debug(
            f"[{model}] payload: max_tokens={payload['max_tokens']}, "
            f"temperature={payload['temperature']}"
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30, read=self.timeout, write=30, pool=30
            )
        ) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if (
                        400 <= exc.response.status_code < 500
                        and exc.response.status_code != 429
                    ):
                        error_body = ""
                        try:
                            error_body = exc.response.text[:300]
                        except Exception:
                            pass
                        logger.error(
                            f"[{model}] 客户端错误 {exc.response.status_code}，不重试: {error_body}"
                        )
                        raise _NonRetryableError(
                            f"客户端错误 ({exc.response.status_code})，不重试: {exc}"
                        ) from exc
                    raise
                logger.debug(
                    f"[{model}] 响应状态: {resp.status_code}, "
                    f"content-type: {resp.headers.get('content-type', 'N/A')}"
                )

                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                usage: dict[str, Any] = {}
                t_start = time.monotonic()
                t_first_token: float | None = None
                t_first_content_token: float | None = None
                t_last_chunk = t_start
                t_last_activity = t_start
                got_done = False
                truncated = False
                content_filtered = False
                final_finish_reason = ""
                chunk_count = 0
                # reasoning 字段自动检测：优先用配置值，否则从首个 delta 自动发现
                configured_reasoning_field = thinking_cfg.get("reasoning_field")
                _REASONING_FIELD_CANDIDATES = {"reasoning_content", "reasoning", "reasoning_details"}
                _SKIP_DELTA_KEYS = {"content", "role", "refusal", "function_call", "tool_calls"}
                detected_reasoning_field: str | None = configured_reasoning_field
                reasoning_field_auto_detected = configured_reasoning_field is None

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    # 兼容 "data:" 和 "data: " 两种 SSE 格式
                    data_str = line[5:]
                    if data_str.startswith(" "):
                        data_str = data_str[1:]
                    chunk_count += 1

                    now = time.monotonic()
                    if (
                        chunk_count > 1
                        and now - t_last_activity > _CHUNK_IDLE_TIMEOUT
                    ):
                        raise TimeoutError(
                            f"流空闲超时: 已 {_CHUNK_IDLE_TIMEOUT}s 未收到数据 "
                            f"(chunk#{chunk_count}, "
                            f"已收集 {len(content_parts)} 个 content 片段)"
                        )
                    t_last_activity = now

                    if data_str == "[DONE]":
                        logger.debug(
                            f"[{model}] 收到 [DONE] 标记，流结束 "
                            f"(共 {chunk_count} chunks, "
                            f"reasoning 片段: {len(reasoning_parts)}, content 片段: {len(content_parts)})"
                        )
                        got_done = True
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            f"[{model}] chunk#{chunk_count} JSON 解析失败: "
                            f"{exc}, data: {data_str[:200]}"
                        )
                        continue

                    if chunk_count % 50 == 0:
                        logger.debug(
                            f"[{model}] 流进度: chunk#{chunk_count}, "
                            f"reasoning 片段: {len(reasoning_parts)}, "
                            f"content 片段: {len(content_parts)}, "
                            f"已耗时 {(time.monotonic() - t_start):.1f}s"
                        )

                    if chunk.get("type") == "error":
                        err = chunk.get("error", {})
                        err_msg = f"SSE 流错误: {err.get('message', 'unknown')}"
                        http_code = err.get("http_code")
                        if http_code:
                            err_msg += f" (http_code={http_code})"
                        logger.error(
                            f"[{model}] {err_msg}, data: {data_str[:300]}"
                        )
                        raise ConnectionError(err_msg)

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        # 自动检测 reasoning 字段（首个含非标准 key 的 delta）
                        if reasoning_field_auto_detected and chunk_count <= 2:
                            for k, v in delta.items():
                                if k not in _SKIP_DELTA_KEYS and v and k in _REASONING_FIELD_CANDIDATES:
                                    detected_reasoning_field = k
                                    reasoning_field_auto_detected = False
                                    logger.debug(
                                        f"[{model}] 自动检测到 reasoning 字段: '{k}'"
                                    )
                                    break

                        delta_reasoning = (
                            delta.get(detected_reasoning_field) if detected_reasoning_field else None
                        )
                        delta_content = delta.get("content")
                        if delta_reasoning:
                            # MiniMax reasoning_details 返回 list[dict]，提取 text 字段
                            if isinstance(delta_reasoning, list):
                                parts = []
                                for r in delta_reasoning:
                                    if isinstance(r, dict):
                                        parts.append(r.get("text", str(r)))
                                    else:
                                        parts.append(str(r))
                                delta_reasoning = "".join(parts)
                            elif not isinstance(delta_reasoning, str):
                                delta_reasoning = str(delta_reasoning)
                            reasoning_parts.append(delta_reasoning)
                            t_last_chunk = time.monotonic()
                            t_last_activity = t_last_chunk
                            if t_first_token is None:
                                t_first_token = t_last_chunk
                                logger.debug(
                                    f"[{model}] 首 reasoning token 延迟: "
                                    f"{(t_first_token - t_start):.3f}s"
                                )
                        if delta_content:
                            content_parts.append(delta_content)
                            t_last_chunk = time.monotonic()
                            t_last_activity = t_last_chunk
                            if t_first_content_token is None:
                                t_first_content_token = t_last_chunk
                                logger.debug(
                                    f"[{model}] 首 content token 延迟: "
                                    f"{(t_first_content_token - t_start):.3f}s"
                                )

                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason:
                            final_finish_reason = finish_reason
                            logger.debug(
                                f"[{model}] finish_reason='{finish_reason}', "
                                f"已收集 {len(content_parts)} 个 content 片段"
                            )
                            if finish_reason == "length":
                                truncated = True
                                logger.warning(
                                    f"[{model}] 输出被 max_tokens 截断 "
                                    f"(finish_reason=length), "
                                    f"当前 max_tokens={payload['max_tokens']}"
                                )
                            elif finish_reason == "content_filter":
                                content_filtered = True
                                logger.warning(
                                    f"[{model}] 内容被安全过滤拦截 "
                                    f"(finish_reason=content_filter)"
                                )
                            got_done = True

                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage = chunk_usage
                        logger.debug(
                            f"[{model}] chunk#{chunk_count} usage: "
                            f"prompt={chunk_usage.get('prompt_tokens')}, "
                            f"completion={chunk_usage.get('completion_tokens')}"
                        )

        logger.debug(
            f"[{model}] 流处理完成: chunks={chunk_count}, "
            f"got_done={got_done}, truncated={truncated}, "
            f"content_filtered={content_filtered}, "
            f"finish_reason='{final_finish_reason}', "
            f"reasoning 长度={len(''.join(reasoning_parts))}, content 长度={len(''.join(content_parts))}"
        )

        if not got_done:
            logger.debug(
                f"[{model}] 流未收到结束标记，但已有内容，视为正常结束"
            )

        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)

        if content_filtered:
            logger.warning(f"[{model}] 内容被安全过滤，跳过重试，返回标记")
            return GenerateResponse(
                content="[CONTENT_FILTERED]",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                duration=time.monotonic() - t_start,
                tokens_per_second=0.0,
                ttft=t_first_token - t_start if t_first_token else 0.0,
                truncated=False,
                finish_reason="content_filter",
            )

        if not full_content.strip():
            comp_tokens = usage.get("completion_tokens", 0)
            if chunk_count == 0:
                raise ConnectionError(
                    f"服务端未返回任何 SSE 数据 (model={model})"
                )
            if comp_tokens > 0:
                raise ConnectionError(
                    f"token 计数非零但内容为空 "
                    f"(chunks={chunk_count}, completion_tokens={comp_tokens}), "
                    f"可能被安全过滤"
                )
            raise ConnectionError(
                f"模型返回空内容 "
                f"(chunks={chunk_count}, completion_tokens={comp_tokens})"
            )

        duration = t_last_chunk - t_start
        ttft = t_first_token - t_start if t_first_token else 0.0

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        tokens_per_second = 0.0
        if duration > 0 and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration

        logger.debug(
            f"[{model}] 请求成功: duration={duration:.2f}s, "
            f"ttft={ttft:.3f}s, "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"speed={tokens_per_second:.1f} tokens/s, "
            f"truncated={truncated}"
        )

        return GenerateResponse(
            content=full_content,
            reasoning_content=full_reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
            duration=duration,
            tokens_per_second=tokens_per_second,
            ttft=ttft,
            ttft_content=t_first_content_token - t_start if t_first_content_token else 0.0,
            truncated=truncated,
            finish_reason=final_finish_reason,
        )

    def _calc_backoff(self, exc: Exception, attempt: int) -> int:
        """根据异常类型计算退避时间。"""
        is_rate_limited = (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code == 429
        )
        if is_rate_limited:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(int(retry_after), 120)
                except ValueError:
                    pass
            return min(10 * (2 ** attempt), 120)
        return min(2 * (2 ** attempt), 120)
