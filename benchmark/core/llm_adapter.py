"""LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
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
# 两个有效 SSE chunk 之间的最大空闲间隔（秒），防止服务端挂起导致无限等待
_CHUNK_IDLE_TIMEOUT = 60


class LLMEvalAdapter:
    """LLM 调用适配器.

    从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
    支持重试（最多 max_retries 次，指数退避）。
    支持 provider 级令牌桶限流。
    """

    # provider -> limiter 实例缓存，同一 provider 的所有模型共享
    _provider_limiters: dict[str, TokenBucketRateLimiter] = {}
    # provider -> asyncio.Semaphore，用于 agenerate 并发控制
    _provider_semaphores: dict[str, asyncio.Semaphore] = {}

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

    def _get_or_create_semaphore(self, model: str) -> asyncio.Semaphore:
        """获取或创建 provider 级并发信号量，用于 agenerate。"""
        cfg = self._get_model_config(model)
        rate = cfg.get("rate_limit")
        if rate is None:
            rate = 100  # 无限制时给一个大值
        provider = cfg["provider"]
        if provider not in self._provider_semaphores:
            self._provider_semaphores[provider] = asyncio.Semaphore(int(rate))
        return self._provider_semaphores[provider]

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
        max_tokens: int | None = None,
    ) -> GenerateResponse:
        """调用 LLM 生成文本.

        Args:
            prompt: 输入提示.
            model: 模型标识（provider/model 格式）.
            temperature: 温度参数（评测时固定为 0）.
            max_tokens: 最大输出 token 数，None 时使用模型配置值。

        Returns:
            GenerateResponse 对象，包含生成文本和 token 用量.

        Raises:
            ValueError: 模型未配置.
            ConnectionError: 重试耗尽后仍失败.
        """
        cfg = self._get_model_config(model)
        api_key = cfg["api_key"]
        api_base = cfg["api_base"].rstrip("/")
        # max_tokens 优先级: 调用方显式指定 > models.yaml 配置 > 兜底默认值
        model_max_tokens = cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)
        effective_max_tokens = (
            min(max_tokens, model_max_tokens)
            if max_tokens is not None
            else model_max_tokens
        )

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
        # 部分 provider（如 Kimi Coding Plan）校验 User-Agent 白名单
        headers["User-Agent"] = os.getenv("LLM_USER_AGENT", "claude-code/1.0.0")
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"[{model}] attempt {attempt + 1}/{self.max_retries} 开始请求"
                )
                logger.debug(f"[{model}] URL: {url}")
                logger.debug(
                    f"[{model}] payload: max_tokens={payload['max_tokens']}, "
                    f"temperature={temperature}"
                )

                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(30, self.timeout),
                    stream=True,
                )
                logger.debug(
                    f"[{model}] 响应状态: {resp.status_code}, "
                    f"content-type: {resp.headers.get('content-type', 'N/A')}"
                )
                resp.raise_for_status()

                content_parts: list[str] = []
                usage: dict[str, Any] = {}
                t_start = time.monotonic()
                t_first_token: float | None = None
                t_last_chunk = t_start
                t_last_activity = t_start
                got_done = False
                truncated = False
                content_filtered = False
                final_finish_reason = ""
                chunk_count = 0

                with resp:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            logger.debug(f"[{model}] 非 SSE 数据行: {line[:100]}")
                            continue

                        data_str = line[6:]
                        chunk_count += 1

                        # 检测流空闲超时（至少收到 1 个 chunk 后才开始检测，
                        # 避免首 token 等待时间被误判）
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
                                f"content 片段: {len(content_parts)})"
                            )
                            got_done = True
                            break

                        # 解析 SSE chunk
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                f"[{model}] chunk#{chunk_count} JSON 解析失败: "
                                f"{exc}, data: {data_str[:200]}"
                            )
                            continue

                        # 每 50 个 chunk 打一次进度日志，避免刷屏
                        if chunk_count % 50 == 0:
                            logger.debug(
                                f"[{model}] 流进度: chunk#{chunk_count}, "
                                f"已收集 {len(content_parts)} 个 content 片段, "
                                f"已耗时 {(time.monotonic() - t_start):.1f}s"
                            )

                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            delta_content = delta.get("content")
                            if delta_content:
                                content_parts.append(delta_content)
                                t_last_chunk = time.monotonic()
                                if t_first_token is None:
                                    t_first_token = t_last_chunk
                                    logger.debug(
                                        f"[{model}] 首 token 延迟: "
                                        f"{(t_first_token - t_start):.3f}s, "
                                        f"首内容: {repr(delta_content[:100])}"
                                    )

                            # finish_reason 语义区分
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
                    f"content 长度={len(''.join(content_parts))}"
                )

                # 允许没有 [DONE] 标记的情况，只要成功解析了响应内容
                # 某些 provider（如 MiniMax）可能不发送 [DONE] 标记
                if not got_done:
                    logger.debug(
                        f"[{model}] 流未收到结束标记，但已有内容，视为正常结束"
                    )

                full_content = "".join(content_parts)

                # 内容被安全过滤 — 不重试，直接返回标记
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

                # 检查返回内容是否为空，分类处理
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
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration=duration,
                    tokens_per_second=tokens_per_second,
                    ttft=ttft,
                    truncated=truncated,
                    finish_reason=final_finish_reason,
                )

            except TimeoutError as exc:
                # 流空闲超时，可重试
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] attempt {attempt + 1}/{self.max_retries} "
                        f"流空闲超时: {exc}. {wait}s 后重试..."
                    )
                    time.sleep(wait)

            except (
                requests.exceptions.RequestException,
                RequestsConnectionError,
                ChunkedEncodingError,
            ) as exc:
                last_error = exc

                # 4xx 客户端错误（非 429）不重试，立即失败
                if (
                    isinstance(exc, requests.exceptions.HTTPError)
                    and exc.response is not None
                    and 400 <= exc.response.status_code < 500
                    and exc.response.status_code != 429
                ):
                    status = exc.response.status_code
                    logger.error(
                        f"[{model}] 客户端错误 {status}，不重试: "
                        f"{exc.response.text[:300]}"
                    )
                    raise ConnectionError(
                        f"客户端错误 ({status})，不重试: {exc}"
                    ) from exc

                if attempt < self.max_retries - 1:
                    is_rate_limited = (
                        isinstance(exc, requests.exceptions.HTTPError)
                        and exc.response is not None
                        and exc.response.status_code == 429
                    )
                    if is_rate_limited:
                        # 优先使用服务端返回的 Retry-After
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = min(int(retry_after), 120)
                            except ValueError:
                                wait = min(10 * (2**attempt), 120)
                            logger.debug(
                                f"[{model}] 429 Retry-After: {retry_after}s, "
                                f"实际等待: {wait}s"
                            )
                        else:
                            wait = min(10 * (2**attempt), 120)
                    else:
                        wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] attempt {attempt + 1}/{self.max_retries} 失败 "
                        f"({type(exc).__name__}): {exc}. {wait}s 后重试..."
                    )
                    time.sleep(wait)

            except ConnectionError as exc:
                # 空内容等场景主动抛出的 builtin ConnectionError，可重试
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] attempt {attempt + 1}/{self.max_retries} "
                        f"内容异常: {exc}. {wait}s 后重试..."
                    )
                    time.sleep(wait)

        raise ConnectionError(
            f"[{model}] 重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error

    async def agenerate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> GenerateResponse:
        """异步调用 LLM 生成文本，使用 httpx 异步流式请求.

        与 generate() 逻辑完全一致，区别在于：
        - 使用 httpx.AsyncClient 替代 requests
        - 使用 asyncio.Semaphore 替代 TokenBucketRateLimiter 控制并发
        - 使用 await asyncio.sleep 替代 time.sleep
        """
        sem = self._get_or_create_semaphore(model)
        async with sem:
            return await self._do_agenerate(prompt, model, temperature, max_tokens)

    async def _do_agenerate(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> GenerateResponse:
        """agenerate 的实际实现，包含重试循环和 SSE 解析。"""
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
        }
        # 部分 provider（如 Kimi Coding Plan）校验 User-Agent 白名单
        headers["User-Agent"] = os.getenv("LLM_USER_AGENT", "claude-code/1.0.0")
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"[{model}] async attempt {attempt + 1}/{self.max_retries} 开始请求"
                )
                logger.debug(f"[{model}] URL: {url}")
                logger.debug(
                    f"[{model}] payload: max_tokens={payload['max_tokens']}, "
                    f"temperature={temperature}"
                )

                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=30, read=self.timeout, write=30, pool=30
                    )
                ) as client:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        logger.debug(
                            f"[{model}] 响应状态: {resp.status_code}, "
                            f"content-type: {resp.headers.get('content-type', 'N/A')}"
                        )

                        content_parts: list[str] = []
                        usage: dict[str, Any] = {}
                        t_start = time.monotonic()
                        t_first_token: float | None = None
                        t_last_chunk = t_start
                        t_last_activity = t_start
                        got_done = False
                        truncated = False
                        content_filtered = False
                        final_finish_reason = ""
                        chunk_count = 0

                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if not line.startswith("data: "):
                                logger.debug(f"[{model}] 非 SSE 数据行: {line[:100]}")
                                continue

                            data_str = line[6:]
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
                                    f"content 片段: {len(content_parts)})"
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
                                    f"已收集 {len(content_parts)} 个 content 片段, "
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
                                delta_content = delta.get("content")
                                if delta_content:
                                    content_parts.append(delta_content)
                                    t_last_chunk = time.monotonic()
                                    if t_first_token is None:
                                        t_first_token = t_last_chunk
                                        logger.debug(
                                            f"[{model}] 首 token 延迟: "
                                            f"{(t_first_token - t_start):.3f}s, "
                                            f"首内容: {repr(delta_content[:100])}"
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
                    f"content 长度={len(''.join(content_parts))}"
                )

                if not got_done:
                    logger.debug(
                        f"[{model}] 流未收到结束标记，但已有内容，视为正常结束"
                    )

                full_content = "".join(content_parts)

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
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration=duration,
                    tokens_per_second=tokens_per_second,
                    ttft=ttft,
                    truncated=truncated,
                    finish_reason=final_finish_reason,
                )

            except TimeoutError as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] async attempt {attempt + 1}/{self.max_retries} "
                        f"流空闲超时: {exc}. {wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)

            except (httpx.HTTPStatusError, httpx.StreamError) as exc:
                last_error = exc

                # 4xx 客户端错误（非 429）不重试，立即失败
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and 400 <= exc.response.status_code < 500
                    and exc.response.status_code != 429
                ):
                    status = exc.response.status_code
                    error_body = ""
                    try:
                        error_body = exc.response.text[:300]
                    except Exception:
                        pass
                    logger.error(f"[{model}] 客户端错误 {status}，不重试: {error_body}")
                    raise ConnectionError(
                        f"客户端错误 ({status})，不重试: {exc}"
                    ) from exc

                if attempt < self.max_retries - 1:
                    is_rate_limited = (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code == 429
                    )
                    if is_rate_limited:
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = min(int(retry_after), 120)
                            except ValueError:
                                wait = min(10 * (2**attempt), 120)
                            logger.debug(
                                f"[{model}] 429 Retry-After: {retry_after}s, "
                                f"实际等待: {wait}s"
                            )
                        else:
                            wait = min(10 * (2**attempt), 120)
                    else:
                        wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] async attempt {attempt + 1}/{self.max_retries} 失败 "
                        f"({type(exc).__name__}): {exc}. {wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] async attempt {attempt + 1}/{self.max_retries} "
                        f"连接/超时错误 ({type(exc).__name__}): {exc}. {wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)

            except ConnectionError as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = min(2 * (2**attempt), 120)
                    logger.warning(
                        f"[{model}] async attempt {attempt + 1}/{self.max_retries} "
                        f"内容异常: {exc}. {wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)

        raise ConnectionError(
            f"[{model}] 重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error
