"""LLM API 评分后端实现。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.llm_scorer.base import LLMScorerBackend

logger = logging.getLogger(__name__)


class LLMAPIScorerBackend(LLMScorerBackend):
    """使用 OpenAI 兼容 API 进行评分的后端实现。"""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4",
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 httpx 客户端。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30, read=self.timeout, write=30, pool=30),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """关闭客户端连接。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def score(
        self,
        context: ScoringContext,
        dimensions: list[str],
    ) -> dict[str, ScoreResult]:
        """对给定上下文进行多维度评分。"""
        prompt = self._build_scoring_prompt(context, dimensions)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                content = await self._call_api(prompt)
                return self._parse_result(content, dimensions)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = self._calc_backoff(exc, attempt)
                    logger.warning(
                        f"API 调用尝试 {attempt + 1}/{self.max_retries} 失败: {exc}. "
                        f"{wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)
                else:
                    break

        raise ConnectionError(
            f"API 调用重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error

    async def health_check(self) -> bool:
        """检查 API 是否可用。"""
        try:
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            client = self._get_client()
            response = await client.get(
                f"{self.api_base}/models",
                headers=headers,
                timeout=10,
            )
            return response.status_code == 200
        except Exception as exc:
            logger.debug(f"API health check failed: {exc}")
            return False

    def _build_scoring_prompt(
        self,
        context: ScoringContext,
        dimensions: list[str],
    ) -> str:
        """构建评分 prompt。"""
        task = context.task

        prompt_parts = [
            "请作为评分专家，对以下模型回答进行多维度评分。",
            "",
            "=== 题目 ===",
            task.prompt,
            "",
            "=== 期望答案 ===",
            context.expected,
            "",
            "=== 模型回答 ===",
            context.model_answer,
            "",
        ]

        # 添加模型完整输出（如果有）
        if context.raw_output and context.raw_output != context.model_answer:
            prompt_parts.extend(
                [
                    "=== 模型完整输出 ===",
                    context.raw_output,
                    "",
                ]
            )

        # 添加推理过程（如果有）
        if context.reasoning_content:
            prompt_parts.extend(
                [
                    "=== 推理过程 ===",
                    context.reasoning_content,
                    "",
                ]
            )

        # 添加评分维度
        prompt_parts.extend(
            [
                "=== 评分维度 ===",
                "请对以下维度进行评分：",
            ]
        )

        dimension_desc = {
            "correctness": "答案的正确性（是否与期望答案一致）",
            "completeness": "答案的完整性（是否覆盖所有要点）",
            "clarity": "表达的清晰度（是否易于理解）",
            "reasoning": "推理过程的合理性（逻辑是否清晰）",
            "code_quality": "代码质量（如适用）",
        }

        for dim in dimensions:
            desc = dimension_desc.get(dim, "")
            prompt_parts.append(f"- {dim}: {desc}")

        prompt_parts.extend(
            [
                "",
                "=== 评分要求 ===",
                "请严格按照以下 JSON 格式返回评分结果：",
                "",
                "{",
            ]
        )

        for dim in dimensions:
            prompt_parts.append(f'  "{dim}": {{')
            prompt_parts.append('    "score": 0-100,  // 0-100 的整数分数')
            prompt_parts.append(
                '    "passed": true/false,  // 是否通过阈值（建议 >=60 为通过）'
            )
            prompt_parts.append('    "reasoning": "...详细评分理由..."')
            prompt_parts.append("  },")

        prompt_parts.extend(
            [
                "}",
                "",
                "注意：",
                "1. 必须返回有效的 JSON 格式",
                "2. score 必须是 0-100 之间的整数",
                "3. 每个维度都需要提供详细的评分理由",
                "4. 不要返回除 JSON 外的任何其他内容",
            ]
        )

        return "\n".join(prompt_parts)

    async def _call_api(self, prompt: str) -> str:
        """调用 LLM API 并返回响应内容。"""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }

        # 尝试添加 response_format（如果支持）
        try:
            payload["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        client = self._get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("API 响应中没有 choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content:
            raise ValueError("API 响应内容为空")

        return content

    def _parse_result(
        self,
        raw: str,
        dimensions: list[str],
    ) -> dict[str, ScoreResult]:
        """解析 API 返回的结果为 ScoreResult 字典。"""
        # 尝试从 raw 中提取 JSON
        json_str = self._extract_json_from_text(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"无法解析 JSON 响应: {exc}\n原始内容: {raw[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(f"JSON 响应不是对象: {type(data)}")

        result: dict[str, ScoreResult] = {}
        for dim in dimensions:
            if dim not in data:
                logger.warning(f"维度 {dim} 在响应中缺失")
                result[dim] = ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning=f"维度 {dim} 在模型响应中缺失",
                )
                continue

            dim_data = data[dim]
            if not isinstance(dim_data, dict):
                logger.warning(f"维度 {dim} 的数据不是对象: {type(dim_data)}")
                result[dim] = ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning=f"维度 {dim} 的数据格式错误",
                )
                continue

            score = float(dim_data.get("score", 0))
            passed = bool(dim_data.get("passed", score >= 60))
            reasoning = str(dim_data.get("reasoning", ""))

            result[dim] = ScoreResult(
                score=score,
                passed=passed,
                reasoning=reasoning,
                details={"raw": dim_data},
            )

        return result

    def _extract_json_from_text(self, text: str) -> str:
        """从文本中提取 JSON 内容（处理 markdown code block）。"""
        # 尝试匹配 ```json ... ```
        json_block_match = re.search(
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if json_block_match:
            return json_block_match.group(1).strip()

        # 尝试匹配单独的 {...}
        json_match = re.search(r"(\{[\s\S]*\})", text)
        if json_match:
            return json_match.group(1).strip()

        return text.strip()

    def _calc_backoff(self, exc: Exception, attempt: int) -> float:
        """计算指数退避等待时间。"""
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 429:
                # 检查 Retry-After 头
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        return min(float(retry_after), 120.0)
                    except ValueError:
                        pass
                return min(10 * (2**attempt), 120.0)
        return min(2 * (2**attempt), 60.0)
