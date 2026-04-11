"""LLM API 评分后端实现。"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.llm_scorer.base import LLMScorerBackend
from benchmark.core.llm_adapter import LLMEvalAdapter

logger = logging.getLogger(__name__)


class LLMAPIScorerBackend(LLMScorerBackend):
    """使用 OpenAI 兼容 API 进行评分的后端实现。"""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        # 从环境变量读取配置，参数优先级高于环境变量
        self.api_key = api_key or os.getenv("SCORING_API_KEY", "")
        self.api_base = (
            api_base or os.getenv("SCORING_API_BASE", "https://api.openai.com/v1")
        ).rstrip("/")
        raw_model = model or os.getenv("SCORING_MODEL", "gpt-4")

        # 如果模型名不是 provider/model 格式，包装为 scoring/{model}
        if "/" in raw_model:
            self._model_name = raw_model
        else:
            self._model_name = f"scoring/{raw_model}"

        self.timeout = timeout
        self.max_retries = max_retries

        # 创建 LLMEvalAdapter 实例（不传递 model，避免自动调用 get_model_config）
        self._llm = LLMEvalAdapter(
            max_retries=max_retries,
            timeout=timeout,
        )

        # 注册模型配置
        config = {
            "provider": "scoring",
            "api_key": self.api_key,
            "api_base": self.api_base,
            "max_tokens": 4096,
            "max_concurrency": 2,
            "thinking": {},
        }
        self._llm.register_model_config(self._model_name, config)

    async def close(self) -> None:
        """关闭客户端连接。"""
        await self._llm.close()

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

        try:
            content = await self._call_api(prompt)
            return self._parse_result(content, dimensions)
        except Exception as exc:
            logger.error(f"评分 API 调用失败: {exc}")
            raise ConnectionError(f"评分 API 调用失败: {exc}") from exc

    async def health_check(self) -> bool:
        """检查 API 是否可用。"""
        try:
            # 使用 LLMEvalAdapter 发起一个简单的请求来验证 API 可用性
            response = await self._llm.agenerate(
                prompt="Hi",
                model=self._model_name,
                temperature=0.0,
                max_tokens=10,
            )
            return bool(response.content)
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
        response = await self._llm.agenerate(
            prompt=prompt,
            model=self._model_name,
            temperature=0.3,
        )
        return response.content

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
