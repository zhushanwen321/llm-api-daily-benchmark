"""Qwen CLI 评分后端实现。"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.llm_scorer.base import LLMScorerBackend

logger = logging.getLogger(__name__)


class QwenCLIBackend(LLMScorerBackend):
    """使用 Qwen CLI 进行评分的后端实现。"""

    def __init__(
        self,
        qwen_path: str = "qwen",
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.qwen_path = qwen_path
        self.timeout = timeout
        self.max_retries = max_retries

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
                raw_output = await self._call_qwen(prompt)
                return self._parse_result(raw_output, dimensions)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = self._calc_backoff(attempt)
                    logger.warning(
                        f"Qwen CLI 尝试 {attempt + 1}/{self.max_retries} 失败: {exc}. "
                        f"{wait}s 后重试..."
                    )
                    await asyncio.sleep(wait)
                else:
                    break

        raise ConnectionError(
            f"Qwen CLI 重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error

    async def health_check(self) -> bool:
        """检查 qwen CLI 是否可用。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.qwen_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            return proc.returncode == 0
        except Exception as exc:
            logger.debug(f"Qwen CLI health check failed: {exc}")
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
                "你必须且只能返回一个 JSON 对象，不要有任何其他文字。",
                "JSON 格式如下：",
                "{",
            ]
        )

        for dim in dimensions:
            prompt_parts.append(f'  "{dim}": {{')
            prompt_parts.append('    "score": 100,')
            prompt_parts.append('    "passed": true,')
            prompt_parts.append('    "reasoning": "评分理由"')
            prompt_parts.append("  },")

        prompt_parts.extend(
            [
                "}",
                "",
                "规则：score 必须是 0 到 100 的数字，passed 必须是 true 或 false。",
            ]
        )

        return "\n".join(prompt_parts)

    async def _call_qwen(self, prompt: str) -> str:
        """调用 qwen CLI 并返回原始输出。"""
        cmd = [
            self.qwen_path,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--yolo",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"Qwen CLI 执行超时（{self.timeout}s）")

        if proc.returncode != 0:
            stderr_str = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Qwen CLI 执行失败: {stderr_str}")

        stdout_str = stdout.decode("utf-8", errors="replace")

        # 解析 qwen CLI 的 JSON 输出
        # qwen cli --output-format json 返回 list，最后一个元素是 type="result"，实际回复在 .result 字段
        try:
            raw_data = json.loads(stdout_str)
            if isinstance(raw_data, list):
                # 从最后一个 type="result" 的元素中提取回复
                for item in reversed(raw_data):
                    if isinstance(item, dict) and item.get("type") == "result":
                        content = item.get("result", "")
                        if content:
                            return str(content)
                # fallback: 从 assistant 类型的消息中提取
                for item in raw_data:
                    if isinstance(item, dict) and item.get("type") == "assistant":
                        msg = item.get("message", {})
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                # content 可能是 [{type: "text", text: "..."}]
                                for part in content:
                                    if isinstance(part, dict) and part.get("text"):
                                        return part["text"]
                            if content:
                                return str(content)
                raise ValueError("Qwen CLI JSON 中未找到回复内容")
            elif isinstance(raw_data, dict):
                content = raw_data.get("response", raw_data.get("result", ""))
                if isinstance(content, str):
                    return content
                return json.dumps(content)
            return stdout_str
        except json.JSONDecodeError:
            return stdout_str

    def _parse_result(
        self,
        raw: str,
        dimensions: list[str],
    ) -> dict[str, ScoreResult]:
        """解析 qwen 返回的结果为 ScoreResult 字典。"""
        # 尝试从 raw 中提取 JSON（处理 ```json ... ``` 包裹）
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

    def _calc_backoff(self, attempt: int) -> float:
        """计算指数退避等待时间。"""
        return min(2 * (2**attempt), 60.0)
