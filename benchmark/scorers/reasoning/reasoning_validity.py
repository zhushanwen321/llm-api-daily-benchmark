from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = "你是数学推理质量评判专家。请检查以下解题过程的正确性。"

_JUDGE_TEMPLATE = """请评估以下数学解题过程的正确性。

题目: {problem}

解题过程:
{reasoning}

请以 JSON 格式返回评分（不要包含其他文字）:
{{"logical_consistency": <0-40>, "math_facts": <0-40>, "computation": <0-20>}}

评分标准:
- logical_consistency (0-40): 逻辑一致性，推理链条是否连贯
- math_facts (0-40): 数学事实正确性，公式和定理使用是否正确
- computation (0-20): 计算正确性，数值计算是否准确"""


class ReasoningValidityScorer(BaseScorer):
    """LLM-as-a-Judge 评估推理过程有效性。只支持异步评分 (ascore)。"""

    def __init__(self, llm: Any, model: str = "zai/glm-5.1") -> None:
        self._llm = llm
        self._model = model
        self._cache: dict[str, float] = {}

    def score(self, ctx: ScoringContext) -> ScoreResult:
        raise NotImplementedError("ReasoningValidityScorer 只支持异步评分 (ascore)")

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(
                score=100.0, passed=True,
                details={"reason": "empty_reasoning"},
                reasoning="No reasoning content, default 100",
            )

        cache_key = self._cache_key(ctx.task.prompt, reasoning)
        if cache_key in self._cache:
            score = self._cache[cache_key]
            return ScoreResult(
                score=score, passed=score >= 60.0,
                details={"source": "cache"},
                reasoning=f"Cached score: {score}",
            )

        try:
            prompt = _JUDGE_TEMPLATE.format(problem=ctx.task.prompt, reasoning=reasoning)
            resp = await self._llm.agenerate(
                prompt=prompt, model=self._model, temperature=0.0,
                system_message=_JUDGE_SYSTEM,
            )
            score = self._parse_response(resp.content)
            self._cache[cache_key] = score
            return ScoreResult(
                score=score, passed=score >= 60.0,
                details={"source": "llm_judge", "raw": resp.content[:200]},
                reasoning=f"LLM judge score: {score}",
            )
        except Exception as exc:
            logger.warning("ReasoningValidityScorer LLM 调用失败: %s", exc)
            return ScoreResult(
                score=50.0, passed=False,
                details={"error": str(exc)},
                reasoning=f"LLM judge failed: {exc}, default 50",
            )

    @staticmethod
    def _cache_key(prompt: str, reasoning: str) -> str:
        h = hashlib.md5()
        h.update(prompt.encode())
        h.update(reasoning.encode())
        return h.hexdigest()

    @staticmethod
    def _parse_response(content: str) -> float:
        json_match = re.search(r'\{[^}]+\}', content)
        if not json_match:
            return 50.0
        try:
            data = json.loads(json_match.group())
            lc = min(40, max(0, float(data.get("logical_consistency", 0))))
            mf = min(40, max(0, float(data.get("math_facts", 0))))
            cp = min(20, max(0, float(data.get("computation", 0))))
            return round(lc + mf + cp, 1)
        except (json.JSONDecodeError, ValueError, TypeError):
            return 50.0

    def get_metric_name(self) -> str:
        return "reasoning_validity"
