"""Probe 专用评分器。基于 expected_answer 精确匹配 + 格式/指令遵从检测。"""

from __future__ import annotations

import json
import re
import unicodedata

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ProbeScorer(BaseScorer):
    """Probe 监控评分器。

    评分策略按题目类型分派：
    - known_answer: 精确答案匹配（100 或 0）
    - reasoning: 答案包含正确数值即通过
    - format: 检查 JSON 格式 + 答案字段存在
    - consistency/instruction: 答案存在且非空即通过（质量由信号检测）
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "known_answer")
        expected = ctx.task.expected_output or ""
        answer = ctx.model_answer or ""

        if task_type == "known_answer":
            return self._score_known_answer(expected, answer)
        if task_type == "reasoning":
            return self._score_reasoning(expected, answer)
        if task_type == "format":
            return self._score_format(expected, answer)
        # consistency / instruction: 基础检查
        return self._score_basic(expected, answer)

    def get_metric_name(self) -> str:
        return "probe"

    @staticmethod
    def _score_known_answer(expected: str, answer: str) -> ScoreResult:
        """精确匹配。"""
        expected_stripped = expected.strip().lower()
        # 从 answer 中提取纯文本（去掉 JSON 包裹等）
        answer_clean = _extract_text(answer).lower()

        if expected_stripped in answer_clean:
            return ScoreResult(
                score=100.0, passed=True,
                details={"match": "exact"},
                reasoning=f"Exact match: '{expected_stripped}' found",
            )

        # 模糊匹配：数字提取后对比
        expected_nums = re.findall(r"[-+]?\d*\.?\d+", expected_stripped)
        answer_nums = re.findall(r"[-+]?\d*\.?\d+", answer_clean)
        if expected_nums and answer_nums:
            if expected_nums[0] in answer_nums:
                return ScoreResult(
                    score=100.0, passed=True,
                    details={"match": "numeric_fuzzy"},
                    reasoning=f"Numeric match: {expected_nums[0]}",
                )

        return ScoreResult(
            score=0.0, passed=False,
            details={"expected": expected, "actual": answer[:100]},
            reasoning=f"No match. Expected '{expected_stripped}'",
        )

    @staticmethod
    def _score_reasoning(expected: str, answer: str) -> ScoreResult:
        """推理题：答案包含正确数值即通过。"""
        expected_nums = re.findall(r"[-+]?\d*\.?\d+", expected.strip())
        if not expected_nums:
            return ScoreResult(score=0.0, passed=False,
                               details={"error": "no numeric expected_answer"},
                               reasoning="No numeric answer to match")

        answer_clean = _extract_text(answer).lower()
        for num in expected_nums:
            if num in answer_clean:
                return ScoreResult(
                    score=100.0, passed=True,
                    details={"match": "numeric_in_answer"},
                    reasoning=f"Found '{num}' in answer",
                )

        return ScoreResult(
            score=0.0, passed=False,
            details={"expected": expected_nums, "actual": answer[:200]},
            reasoning=f"Answer doesn't contain any of {expected_nums}",
        )

    @staticmethod
    def _score_format(expected: str, answer: str) -> ScoreResult:
        """格式题：检查 JSON 格式 + 答案存在。"""
        answer_clean = _extract_text(answer)
        # JSON 格式检查
        has_json = False
        try:
            obj = json.loads(answer_clean)
            has_json = isinstance(obj, dict)
        except (json.JSONDecodeError, ValueError):
            # 尝试提取 JSON 块
            depth = 0
            start = -1
            for i, ch in enumerate(answer_clean):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}" and depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            obj = json.loads(answer_clean[start:i+1])
                            if isinstance(obj, dict):
                                has_json = True
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

        format_score = 50.0 if has_json else 0.0

        # 答案存在性检查
        answer_score = 50.0 if expected.strip() and answer.strip() else 0.0

        total = format_score + answer_score
        passed = total >= 50.0
        return ScoreResult(
            score=total, passed=passed,
            details={"format": format_score, "answer": answer_score, "has_json": has_json},
            reasoning=f"format={'valid' if has_json else 'invalid'}, answer={'present' if answer.strip() else 'empty'}",
        )

    @staticmethod
    def _score_basic(expected: str, answer: str) -> ScoreResult:
        """基础评分：答案非空且有内容。"""
        answer_stripped = answer.strip()
        if not answer_stripped:
            return ScoreResult(score=0.0, passed=False,
                               details={"error": "empty_answer"},
                               reasoning="Empty answer")

        # 检查是否包含 expected_answer 的关键信息
        expected_clean = expected.strip().lower()
        answer_clean = answer_stripped.lower()
        if expected_clean and expected_clean in answer_clean:
            return ScoreResult(score=100.0, passed=True,
                               details={"match": "substring"},
                               reasoning="Answer contains expected content")

        return ScoreResult(
            score=50.0, passed=True,
            details={"answer_length": len(answer_stripped)},
            reasoning="Non-empty answer",
        )


def _extract_text(text: str) -> str:
    """从可能的 JSON 包裹中提取纯文本。"""
    text = text.strip()
    # 尝试 JSON 解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # 优先取 answer 字段
            if "answer" in obj:
                return str(obj["answer"])
            # 否则取第一个字符串值
            for v in obj.values():
                if isinstance(v, str) and v.strip():
                    return v
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return text
