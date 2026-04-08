"""从模型输出中提取 13 个质量信号。

信号分为两类：
- 纯计算类（不需要查询历史）：format_compliance, repetition_ratio,
  garbled_text_ratio, refusal_detected, language_consistency,
  thinking_ratio, empty_reasoning, truncated
- 历史基线类（需要查询历史计算 z-score）：output_length_zscore,
  token_efficiency_zscore, tps_zscore, ttft_zscore
- 批次级计算（由 StabilityAnalyzer 负责）：answer_entropy
"""

from __future__ import annotations

import json
import math
import re
import statistics
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.models.database import Database
    from benchmark.models.schemas import TaskDefinition

# 拒答模式（大小写不敏感）
_REFUSAL_PATTERNS = re.compile(
    r"作为.*AI"
    r"|I cannot"
    r"|I'm unable"
    r"|I am unable"
    r"|抱歉.*无法"
    r"|Sorry.*can't"
    r"|Sorry, I cannot",
    re.IGNORECASE,
)

# 英文单词（连续 a-zA-Z）
_EN_WORD_RE = re.compile(r"[a-zA-Z]+")

# CJK 字符范围
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


class QualitySignalCollector:
    """从单条评测结果中提取质量信号并持久化。"""

    def __init__(self, db: Database, model: str) -> None:
        self._db = db
        self._model = model
        self._cache: dict = {}

    def _get_cache_key(self, query_key: str, filters: dict[str, str]) -> str:
        dimension = filters.get("dimension", "")
        task_id = filters.get("task_id", "")
        return f"{self._model}:{query_key}:{dimension}:{task_id}"

    # ── 公共 API ──

    async def collect_and_save(
        self,
        result_id: str,
        raw_output: str,
        reasoning_content: str | None,
        gen_metrics: dict,
        finish_reason: str,
        task: TaskDefinition,
        dimension: str,
    ) -> dict:
        """提取所有信号并保存到数据库，返回信号字典。"""
        signals = {
            "format_compliance": self._check_format(raw_output, dimension),
            "repetition_ratio": self._calc_repetition(raw_output),
            "garbled_text_ratio": self._calc_garbled(raw_output),
            "refusal_detected": self._check_refusal(raw_output),
            "language_consistency": self._calc_language_consistency(raw_output),
            "output_length_zscore": await self._calc_length_zscore(
                raw_output, dimension, task
            ),
            "thinking_ratio": self._calc_thinking_ratio(gen_metrics),
            "empty_reasoning": self._check_empty_reasoning(
                reasoning_content, gen_metrics
            ),
            "truncated": 1 if finish_reason == "length" else 0,
            "token_efficiency_zscore": await self._calc_token_efficiency_zscore(
                gen_metrics, task
            ),
            "tps_zscore": await self._calc_tps_zscore(gen_metrics),
            "ttft_zscore": await self._calc_ttft_zscore(gen_metrics),
            "answer_entropy": 0.0,  # 由 StabilityAnalyzer 批次级计算
            "raw_output_length": len(raw_output),
        }
        await self._db.asave_quality_signals(
            {
                "result_id": result_id,
                **signals,
            }
        )
        return signals

    # ── 信号 1: format_compliance ──

    @staticmethod
    def _check_format(raw_output: str, dimension: str) -> float:
        if dimension == "reasoning":
            return 1.0 if "\\boxed{" in raw_output else 0.0
        if dimension in ("backend-dev", "system-architecture"):
            return QualitySignalCollector._has_valid_json(raw_output)
        if dimension == "frontend-dev":
            return QualitySignalCollector._has_valid_json_with_code(raw_output)
        # 其他维度无特定格式要求
        return 1.0

    @staticmethod
    def _has_valid_json(text: str) -> float:
        """尝试在文本中找到有效 JSON dict。"""
        # 寻找第一个 { ... } 块
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return 1.0
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = -1
        return 0.0

    @staticmethod
    def _has_valid_json_with_code(text: str) -> float:
        """检测 JSON dict 且包含 'code' 键。"""
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict) and "code" in obj:
                            return 1.0
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = -1
        return 0.0

    # ── 信号 2: repetition_ratio ──

    @staticmethod
    def _calc_repetition(raw_output: str) -> float:
        words = raw_output.split()
        if len(words) < 3:
            return 0.0
        trigrams: set[str] = set()
        total = 0
        for i in range(len(words) - 2):
            tri = (words[i], words[i + 1], words[i + 2])
            total += 1
            trigrams.add(tri)
        if total == 0:
            return 0.0
        return (total - len(trigrams)) / total

    # ── 信号 3: garbled_text_ratio ──

    @staticmethod
    def _calc_garbled(raw_output: str) -> float:
        if not raw_output:
            return 0.0
        garbled = 0
        for ch in raw_output:
            if ch in ("\n", "\t", "\r"):
                continue
            if not ch.isprintable():
                garbled += 1
                continue
            cp = ord(ch)
            if 0xE000 <= cp <= 0xF8FF:
                garbled += 1
        return garbled / len(raw_output)

    # ── 信号 4: refusal_detected ──

    @staticmethod
    def _check_refusal(raw_output: str) -> int:
        return 1 if _REFUSAL_PATTERNS.search(raw_output) else 0

    # ── 信号 5: language_consistency ──

    @staticmethod
    def _calc_language_consistency(raw_output: str) -> float:
        cjk_count = sum(1 for ch in raw_output if _is_cjk(ch))
        en_words = _EN_WORD_RE.findall(raw_output)
        en_count = len(en_words)

        if cjk_count == 0 and en_count == 0:
            return 1.0
        if cjk_count == 0 or en_count == 0:
            return 1.0

        ratio = min(cjk_count, en_count) / max(cjk_count, en_count)
        # 比例 < 0.1 时才认为混杂（一种语言占比 < 10%）
        if ratio < 0.1:
            return ratio
        return 1.0

    # ── 信号 6: output_length_zscore ──

    async def _calc_length_zscore(
        self, raw_output: str, dimension: str, task: TaskDefinition
    ) -> float:
        mean, std = await self._get_history_stats(
            query_key="output_length",
            filters={
                "dimension": dimension,
                "task_id": task.task_id,
            },
            value_expr="LENGTH(er.model_output)",
        )
        if std == 0:
            return 0.0
        return (len(raw_output) - mean) / std

    # ── 信号 7: thinking_ratio ──

    @staticmethod
    def _calc_thinking_ratio(gen_metrics: dict) -> float:
        reasoning_tokens = gen_metrics.get("reasoning_tokens", 0)
        completion_tokens = gen_metrics.get("completion_tokens", 1)
        if completion_tokens == 0:
            return 0.0
        ratio = reasoning_tokens / completion_tokens
        return max(0.0, min(1.0, ratio))

    # ── 信号 8: empty_reasoning ──

    @staticmethod
    def _check_empty_reasoning(reasoning_content: str | None, gen_metrics: dict) -> int:
        """模型配置了 thinking 但当前 reasoning_content 为空时返回 1。"""
        if reasoning_content is not None and reasoning_content.strip():
            return 0
        # reasoning_content 为空，检查是否有 reasoning_tokens 记录
        # 有 token 计数但无内容 = 内容丢失
        if gen_metrics.get("reasoning_tokens", 0) > 0:
            return 1
        return 0

    # ── 信号 10: token_efficiency_zscore ──

    async def _calc_token_efficiency_zscore(
        self, gen_metrics: dict, task: TaskDefinition
    ) -> float:
        mean, std = await self._get_history_stats(
            query_key="token_efficiency",
            filters={"task_id": task.task_id},
            value_expr="acm.prompt_tokens",
        )
        if std == 0:
            return 0.0
        return (gen_metrics.get("prompt_tokens", 0) - mean) / std

    # ── 信号 11: tps_zscore ──

    async def _calc_tps_zscore(self, gen_metrics: dict) -> float:
        mean, std = await self._get_history_stats(
            query_key="tps",
            filters={},
            value_expr="acm.tokens_per_second",
        )
        if std == 0:
            return 0.0
        return (gen_metrics.get("tokens_per_second", 0) - mean) / std

    # ── 信号 12: ttft_zscore ──

    async def _calc_ttft_zscore(self, gen_metrics: dict) -> float:
        mean, std = await self._get_history_stats(
            query_key="ttft",
            filters={},
            value_expr="acm.ttft_content",
        )
        if std == 0:
            return 0.0
        return (gen_metrics.get("ttft_content", 0) - mean) / std

    # ── 历史基线查询 ──

    async def _get_history_stats(
        self,
        query_key: str,
        filters: dict[str, str],
        value_expr: str,
        days: int = 7,
    ) -> tuple[float, float]:
        """查询历史均值和标准差，返回 (mean, std)。"""
        cache_key = self._get_cache_key(query_key, filters)
        if cache_key in self._cache:
            return self._cache[cache_key]

        sql = f"""
            SELECT {value_expr} AS val
            FROM api_call_metrics acm
            JOIN eval_results er ON acm.result_id = er.result_id
            JOIN eval_runs ev ON er.run_id = ev.run_id
            WHERE ev.model = ?
              AND ev.status = 'completed'
              AND acm.created_at >= datetime('now', ?)
        """
        params: list = [self._model, f"-{days} days"]

        if "dimension" in filters:
            sql += " AND ev.dimension = ?"
            params.append(filters["dimension"])
        if "task_id" in filters:
            sql += " AND er.task_id = ?"
            params.append(filters["task_id"])

        import asyncio

        rows = await asyncio.to_thread(self._query_history, sql, params)

        if len(rows) < 2:
            return (0.0, 0.0)

        values = [r["val"] for r in rows if r["val"] is not None]
        if len(values) < 2:
            return (0.0, 0.0)

        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        result = (mean, 0.0) if std == 0 else (mean, std)

        self._cache[cache_key] = result
        return result

    def _query_history(self, sql: str, params: list) -> list[dict]:
        """同步执行历史查询（在 asyncio.to_thread 中运行）。"""
        conn = self._db._get_conn()
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
