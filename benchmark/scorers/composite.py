"""组合评分器。按权重聚合多个子评分器的分数。"""

from __future__ import annotations

import asyncio
import logging

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class CompositeScorer(BaseScorer):
    """按权重聚合多个子评分器，计算加权总分。

    子 scorer 异常时该维度默认 100 分。
    """

    def __init__(self, scorers: list[tuple[float, BaseScorer]]) -> None:
        # 过滤掉权重为 0 的 scorer
        self._scorers = [(w, s) for w, s in scorers if w > 0]
        if not self._scorers:
            raise ValueError("至少需要一个子评分器（权重>0）")
        total_weight = sum(w for w, _ in self._scorers)
        if not (abs(total_weight - 1.0) < 1e-9):
            raise ValueError(f"权重之和必须等于 1.0，当前为 {total_weight}")
        logger.debug(f"CompositeScorer 初始化，{len(self._scorers)} 个子评分器")

    def score(self, ctx: ScoringContext) -> ScoreResult:
        weights: dict[str, float] = {}
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}

        weighted_sum = 0.0
        for weight, scorer in self._scorers:
            name = scorer.get_metric_name()
            weights[name] = weight
            try:
                result = scorer.score(ctx)
                scores[name] = result.score
            except Exception as exc:
                logger.warning("子评分器 %s 异常，默认 100 分: %s", name, exc)
                scores[name] = 100.0
                errors[name] = f"{type(exc).__name__}: {exc}"
            weighted_sum += weight * scores[name]

        score = round(weighted_sum, 2)
        details: dict = {
            "composite.weights": weights,
            "composite.scores": scores,
            "expected": ctx.expected,  # 存储期望答案便于查看
        }
        if errors:
            details["composite.errors"] = errors

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"加权总分={score}",
        )

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """并行执行所有子 scorer。"""
        weights: dict[str, float] = {}
        errors: dict[str, str] = {}
        judge_duration: float = 0.0

        async def _run_one(weight: float, scorer: BaseScorer) -> tuple[str, float]:
            nonlocal judge_duration
            name = scorer.get_metric_name()
            weights[name] = weight
            try:
                result = await scorer.ascore(ctx)
                # 收集子 scorer 中的 judge LLM 实际 API 耗时
                jd = result.details.get("judge_duration", 0.0)
                if jd > 0:
                    judge_duration = max(judge_duration, jd)
                return name, result.score
            except Exception as exc:
                logger.warning("子评分器 %s 异常，默认 100 分: %s", name, exc)
                errors[name] = f"{type(exc).__name__}: {exc}"
                return name, 100.0

        coros = [_run_one(w, s) for w, s in self._scorers]
        results = await asyncio.gather(*coros)

        scores = dict(results)
        weighted_sum = sum(weights[name] * score for name, score in scores.items())
        score = round(weighted_sum, 2)

        details: dict = {
            "composite.weights": weights,
            "composite.scores": scores,
        }
        if judge_duration > 0:
            details["judge_duration"] = judge_duration
        if errors:
            details["composite.errors"] = errors

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"加权总分={score}",
        )

    def get_metric_name(self) -> str:
        return "composite"
