"""性能评分器。用 timeit 对比生成代码和标准答案的执行时间。"""

from __future__ import annotations

import json
import timeit

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class PerformanceScorer(BaseScorer):
    """性能评分器。对比生成代码与标准答案的执行时间。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        canonical = ctx.task.metadata.get("canonical_solution", "")
        if not canonical:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"reason": "no_canonical_solution"},
                reasoning="No canonical solution, skip performance check",
            )

        code = self._extract_code(ctx.model_answer)

        try:
            canon_time = timeit.repeat(stmt=canonical, setup="pass", number=1000, repeat=3)
            canon_avg = min(canon_time) / 1000
        except Exception:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"reason": "canonical_exec_error"},
                reasoning="Canonical solution exec error, skip",
            )

        try:
            gen_time = timeit.repeat(stmt=code, setup="pass", number=1000, repeat=3)
            gen_avg = min(gen_time) / 1000
        except Exception:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"reason": "gen_exec_error"},
                reasoning="Generated code exec error, skip",
            )

        if canon_avg == 0:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"reason": "canon_time_zero"},
                reasoning="Canonical time is zero, skip",
            )

        ratio = gen_avg / canon_avg
        if ratio < 1.5:
            score = 100.0
        elif ratio < 3:
            score = 75.0
        elif ratio < 10:
            score = 40.0
        else:
            score = 0.0

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details={
                "gen_time_ms": round(gen_avg * 1000, 4),
                "canon_time_ms": round(canon_avg * 1000, 4),
                "ratio": round(ratio, 2),
            },
            reasoning=f"Performance ratio={ratio:.2f}, score={score}",
        )

    @staticmethod
    def _extract_code(model_answer: str) -> str:
        """从 model_answer 中提取代码。支持 JSON 格式或纯代码。"""
        try:
            data = json.loads(model_answer)
            if isinstance(data, dict) and "code" in data:
                return data["code"]
        except (json.JSONDecodeError, TypeError):
            pass
        return model_answer

    def get_metric_name(self) -> str:
        return "performance"
