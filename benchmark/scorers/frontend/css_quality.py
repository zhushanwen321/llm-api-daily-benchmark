from __future__ import annotations
import re
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

class CSSQualityScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "non_css_type"}, reasoning=f"Not CSS type ({task_type}), default 100")
        code = ctx.model_answer
        has_media_query = bool(re.search(r"@media\s", code))
        has_relative_units = bool(re.search(r"(?:rem|em|%|vw|vh)\b", code))
        has_modern_layout = bool(re.search(r"(?:flexbox|flex|grid)\b", code))
        checks = {"media_query": has_media_query, "relative_units": has_relative_units, "modern_layout": has_modern_layout}
        violations = sum(1 for v in checks.values() if not v)
        if violations == 0:
            score = 100.0
        elif violations <= 3:
            score = 80.0
        else:
            score = 60.0
        return ScoreResult(score=score, passed=score >= 80.0, details={"checks": checks, "violations": violations}, reasoning=f"CSS quality: {checks}, violations={violations}")

    def get_metric_name(self) -> str:
        return "css_quality"
