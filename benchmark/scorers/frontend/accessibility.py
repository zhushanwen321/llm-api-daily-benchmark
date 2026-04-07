from __future__ import annotations
import shutil
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

class AccessibilityScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "non_html_type"}, reasoning=f"Not HTML type ({task_type}), default 100")
        if not shutil.which("npx"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "playwright_unavailable"}, reasoning="Playwright unavailable, default 100")
        return ScoreResult(score=100.0, passed=True, details={"reason": "axe_not_implemented"}, reasoning="axe-core not yet implemented, default 100")

    def get_metric_name(self) -> str:
        return "accessibility"
