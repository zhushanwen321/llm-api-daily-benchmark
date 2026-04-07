from __future__ import annotations
import re
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_VENDOR_PREFIX_RE = re.compile(r"-(?:webkit|moz|ms|o)-")
_SUPPORTS_RE = re.compile(r"@supports\s")

class BrowserCompatScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "non_css_type"}, reasoning=f"Not CSS type ({task_type}), default 100")
        code = ctx.model_answer
        has_prefixes = bool(_VENDOR_PREFIX_RE.search(code))
        has_supports = bool(_SUPPORTS_RE.search(code))
        if not has_prefixes:
            score = 100.0
        elif has_supports:
            score = 80.0
        else:
            score = 60.0
        return ScoreResult(score=score, passed=score >= 80.0, details={
            "has_vendor_prefixes": has_prefixes, "has_supports": has_supports,
        }, reasoning=f"Browser compat: prefixes={has_prefixes}, supports={has_supports}")

    def get_metric_name(self) -> str:
        return "browser_compat"
