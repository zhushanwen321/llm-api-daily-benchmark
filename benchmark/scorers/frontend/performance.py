from __future__ import annotations
import re
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

class PerformanceScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "non_html_type"}, reasoning=f"Not HTML type ({task_type}), default 100")
        code = ctx.model_answer
        deductions = []
        img_tags = re.findall(r"<img[^>]*>", code, re.IGNORECASE)
        for img in img_tags:
            if "width" not in img.lower() or "height" not in img.lower():
                deductions.append("img_without_dimensions")
        if re.search(r"document\.write\s*\(", code):
            deductions.append("sync_dom_write")
        if re.search(r"new\s+XMLHttpRequest\s*\(\s*\)", code):
            deductions.append("sync_xhr")
        score = max(0.0, 70.0 - len(deductions) * 10)
        return ScoreResult(score=score, passed=score >= 50.0, details={
            "base": 70.0, "deductions": deductions, "deduction_count": len(deductions),
        }, reasoning=f"Performance: base=70, deductions={len(deductions)}")

    def get_metric_name(self) -> str:
        return "performance"
