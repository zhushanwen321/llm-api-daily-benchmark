from __future__ import annotations
from typing import Any
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_SEMANTIC_TAGS = {"header", "nav", "main", "article", "section", "aside", "footer"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

class HTMLSemanticScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "non_html_type"}, reasoning=f"Not HTML type ({task_type}), default 100")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(ctx.model_answer, "html.parser")
        except Exception:
            return ScoreResult(score=100.0, passed=True, details={"reason": "parse_error"}, reasoning="HTML parse error, default 100")
        all_elements = soup.find_all(True)
        if not all_elements:
            return ScoreResult(score=100.0, passed=True, details={"reason": "no_elements"}, reasoning="No HTML elements, default 100")
        semantic_count = sum(1 for el in all_elements if el.name in _SEMANTIC_TAGS)
        ratio = semantic_count / len(all_elements)
        if ratio >= 0.6:
            score = 100.0
        elif ratio >= 0.3:
            score = 60.0
        else:
            score = 30.0
        heading_ok = self._check_heading_hierarchy(soup)
        return ScoreResult(score=score, passed=score >= 60.0, details={
            "semantic_count": semantic_count, "total_elements": len(all_elements),
            "semantic_ratio": round(ratio, 2), "heading_ok": heading_ok,
        }, reasoning=f"Semantic ratio={ratio:.2f}, heading_ok={heading_ok}")

    @staticmethod
    def _check_heading_hierarchy(soup: Any) -> bool:
        headings = soup.find_all(_HEADING_TAGS)
        if not headings:
            return True
        prev_level = 0
        for h in headings:
            level = int(h.name[1])
            if level > prev_level + 1 and prev_level > 0:
                return False
            prev_level = level
        return True

    def get_metric_name(self) -> str:
        return "html_semantic"
