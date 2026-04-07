from __future__ import annotations
import re
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

class CodeOrganizationScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("javascript", "react"):
            return ScoreResult(score=100.0, passed=True, details={"reason": "non_js_type"}, reasoning=f"Not JS type ({task_type}), default 100")
        code = ctx.model_answer
        violations = 0
        if task_type == "react":
            components = re.findall(r"function\s+(\w+)", code)
            non_pascal = [c for c in components if c[0].islower() and c not in ("if", "for", "while", "return")]
            violations += len(non_pascal)
        functions = re.split(r"function\s+\w+\s*\(", code)
        for func_body in functions[1:]:
            lines = func_body.split("\n")
            if len(lines) > 50:
                violations += 1
        if violations == 0:
            score = 100.0
        elif violations <= 5:
            score = 80.0
        else:
            score = 60.0
        return ScoreResult(score=score, passed=score >= 80.0, details={"violations": violations}, reasoning=f"Code organization: violations={violations}")

    def get_metric_name(self) -> str:
        return "code_organization"
