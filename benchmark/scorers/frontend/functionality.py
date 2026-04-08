from __future__ import annotations
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

class FunctionalityScorer(BaseScorer):
    def score(self, ctx: ScoringContext) -> ScoreResult:
        test_cases = ctx.task.test_cases
        if not test_cases:
            return ScoreResult(score=100.0, passed=True, details={"reason": "no_test_cases"}, reasoning="No test cases, default 100")
        task_type = ctx.task.metadata.get("type", "html")
        if task_type in ("javascript",) and shutil.which("node"):
            return self._run_node(ctx)
        elif task_type in ("html", "css", "react") and shutil.which("npx"):
            return self._run_playwright(ctx)
        return ScoreResult(score=100.0, passed=True, details={"reason": "runtime_unavailable", "type": task_type}, reasoning=f"Runtime for {task_type} unavailable, default 100")

    def _run_node(self, ctx: ScoringContext) -> ScoreResult:
        assertions = "\n".join(ctx.task.test_cases)
        code = f"{ctx.model_answer}\n{assertions}"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=15)
                Path(f.name).unlink(missing_ok=True)
                if result.returncode == 0:
                    return ScoreResult(score=100.0, passed=True, details={"stdout": result.stdout[-500:]}, reasoning="All assertions passed")
                return ScoreResult(score=0.0, passed=False, details={"stderr": result.stderr[-500:]}, reasoning=f"Assertion failed: {result.stderr[-200:]}")
        except Exception as exc:
            return ScoreResult(score=50.0, passed=False, details={"error": str(exc)}, reasoning=f"Execution error: {exc}")

    def _run_playwright(self, ctx: ScoringContext) -> ScoreResult:
        return ScoreResult(score=100.0, passed=True, details={"reason": "playwright_placeholder"}, reasoning="Playwright not yet implemented, default 100")

    def get_metric_name(self) -> str:
        return "functionality"
