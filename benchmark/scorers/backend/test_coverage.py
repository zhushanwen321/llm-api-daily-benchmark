"""测试覆盖率评分器。解析 unittest 输出统计通过/失败数。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# unittest 输出解析: "Ran X tests ... OK" 或 "FAILED (failures=N, errors=M)"
_OK_RE = re.compile(r"Ran (\d+) test.*OK", re.DOTALL)
_RAN_RE = re.compile(r"Ran (\d+) test")
_FAIL_RE = re.compile(r"FAILED \((?:failures=(\d+)(?:, )?)?(?:errors=(\d+))?\)")


class TestCoverageScorer(BaseScorer):
    """测试覆盖率评分器。执行代码并统计 unittest 通过率。"""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=0.0, passed=False,
                               details={"error": "empty_output"},
                               reasoning="Empty model output")

        code = self._extract_code(ctx.model_answer)
        test_code = ctx.task.metadata.get("test", "")
        full_code = f"{code}\n\n# --- Test cases ---\n{test_code}"

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_tc_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            result = self._run_and_score(temp_path)
            # 如果从 JSON 提取了 code，记录到 details
            if code != ctx.model_answer:
                details = dict(result.details)
                details["code_extracted"] = True
                return ScoreResult(
                    score=result.score,
                    passed=result.passed,
                    details=details,
                    reasoning=result.reasoning,
                )
            return result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _extract_code(self, model_answer: str) -> str:
        """尝试从 JSON 格式中提取 code 字段。"""
        try:
            data = json.loads(model_answer)
            if isinstance(data, dict) and "code" in data:
                return data["code"]
        except (json.JSONDecodeError, TypeError):
            pass
        return model_answer

    def _run_and_score(self, script_path: str) -> ScoreResult:
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return self._parse_output(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return ScoreResult(score=0.0, passed=False,
                               details={"error": f"timeout_{self.timeout}s"},
                               reasoning=f"Timeout after {self.timeout}s")
        except Exception as exc:
            return ScoreResult(score=0.0, passed=False,
                               details={"error": str(exc)},
                               reasoning=f"Execution error: {exc}")

    def _parse_output(self, stdout: str, stderr: str, returncode: int) -> ScoreResult:
        output = stdout + stderr

        # 首先尝试获取总测试数
        ran_match = _RAN_RE.search(output)
        if not ran_match:
            return ScoreResult(score=0.0, passed=False,
                               details={"returncode": returncode, "output": output[-500:]},
                               reasoning="Cannot parse test output")
        total = int(ran_match.group(1))

        if returncode == 0:
            # 全部通过
            return ScoreResult(
                score=100.0, passed=True,
                details={"total": total, "passed": total, "failed": 0},
                reasoning=f"All {total} tests passed",
            )

        # 尝试解析 FAILED 输出
        fail_match = _FAIL_RE.search(output)
        if fail_match:
            failures = int(fail_match.group(1) or 0)
            errors = int(fail_match.group(2) or 0)
            failed = failures + errors
            passed = total - failed
            score = (passed / total) * 100 if total > 0 else 0.0
            return ScoreResult(
                score=score, passed=False,
                details={"total": total, "passed": passed, "failed": failed,
                          "failures": failures, "errors": errors},
                reasoning=f"{passed}/{total} tests passed",
            )

        return ScoreResult(score=0.0, passed=False,
                           details={"returncode": returncode, "output": output[-500:]},
                           reasoning="Cannot parse test output")

    def get_metric_name(self) -> str:
        return "test_coverage"
