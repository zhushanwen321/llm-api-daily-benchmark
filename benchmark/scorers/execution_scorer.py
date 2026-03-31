"""执行验证评分器.在 subprocess 沙箱中运行模型生成的代码并检查测试用例."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from benchmark.models.schemas import ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class ExecutionScorer(BaseScorer):
    """执行验证评分器，用于 backend-dev 维度（BigCodeBench）。

    将模型生成的代码写入临时文件，附加测试用例，
    在 subprocess 中执行，30 秒超时。
    退出码 0 → score=100，非 0 → score=0。
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def score(
        self,
        model_output: str,
        expected: str,  # noqa: ARG002 — 基类接口要求
        task: TaskDefinition,
    ) -> ScoreResult:
        test_code = task.metadata.get("test", "")
        entry_point = task.metadata.get("entry_point", "")

        full_code = self._build_executable(model_output, test_code, entry_point)

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_exec_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            return self._run_and_score(temp_path, task.task_id)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _build_executable(
        self, model_code: str, test_code: str, entry_point: str
    ) -> str:
        """构造包含模型代码和测试的完整可执行脚本."""
        parts = [model_code]
        if test_code:
            parts.append("")
            parts.append("# --- Test cases ---")
            parts.append(test_code)
        return "\n".join(parts)

    def _run_and_score(self, script_path: str, task_id: str) -> ScoreResult:
        """执行脚本并评分."""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return ScoreResult(
                    score=100.0,
                    passed=True,
                    details={"stdout": result.stdout[-500:]},
                    reasoning="All test cases passed",
                )

            return ScoreResult(
                score=0.0,
                passed=False,
                details={
                    "returncode": result.returncode,
                    "stderr": result.stderr[-1000:],
                },
                reasoning=f"Execution failed with return code {result.returncode}",
            )

        except subprocess.TimeoutExpired:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": f"Timeout after {self.timeout}s"},
                reasoning=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as exc:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": str(exc)},
                reasoning=f"Execution error: {exc}",
            )

    def get_metric_name(self) -> str:
        return "execution"
