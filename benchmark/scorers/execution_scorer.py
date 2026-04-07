"""执行验证评分器.在 subprocess 沙箱中运行模型生成的代码并检查测试用例."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ExecutionScorer(BaseScorer):
    """执行验证评分器，用于 backend-dev 维度（BigCodeBench）。

    将模型生成的代码写入临时文件，附加测试用例，
    在 subprocess 中执行，30 秒超时。
    退出码 0 → score=100，非 0 → score=0。
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def score(self, ctx: ScoringContext) -> ScoreResult:
        # 模型输出为空时直接判 0 分，避免仅执行测试代码意外通过
        if not ctx.model_answer.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "Empty model output"},
                reasoning="Model produced no code",
            )

        test_code = ctx.task.metadata.get("test", "")
        entry_point = ctx.task.metadata.get("entry_point", "")

        full_code = self._build_executable(ctx.model_answer, test_code, entry_point)

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_exec_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            return self._run_and_score(temp_path, ctx.task.task_id)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _evaluate_result(self, returncode: int | None, stdout: str, stderr: str) -> ScoreResult:
        """根据子进程退出码和输出构建评分结果。"""
        if returncode == 0:
            return ScoreResult(
                score=100.0, passed=True,
                details={"stdout": stdout[-500:]},
                reasoning="All test cases passed",
            )
        return ScoreResult(
            score=0.0, passed=False,
            details={"returncode": returncode, "stderr": stderr[-1000:]},
            reasoning=f"Execution failed with return code {returncode}",
        )

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

            return self._evaluate_result(result.returncode, result.stdout, result.stderr)

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

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """异步评分：用 asyncio 原生子进程替代 subprocess.run。"""
        if not ctx.model_answer.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "Empty model output"},
                reasoning="Model produced no code",
            )

        test_code = ctx.task.metadata.get("test", "")
        entry_point = ctx.task.metadata.get("entry_point", "")
        full_code = self._build_executable(ctx.model_answer, test_code, entry_point)

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_exec_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            return await self._async_run_and_score(temp_path, ctx.task.task_id)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _async_run_and_score(self, script_path: str, task_id: str) -> ScoreResult:
        """异步执行脚本并评分。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    details={"error": f"Timeout after {self.timeout}s"},
                    reasoning=f"Execution timed out after {self.timeout} seconds",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            return self._evaluate_result(proc.returncode, stdout, stderr)

        except Exception as exc:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": str(exc)},
                reasoning=f"Execution error: {exc}",
            )
