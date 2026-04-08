# Phase 1: Backend Execution Tasks (Task 2-4)

> 依赖: [主文档 Task 1](./phase1-infra-backend-sysarch.md) — CompositeScorer

---

## Task 2: BigCodeBenchAdapter 添加 canonical_solution

**Files:**
- Modify: `benchmark/adapters/bigcodebench_adapter.py:77-82`

- [ ] **Step 1: 修改 metadata 添加 canonical_solution**

将 `benchmark/adapters/bigcodebench_adapter.py` 第 77-82 行的 metadata 字典修改为：

```python
                metadata={
                    "difficulty": "hard",
                    "source": "bigcode/bigcodebench-hard",
                    "test": item.get("test", ""),
                    "entry_point": item.get("entry_point", ""),
                    "canonical_solution": item.get("canonical_solution", ""),
                },
```

仅新增一行 `"canonical_solution": item.get("canonical_solution", "")`。

- [ ] **Step 2: 验证现有测试不受影响**

```bash
pytest tests/ -k "bigcode or execution" -v
```
Expected: ALL PASS

- [ ] **Step 3: 提交**

```bash
git add benchmark/adapters/bigcodebench_adapter.py
git commit -m "feat(adapter): add canonical_solution to BigCodeBench metadata"
```

---

## Task 3: Backend TestCoverageScorer (权重 40%)

**Files:**
- Create: `benchmark/scorers/backend/__init__.py` (空文件)
- Create: `benchmark/scorers/backend/test_coverage.py`
- Test: `tests/test_backend_test_coverage.py`

**设计要点:**
- 复用 ExecutionScorer 的 subprocess 执行模式
- 解析 unittest 输出统计 `Ran N tests`、`OK`、`FAILED (failures=X, errors=Y)`
- 通过率 = passed / total * 100
- 模型输出为空返回 0 分

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_test_coverage.py
from benchmark.models.schemas import ScoreResult, ScoringContext, TaskDefinition
from benchmark.scorers.backend.test_coverage import TestCoverageScorer


def _make_ctx(code: str, test_code: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={"test": test_code, "entry_point": ""},
        ),
    )


class TestTestCoverageScorer:
    def test_all_pass(self):
        code = "def add(a, b):\n    return a + b"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
        scorer = TestCoverageScorer(timeout=10)
        result = scorer.score(_make_ctx(code, test))
        assert result.score == 100.0
        assert result.passed is True

    def test_partial_pass(self):
        code = "def add(a, b):\n    return a + b"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
    def test_sub(self):
        self.assertEqual(add(5, 3), 1)
if __name__ == "__main__":
    unittest.main()
"""
        scorer = TestCoverageScorer(timeout=10)
        result = scorer.score(_make_ctx(code, test))
        assert result.score == 50.0
        assert result.details["total"] == 2
        assert result.details["passed"] == 1

    def test_all_fail(self):
        code = "def add(a, b):\n    return a - b"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
        scorer = TestCoverageScorer(timeout=10)
        result = scorer.score(_make_ctx(code, test))
        assert result.score == 0.0

    def test_empty_output(self):
        scorer = TestCoverageScorer(timeout=10)
        result = scorer.score(_make_ctx(""))
        assert result.score == 0.0
        assert result.passed is False

    def test_syntax_error(self):
        scorer = TestCoverageScorer(timeout=10)
        result = scorer.score(_make_ctx("def(", "print(1)"))
        assert result.score == 0.0

    def test_timeout(self):
        code = "import time; time.sleep(100)"
        scorer = TestCoverageScorer(timeout=1)
        result = scorer.score(_make_ctx(code, "print(1)"))
        assert result.score == 0.0

    def test_get_metric_name(self):
        assert TestCoverageScorer().get_metric_name() == "test_coverage"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_test_coverage.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 TestCoverageScorer**

```python
# benchmark/scorers/backend/__init__.py
# 空

# benchmark/scorers/backend/test_coverage.py
"""测试覆盖率评分器。解析 unittest 输出计算通过率。"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 匹配 unittest 输出中的测试统计
_RUN_RE = re.compile(r"Ran (\d+) test")
_OK_RE = re.compile(r"OK(?:\s*\((?:skipped=\d+)?\))?$", re.MULTILINE)
_FAILED_RE = re.compile(r"FAILED \((?:failures=(\d+),?\s*)?(?:errors=(\d+))?\)")


class TestCoverageScorer(BaseScorer):
    """解析 unittest 输出统计通过/失败数，计算通过率。"""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=0.0, passed=False, reasoning="Empty model output")

        test_code = ctx.task.metadata.get("test", "")
        entry_point = ctx.task.metadata.get("entry_point", "")
        full_code = self._build_executable(ctx.model_answer, test_code, entry_point)

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_tc_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            return self._run_and_score(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _build_executable(self, model_code: str, test_code: str, entry_point: str) -> str:
        parts = [model_code]
        if test_code:
            parts.append("\n# --- Test cases ---\n")
            parts.append(test_code)
        return "\n".join(parts)

    def _run_and_score(self, script_path: str) -> ScoreResult:
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return self._parse_result(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Timeout after {self.timeout}s")
        except Exception as exc:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Execution error: {exc}")

    def _parse_result(self, stdout: str, stderr: str, returncode: int) -> ScoreResult:
        combined = stdout + "\n" + stderr

        run_match = _RUN_RE.search(combined)
        if not run_match:
            # 非 unittest 输出，回退到退出码判断
            if returncode == 0:
                return ScoreResult(score=100.0, passed=True, reasoning="Exit code 0 (no unittest output)")
            return ScoreResult(
                score=0.0, passed=False,
                details={"returncode": returncode, "stderr": stderr[-500:]},
                reasoning=f"Non-unittest execution failed (exit={returncode})",
            )

        total = int(run_match.group(1))

        if _OK_RE.search(combined):
            passed = total
        else:
            fail_match = _FAILED_RE.search(combined)
            if fail_match:
                failures = int(fail_match.group(1) or 0)
                errors = int(fail_match.group(2) or 0)
                passed = total - failures - errors
            else:
                passed = 0 if returncode != 0 else total

        passed = max(0, passed)
        score = round(passed / total * 100, 1) if total > 0 else 0.0

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details={"total": total, "passed": passed, "failed": total - passed},
            reasoning=f"测试通过率: {passed}/{total} = {score}%",
        )

    def get_metric_name(self) -> str:
        return "test_coverage"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_test_coverage.py -v
```
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/__init__.py benchmark/scorers/backend/test_coverage.py tests/test_backend_test_coverage.py
git commit -m "feat(scorer): add TestCoverageScorer for unittest pass rate"
```

---

## Task 4: Backend PerformanceScorer (权重 25%)

**Files:**
- Create: `benchmark/scorers/backend/performance.py`
- Test: `tests/test_backend_performance.py`

**设计要点:**
- 用 `timeit.repeat()` 对比模型代码和 canonical_solution 的执行时间
- `canonical_solution` 不存在或为空时返回默认 100 分
- 模型代码执行报错时返回 0 分
- 评分公式: `score = min(100, 100 * (canonical_time / model_time))`，模型更快或相当得满分，慢则按比例扣分
- 两者都超时则返回 100 分（不惩罚）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_performance.py
from benchmark.models.schemas import ScoreResult, ScoringContext, TaskDefinition
from benchmark.scorers.backend.performance import PerformanceScorer


def _make_ctx(code: str, canonical: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={"test": "", "entry_point": "", "canonical_solution": canonical},
        ),
    )


class TestPerformanceScorer:
    def test_same_performance(self):
        code = "def f(n): return sum(range(n))"
        canonical = code
        scorer = PerformanceScorer()
        result = scorer.score(_make_ctx(code, canonical))
        assert result.score == 100.0

    def test_slower_model_code(self):
        # 模型用 O(n^2)，标准用 O(n)
        model = "def f(n):\n    s = 0\n    for i in range(n):\n        for j in range(n): s += 1\n    return s"
        canonical = "def f(n): return n * n"
        scorer = PerformanceScorer(benchmark_n=1000, repeats=3)
        result = scorer.score(_make_ctx(model, canonical))
        assert result.score < 100.0
        assert result.score > 0.0
        assert "model_time" in result.details
        assert "canonical_time" in result.details

    def test_no_canonical_solution(self):
        scorer = PerformanceScorer()
        result = scorer.score(_make_ctx("def f(): pass"))
        assert result.score == 100.0
        assert "no canonical_solution" in result.reasoning

    def test_model_code_error(self):
        scorer = PerformanceScorer()
        result = scorer.score(_make_ctx("1/0", "def f(): pass"))
        assert result.score == 0.0

    def test_both_timeout(self):
        scorer = PerformanceScorer(timeout=0.01)
        result = scorer.score(_make_ctx(
            "import time; time.sleep(10)",
            "import time; time.sleep(10)",
        ))
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert PerformanceScorer().get_metric_name() == "performance"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_performance.py -v
```
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 PerformanceScorer**

```python
# benchmark/scorers/backend/performance.py
"""性能评分器。用 timeit 对比模型代码与标准答案的执行时间。"""

from __future__ import annotations

import time
import timeit

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 用于执行计时的辅助代码
_BENCHMARK_WRAPPER = """
def _bench():
    {code}
_bench()
"""


class PerformanceScorer(BaseScorer):
    """对比模型代码和 canonical_solution 的执行时间。"""

    def __init__(self, benchmark_n: int = 10000, repeats: int = 5, timeout: float = 10.0) -> None:
        self.benchmark_n = benchmark_n
        self.repeats = repeats
        self.timeout = timeout

    def score(self, ctx: ScoringContext) -> ScoreResult:
        canonical = ctx.task.metadata.get("canonical_solution", "")
        if not canonical or not canonical.strip():
            return ScoreResult(
                score=100.0, passed=True,
                reasoning="no canonical_solution, skip performance check",
            )

        if not ctx.model_answer.strip():
            return ScoreResult(score=0.0, passed=False, reasoning="Empty model output")

        model_time = self._measure(ctx.model_answer)
        if model_time is None:
            return ScoreResult(score=0.0, passed=False, reasoning="Model code execution error")

        canonical_time = self._measure(canonical)
        if canonical_time is None:
            # 标准答案无法执行，不惩罚模型
            return ScoreResult(score=100.0, passed=True, reasoning="canonical_solution execution error")

        if model_time == 0:
            score = 100.0
        elif canonical_time == 0:
            score = 0.0
        else:
            ratio = canonical_time / model_time
            score = min(100.0, round(ratio * 100, 1))

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details={
                "model_time": round(model_time, 6),
                "canonical_time": round(canonical_time, 6),
                "ratio": round(canonical_time / model_time, 3) if model_time else float("inf"),
            },
            reasoning=f"性能比: canonical/model = {canonical_time:.6f}/{model_time:.6f}",
        )

    def _measure(self, code: str) -> float | None:
        """返回平均执行时间（秒），出错返回 None。"""
        wrapper = _BENCHMARK_WRAPPER.format(code=code.strip())
        try:
            times = timeit.repeat(
                stmt=wrapper,
                number=self.benchmark_n,
                repeat=self.repeats,
                globals={},
            )
            return sum(times) / len(times)
        except Exception:
            return None

    def get_metric_name(self) -> str:
        return "performance"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_performance.py -v
```
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/performance.py tests/test_backend_performance.py
git commit -m "feat(scorer): add PerformanceScorer for timeit-based comparison"
```
