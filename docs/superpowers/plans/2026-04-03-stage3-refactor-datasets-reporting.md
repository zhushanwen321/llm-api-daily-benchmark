# Stage 3: BaseScorer 重构 + 数据集替换 + 报告生成 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构评测架构以分离编排与评分逻辑，替换饱和数据集提升区分度，新增高级统计和报告功能。

**Architecture:** 引入 `ScoringContext` 统一评分上下文、`BaseEvaluator` 分离评测编排与评分逻辑。BaseScorer 接口从 3 参数简化为接收 `ScoringContext`。新数据集（MATH、MMLU-Pro、Web-Bench）替换饱和的旧数据集（GSM8K、MMLU、FrontCode）。

**Tech Stack:** Python 3.11+, Pydantic 2.x, sympy (MATH评分), jinja2 (报告), datasets (HuggingFace), pytest

**Spec:** `.sisyphus/specs/llm-benchmark-stage3.md`

---

## 文件结构总览

### 新增文件
| 文件 | 职责 |
|------|------|
| `benchmark/core/evaluator.py` | BaseEvaluator + SingleTurnEvaluator |
| `benchmark/adapters/math_adapter.py` | MATH 数据集加载 |
| `benchmark/adapters/mmlu_pro_adapter.py` | MMLU-Pro 数据集加载 |
| `benchmark/adapters/webbench_adapter.py` | Web-Bench 数据集加载 |
| `benchmark/scorers/math_scorer.py` | 数学表达式评分（数值+符号） |
| `benchmark/scorers/playwright_scorer.py` | Web-Bench 测试评分 |
| `benchmark/core/advanced_statistics.py` | Bootstrap CI + t-test |
| `benchmark/core/reporter.py` | HTML 报告生成 |
| `benchmark/templates/report.html` | Jinja2 报告模板 |
| `tests/test_scoring_context.py` | ScoringContext 测试 |
| `tests/test_evaluator.py` | Evaluator 测试 |
| `tests/test_math_scorer.py` | MathScorer 测试 |
| `tests/test_math_adapter.py` | MATHAdapter 测试 |
| `tests/test_mmlu_pro_adapter.py` | MMLUProAdapter 测试 |
| `tests/test_advanced_statistics.py` | 高级统计测试 |

### 修改文件
| 文件 | 变更内容 |
|------|---------|
| `benchmark/models/schemas.py` | 新增 ScoringContext |
| `benchmark/scorers/base.py` | 接口签名 score(ctx) |
| `benchmark/scorers/exact_match_scorer.py` | 迁移到新签名 |
| `benchmark/scorers/execution_scorer.py` | 迁移到新签名 |
| `benchmark/scorers/choice_match_scorer.py` | 迁移到新签名 |
| `benchmark/scorers/keyword_match_scorer.py` | 迁移到新签名 |
| `benchmark/core/prompt_builder.py` | 新增 MATH schema（\boxed{}格式） |
| `benchmark/core/response_parser.py` | 新增 \boxed{} 提取 + system-architecture 处理 |
| `benchmark/adapters/bigcodebench_adapter.py` | 题量 5→15 |
| `benchmark/cli.py` | registry 3-tuple + _evaluate_task 简化 |
| `benchmark/configs/default.yaml` | 维度配置更新 |
| `pyproject.toml` | 新增 sympy, jinja2 依赖 |
| `tests/test_choice_match_scorer.py` | 迁移到新签名 |
| `tests/test_keyword_match_scorer.py` | 迁移到新签名 |

---

## Phase 1: ScoringContext + Evaluator（纯新增，不改旧代码）

### Task 1: ScoringContext 数据类

**Files:**
- Modify: `benchmark/models/schemas.py:92` (文件末尾追加)
- Create: `tests/test_scoring_context.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_scoring_context.py
from benchmark.models.schemas import ScoringContext, TaskDefinition


def test_scoring_context_basic():
    """基本字段赋值和默认值."""
    task = TaskDefinition(
        task_id="test_1",
        dimension="reasoning",
        dataset="math",
        prompt="What is 2+2?",
        expected_output="4",
    )
    ctx = ScoringContext(
        model_answer="4",
        raw_output=r"The answer is \boxed{4}",
        expected="4",
        task=task,
    )
    assert ctx.model_answer == "4"
    assert ctx.raw_output == r"The answer is \boxed{4}"
    assert ctx.expected == "4"
    assert ctx.task.task_id == "test_1"
    assert ctx.execution_trace is None
    assert ctx.execution_metrics is None


def test_scoring_context_with_optional_fields():
    """可选字段赋值."""
    task = TaskDefinition(
        task_id="test_2",
        dimension="backend-dev",
        dataset="bigcodebench",
        prompt="Write a function",
        expected_output="",
    )
    ctx = ScoringContext(
        model_answer="def foo(): pass",
        raw_output="```python\ndef foo(): pass\n```",
        expected="",
        task=task,
        execution_trace=[{"tool": "exec", "result": "ok"}],
        execution_metrics={"time": 1.5},
    )
    assert ctx.execution_trace == [{"tool": "exec", "result": "ok"}]
    assert ctx.execution_metrics == {"time": 1.5}
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_scoring_context.py -v`
Expected: FAIL — `ImportError: cannot import name 'ScoringContext'`

- [ ] **Step 3: 实现 ScoringContext**

在 `benchmark/models/schemas.py` 文件末尾追加:

```python

class ScoringContext(BaseModel):
    """统一的评分上下文."""

    model_answer: str                           # 解析后的答案
    raw_output: str                             # 模型原始输出
    expected: str                               # 期望输出
    task: TaskDefinition                        # 任务定义
    reasoning_content: str = ""                 # 推理过程（从 API reasoning_content 获取）
    gen_metrics: dict | None = None             # API 调用指标（prompt_tokens, completion_tokens, ttft 等）
    execution_trace: list[dict] | None = None   # 工具调用记录（未来扩展用）
    execution_metrics: dict | None = None       # 执行指标（未来扩展用）
```

gen_metrics 字段格式:
```python
{
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "reasoning_tokens": 0,
    "duration": 1.5,
    "tokens_per_second": 33.3,
    "ttft": 0.3,
    "ttft_content": 0.5,
    "truncated": False,
    "finish_reason": "stop",
}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_scoring_context.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/models/schemas.py tests/test_scoring_context.py
git commit -m "feat: 新增 ScoringContext 统一评分上下文"
```

---

### Task 2: BaseEvaluator + SingleTurnEvaluator

**Files:**
- Create: `benchmark/core/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_evaluator.py
import pytest
from unittest.mock import AsyncMock

from benchmark.core.evaluator import SingleTurnEvaluator
from benchmark.models.schemas import GenerateResponse, TaskDefinition


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.agenerate.return_value = GenerateResponse(
        content='{"answer": "42"}',
        prompt_tokens=10,
        completion_tokens=5,
    )
    return llm


@pytest.mark.asyncio
async def test_single_turn_evaluator_reasoning(mock_llm):
    """reasoning 维度: 从 JSON 提取 answer 字段."""
    task = TaskDefinition(
        task_id="gsm8k_1",
        dimension="reasoning",
        dataset="gsm8k",
        prompt="What is 6 * 7?",
        expected_output="42",
    )
    evaluator = SingleTurnEvaluator()
    ctx = await evaluator.evaluate(task, "test/model", mock_llm)

    assert ctx.model_answer == "42"
    assert ctx.raw_output == '{"answer": "42"}'
    assert ctx.expected == "42"
    assert ctx.task.task_id == "gsm8k_1"


@pytest.mark.asyncio
async def test_single_turn_evaluator_backend_dev(mock_llm):
    """backend-dev 维度: 从 JSON 提取 code 字段."""
    task = TaskDefinition(
        task_id="bigcodebench_1",
        dimension="backend-dev",
        dataset="bigcodebench",
        prompt="Write a function",
        expected_output="",
    )
    mock_llm.agenerate.return_value = GenerateResponse(
        content='{"code": "def foo(): pass"}',
        prompt_tokens=10,
        completion_tokens=5,
    )
    evaluator = SingleTurnEvaluator()
    ctx = await evaluator.evaluate(task, "test/model", mock_llm)

    assert ctx.model_answer == "def foo(): pass"
    assert ctx.expected == ""
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.core.evaluator'`

- [ ] **Step 3: 实现 evaluator.py**

```python
# benchmark/core/evaluator.py
"""评测编排器。将"如何调用模型"与"评分逻辑"分离."""
from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.response_parser import parse_response
from benchmark.models.schemas import ScoringContext, TaskDefinition


class BaseEvaluator(ABC):
    """评测编排器基类."""

    @abstractmethod
    async def evaluate(
        self,
        task: TaskDefinition,
        model: str,
        llm: LLMEvalAdapter,
    ) -> ScoringContext:
        """执行评测，返回评分上下文."""


class SingleTurnEvaluator(BaseEvaluator):
    """单轮生成：prompt -> generate -> parse."""

    async def evaluate(
        self,
        task: TaskDefinition,
        model: str,
        llm: LLMEvalAdapter,
    ) -> ScoringContext:
        response = await llm.agenerate(task.prompt, model=model)
        parsed = parse_response(response.content, task.dimension)

        # 保留 API 指标
        duration = response.duration
        completion_tokens = response.completion_tokens
        gen_metrics = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "duration": duration,
            "tokens_per_second": (
                completion_tokens / duration if duration > 0 and completion_tokens > 0 else 0.0
            ),
            "ttft": response.ttft,
            "ttft_content": response.ttft_content,
            "truncated": response.truncated,
            "finish_reason": response.finish_reason,
        }

        return ScoringContext(
            model_answer=parsed.answer,
            raw_output=response.content,
            expected=task.expected_output,
            task=task,
            reasoning_content=response.reasoning_content,
            gen_metrics=gen_metrics,
        )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/core/evaluator.py tests/test_evaluator.py
git commit -m "feat: 新增 BaseEvaluator + SingleTurnEvaluator 评测编排器"
```

---

## Phase 2: BaseScorer 接口迁移

> 前置: Phase 1 完成（ScoringContext 已存在）

### Task 3: BaseScorer 接口改为 score(ctx)

**Files:**
- Modify: `benchmark/scorers/base.py`
- Create: `tests/test_base_scorer.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_base_scorer.py
from benchmark.models.schemas import ScoringContext, ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class DummyScorer(BaseScorer):
    """用于测试 BaseScorer 接口的 mock 评分器."""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        return ScoreResult(
            score=100.0 if ctx.model_answer == ctx.expected else 0.0,
            passed=ctx.model_answer == ctx.expected,
        )

    def get_metric_name(self) -> str:
        return "dummy"


def _make_ctx(answer: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=answer,
        raw_output=answer,
        expected=expected,
        task=TaskDefinition(
            task_id="test",
            dimension="reasoning",
            dataset="test",
            prompt="test",
            expected_output=expected,
        ),
    )


def test_base_scorer_correct_answer():
    scorer = DummyScorer()
    result = scorer.score(_make_ctx("42", "42"))
    assert result.passed is True
    assert result.score == 100.0


def test_base_scorer_wrong_answer():
    scorer = DummyScorer()
    result = scorer.score(_make_ctx("99", "42"))
    assert result.passed is False
    assert result.score == 0.0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_base_scorer.py -v`
Expected: FAIL — `score()` 参数不匹配（旧签名需要 3 个参数）

- [ ] **Step 3: 修改 BaseScorer 接口**

将 `benchmark/scorers/base.py` 改为:

```python
# benchmark/scorers/base.py
"""评分器基类.

所有评分器必须继承 BaseScorer 并实现 score/get_metric_name 方法。
Stage 3 重构: score() 接收 ScoringContext 替代原来的 3 个参数。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.models.schemas import ScoreResult, ScoringContext


class BaseScorer(ABC):
    """评分器抽象基类."""

    @abstractmethod
    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行评分.

        Args:
            ctx: 统一评分上下文，包含 model_answer/raw_output/expected/task.

        Returns:
            ScoreResult 包含分数、是否通过、详情、理由。
        """

    @abstractmethod
    def get_metric_name(self) -> str:
        """返回此评分器的指标名称（如 exact_match, execution）。"""
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_base_scorer.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/base.py tests/test_base_scorer.py
git commit -m "refactor: BaseScorer 接口改为 score(ctx: ScoringContext)"
```

---

### Task 4: 迁移 ExactMatchScorer

**Files:**
- Modify: `benchmark/scorers/exact_match_scorer.py`
- Create: `tests/test_exact_match_scorer.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_exact_match_scorer.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.exact_match_scorer import ExactMatchScorer


def _make_ctx(answer: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=answer,
        raw_output=answer,
        expected=expected,
        task=TaskDefinition(
            task_id="test",
            dimension="reasoning",
            dataset="gsm8k",
            prompt="test",
            expected_output=expected,
        ),
    )


def test_exact_match_correct():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("42", "42"))
    assert result.passed is True
    assert result.score == 100.0


def test_exact_match_numeric_close():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("42.0", "42"))
    assert result.passed is True


def test_exact_match_wrong():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("99", "42"))
    assert result.passed is False
    assert result.score == 0.0


def test_exact_match_no_number():
    scorer = ExactMatchScorer()
    result = scorer.score(_make_ctx("no number here", "42"))
    assert result.passed is False
    assert "No number" in result.details["error"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_exact_match_scorer.py -v`
Expected: FAIL — score() 参数签名不匹配

- [ ] **Step 3: 迁移 ExactMatchScorer**

将 `benchmark/scorers/exact_match_scorer.py` 改为:

```python
# benchmark/scorers/exact_match_scorer.py
"""精确匹配评分器。从模型输出中提取数字，与期望答案数值比较."""

from __future__ import annotations

import math
import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """精确匹配评分器，用于 reasoning 维度（GSM8K）。

    从模型输出中提取最后一个数字作为预测答案，
    与 expected 中的期望答案进行数值比较。
    使用 math.isclose 容忍浮点精度差异。
    """

    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def score(self, ctx: ScoringContext) -> ScoreResult:
        model_output = ctx.model_answer
        expected = ctx.expected

        numbers = self._NUMBER_RE.findall(model_output)
        if not numbers:
            return ScoreResult(
                score=0,
                passed=False,
                details={
                    "error": "No number found in output",
                    "raw_output": model_output[:200],
                },
                reasoning="Model output contains no numeric answer",
            )

        predicted_str = numbers[-1]
        expected_str = expected.strip()

        if predicted_str == expected_str:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted_str, "expected": expected_str},
                reasoning=f"Correct: predicted={predicted_str}",
            )

        try:
            predicted_val = float(predicted_str)
            expected_val = float(expected_str)
            passed = math.isclose(predicted_val, expected_val, rel_tol=1e-9)
        except ValueError:
            passed = False

        score = 100.0 if passed else 0.0
        return ScoreResult(
            score=score,
            passed=passed,
            details={"predicted": predicted_str, "expected": expected_str},
            reasoning=(
                f"Correct: predicted={predicted_str}"
                if passed
                else f"Incorrect: predicted={predicted_str}, expected={expected_str}"
            ),
        )

    def get_metric_name(self) -> str:
        return "exact_match"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_exact_match_scorer.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/exact_match_scorer.py tests/test_exact_match_scorer.py
git commit -m "refactor: ExactMatchScorer 迁移到 score(ctx) 接口"
```

---

### Task 5: 迁移 ExecutionScorer

**Files:**
- Modify: `benchmark/scorers/execution_scorer.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_execution_scorer.py
import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.execution_scorer import ExecutionScorer


def _make_ctx(code: str, test_code: str = "", entry_point: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="test_exec",
            dimension="backend-dev",
            dataset="bigcodebench",
            prompt="test",
            expected_output="",
            metadata={"test": test_code, "entry_point": entry_point},
        ),
    )


def test_execution_correct_code():
    code = "def add(a, b):\n    return a + b"
    test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
    scorer = ExecutionScorer(timeout=10)
    result = scorer.score(_make_ctx(code, test, "add"))
    assert result.passed is True
    assert result.score == 100.0


def test_execution_wrong_code():
    code = "def add(a, b):\n    return a - b"
    test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
    scorer = ExecutionScorer(timeout=10)
    result = scorer.score(_make_ctx(code, test, "add"))
    assert result.passed is False
    assert result.score == 0.0


def test_execution_empty_output():
    scorer = ExecutionScorer()
    result = scorer.score(_make_ctx(""))
    assert result.passed is False
    assert result.details["error"] == "Empty model output"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_execution_scorer.py -v`
Expected: FAIL — score() 参数签名不匹配

- [ ] **Step 3: 迁移 ExecutionScorer**

将 `benchmark/scorers/execution_scorer.py` 的 `score` 方法改为:

```python
# benchmark/scorers/execution_scorer.py
"""执行验证评分器.在 subprocess 沙箱中运行模型生成的代码并检查测试用例."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ExecutionScorer(BaseScorer):
    """执行验证评分器，用于 backend-dev 维度（BigCodeBench）。"""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def score(self, ctx: ScoringContext) -> ScoreResult:
        model_output = ctx.model_answer

        if not model_output.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "Empty model output"},
                reasoning="Model produced no code",
            )

        test_code = ctx.task.metadata.get("test", "")
        entry_point = ctx.task.metadata.get("entry_point", "")

        full_code = self._build_executable(model_output, test_code, entry_point)

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_exec_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            return self._run_and_score(temp_path, ctx.task.task_id)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _build_executable(
        self, model_code: str, test_code: str, entry_point: str
    ) -> str:
        parts = [model_code]
        if test_code:
            parts.append("")
            parts.append("# --- Test cases ---")
            parts.append(test_code)
        return "\n".join(parts)

    def _run_and_score(self, script_path: str, task_id: str) -> ScoreResult:
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_execution_scorer.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/execution_scorer.py tests/test_execution_scorer.py
git commit -m "refactor: ExecutionScorer 迁移到 score(ctx) 接口"
```

---

### Task 6: 迁移 ChoiceMatchScorer

**Files:**
- Modify: `benchmark/scorers/choice_match_scorer.py`
- Modify: `tests/test_choice_match_scorer.py`

- [ ] **Step 1: 更新测试**

将 `tests/test_choice_match_scorer.py` 改为:

```python
# tests/test_choice_match_scorer.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.choice_match_scorer import ChoiceMatchScorer


def _make_ctx(model_answer: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=model_answer,
        raw_output=model_answer,
        expected=expected,
        task=TaskDefinition(
            task_id="mmlu_test_1",
            dimension="system-architecture",
            dataset="mmlu",
            prompt="Test",
            expected_output=expected,
            metadata={},
        ),
    )


def test_correct_choice():
    scorer = ChoiceMatchScorer()
    result = scorer.score(_make_ctx("B", "B"))
    assert result.passed is True
    assert result.score == 100.0


def test_correct_choice_with_explanation():
    scorer = ChoiceMatchScorer()
    result = scorer.score(_make_ctx("The answer is B because...", "B"))
    assert result.passed is True


def test_case_insensitive():
    scorer = ChoiceMatchScorer()
    result = scorer.score(_make_ctx("b", "B"))
    assert result.passed is True


def test_wrong_choice():
    scorer = ChoiceMatchScorer()
    result = scorer.score(_make_ctx("A", "B"))
    assert result.passed is False
    assert result.score == 0.0


def test_no_choice_letter():
    scorer = ChoiceMatchScorer()
    result = scorer.score(_make_ctx("maybe 42", "B"))
    assert result.passed is False
    assert "No choice letter" in result.details["error"]


def test_get_metric_name():
    scorer = ChoiceMatchScorer()
    assert scorer.get_metric_name() == "choice_match"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_choice_match_scorer.py -v`
Expected: FAIL — score() 参数签名不匹配

- [ ] **Step 3: 迁移 ChoiceMatchScorer**

将 `benchmark/scorers/choice_match_scorer.py` 改为:

```python
# benchmark/scorers/choice_match_scorer.py
"""选择题评分器。从模型输出中提取选项字母，与期望答案字母比较."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class ChoiceMatchScorer(BaseScorer):
    """选择题字母匹配评分器，用于 MMLU / MMLU-Pro 选择题。"""

    _CHOICE_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected_letter = ctx.expected.strip().upper()
        model_output = ctx.model_answer

        matches = self._CHOICE_RE.findall(model_output)

        if not matches:
            return ScoreResult(
                score=0,
                passed=False,
                details={
                    "error": "No choice letter found in output",
                    "raw_output": model_output[:200],
                },
                reasoning="Model output contains no choice letter",
            )

        predicted = matches[-1].upper()

        passed = predicted == expected_letter
        score = 100.0 if passed else 0.0
        return ScoreResult(
            score=score,
            passed=passed,
            details={"predicted": predicted, "expected": expected_letter},
            reasoning=(
                f"Correct: predicted={predicted}"
                if passed
                else f"Incorrect: predicted={predicted}, expected={expected_letter}"
            ),
        )

    def get_metric_name(self) -> str:
        return "choice_match"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_choice_match_scorer.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/choice_match_scorer.py tests/test_choice_match_scorer.py
git commit -m "refactor: ChoiceMatchScorer 迁移到 score(ctx) 接口"
```

---

### Task 7: 迁移 KeywordMatchScorer

**Files:**
- Modify: `benchmark/scorers/keyword_match_scorer.py`
- Modify: `tests/test_keyword_match_scorer.py`

- [ ] **Step 1: 更新测试**

将 `tests/test_keyword_match_scorer.py` 改为:

```python
# tests/test_keyword_match_scorer.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.keyword_match_scorer import KeywordMatchScorer


def _make_ctx(model_answer: str, keywords: list[str] | None = None) -> ScoringContext:
    return ScoringContext(
        model_answer=model_answer,
        raw_output=model_answer,
        expected="",
        task=TaskDefinition(
            task_id="fc_test",
            dimension="frontend-dev",
            dataset="frontcode",
            prompt="Build a button",
            expected_output="",
            metadata={"keywords": keywords or ["<button>", "onClick"]},
        ),
    )


def test_all_keywords_matched():
    scorer = KeywordMatchScorer()
    result = scorer.score(_make_ctx('<button onClick="handle()">Click</button>'))
    assert result.passed is True
    assert result.score == 100.0


def test_partial_match():
    scorer = KeywordMatchScorer()
    result = scorer.score(_make_ctx("<div>no button here</div>"))
    assert result.score == 50.0
    assert result.passed is True  # >= 50% passes


def test_no_match():
    scorer = KeywordMatchScorer()
    result = scorer.score(_make_ctx("<span>hello</span>"))
    assert result.passed is False
    assert result.score == 0.0


def test_no_keywords_configured():
    scorer = KeywordMatchScorer()
    result = scorer.score(_make_ctx("some code", keywords=[]))
    assert result.passed is False
    assert result.details["error"] == "No keywords configured"


def test_get_metric_name():
    scorer = KeywordMatchScorer()
    assert scorer.get_metric_name() == "keyword_match"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_keyword_match_scorer.py -v`
Expected: FAIL — score() 参数签名不匹配

- [ ] **Step 3: 迁移 KeywordMatchScorer**

将 `benchmark/scorers/keyword_match_scorer.py` 的 `score` 方法改为:

```python
# benchmark/scorers/keyword_match_scorer.py
"""关键词匹配评分器。用于前端代码评测."""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class KeywordMatchScorer(BaseScorer):
    """关键词匹配评分器.

    检查代码是否包含预期的关键词或正则表达式模式。
    根据匹配项数量计算得分（匹配数/总数 * 100）。
    """

    def __init__(self, use_regex: bool = False, case_sensitive: bool = False):
        self.use_regex = use_regex
        self.case_sensitive = case_sensitive

    def score(self, ctx: ScoringContext) -> ScoreResult:
        keywords: list[str] = ctx.task.metadata.get("keywords", [])
        if not keywords:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "No keywords configured"},
                reasoning="No keywords to match",
            )

        model_output = ctx.model_answer
        search_text = model_output if self.case_sensitive else model_output.lower()
        matched = []
        matched_indices = []

        for idx, keyword in enumerate(keywords):
            search_keyword = keyword if self.case_sensitive else keyword.lower()

            if self.use_regex:
                if re.search(search_keyword, search_text):
                    matched.append(keyword)
                    matched_indices.append(idx)
            else:
                if search_keyword in search_text:
                    matched.append(keyword)
                    matched_indices.append(idx)

        score = len(matched) / len(keywords) * 100
        passed = score >= 50.0

        return ScoreResult(
            score=score,
            passed=passed,
            details={
                "matched": matched,
                "matched_indices": matched_indices,
                "total_keywords": len(keywords),
                "match_rate": f"{len(matched)}/{len(keywords)}",
            },
            reasoning=f"Matched {len(matched)}/{len(keywords)} keywords: {matched}",
        )

    def get_metric_name(self) -> str:
        return "keyword_match"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_keyword_match_scorer.py -v`
Expected: 5 passed

- [ ] **Step 5: 运行全量测试确认无回归**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_llm_adapter.py --ignore=tests/test_app_integration.py`
Expected: ALL PASSED（跳过需要 API 的测试）

- [ ] **Step 6: 提交**

```bash
git add benchmark/scorers/keyword_match_scorer.py tests/test_keyword_match_scorer.py
git commit -m "refactor: KeywordMatchScorer 迁移到 score(ctx) 接口"
```

---

<!-- 以下 Phase 将在后续步骤中追加 -->

## Phase 3: CLI 集成新接口

> 前置: Phase 2 完成（所有评分器已迁移）

### Task 8: CLI DIMENSION_REGISTRY 改 3-tuple + _evaluate_task 简化

**Files:**
- Modify: `benchmark/cli.py:37-49` (registry) + `cli.py:109-216` (_evaluate_task + _run_evaluation)

- [ ] **Step 1: 更新 import 和 registry**

在 `benchmark/cli.py` 顶部添加 import:

```python
from benchmark.core.evaluator import SingleTurnEvaluator
```

将 `DIMENSION_REGISTRY` 从 2-tuple 改为 3-tuple:

```python
DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (GSM8KAdapter, ExactMatchScorer, SingleTurnEvaluator),
    "backend-dev": (BigCodeBenchAdapter, ExecutionScorer, SingleTurnEvaluator),
    "system-architecture": (MMLUAdapter, ChoiceMatchScorer, SingleTurnEvaluator),
    "frontend-dev": (FrontCodeAdapter, KeywordMatchScorer, SingleTurnEvaluator),
}
```

- [ ] **Step 2: 简化 _evaluate_task**

将 `_evaluate_task` 函数的核心逻辑替换为:

```python
async def _evaluate_task(
    task_idx: int,
    task: Any,
    model: str,
    llm: LLMEvalAdapter,
    scorer: Any,
    evaluator: Any,
    db: Database,
    run_id: str,
    total: int,
    debug: bool,
) -> dict[str, Any]:
    """单个 task 的异步评测协程。"""
    try:
        logger.debug(f"处理任务 {task_idx + 1}/{total}: {task.task_id}")
        start_time = datetime.now()

        # 委托给 Evaluator 执行评测编排
        ctx = await evaluator.evaluate(task, model, llm)
        execution_time = (datetime.now() - start_time).total_seconds()

        logger.debug(
            f"任务 {task.task_id} 生成完成，输出长度: {len(ctx.raw_output)} 字符"
        )
        if debug:
            logger.debug(f"模型输出:\n{ctx.raw_output[:500]}...")

        # 使用 ScoringContext 评分
        score_result = scorer.score(ctx)
        logger.debug(
            f"任务 {task.task_id} 评分结果: score={score_result.score}, passed={score_result.passed}"
        )

        result_id = str(uuid.uuid4())[:12]
        result = EvalResult(
            result_id=result_id,
            run_id=run_id,
            task_id=task.task_id,
            task_content=task.prompt,
            model_output=ctx.raw_output,
            model_think=ctx.reasoning_content,
            model_answer=ctx.model_answer,
            functional_score=score_result.score,
            final_score=score_result.score,
            passed=score_result.passed,
            details=score_result.details,
            execution_time=execution_time,
            created_at=datetime.now(),
        )
        db.save_result(result)

        # 从 ScoringContext.gen_metrics 恢复 API 指标
        gm = ctx.gen_metrics or {}
        tps = gm.get("tokens_per_second", 0.0)
        db.save_metrics(
            ApiCallMetrics(
                result_id=result_id,
                prompt_tokens=gm.get("prompt_tokens", 0),
                completion_tokens=gm.get("completion_tokens", 0),
                reasoning_tokens=gm.get("reasoning_tokens", 0),
                reasoning_content=ctx.reasoning_content,
                duration=gm.get("duration", execution_time),
                tokens_per_second=tps,
                ttft_content=gm.get("ttft_content", 0.0),
                created_at=datetime.now(),
            )
        )

        status_icon = (
            "[green]PASS[/green]" if score_result.passed else "[red]FAIL[/red]"
        )
        console.print(
            f"  [{task_idx + 1}/{total}] {task.task_id} | "
            f"Score: {score_result.score:.0f} | {status_icon} | "
            f"Time: {execution_time:.1f}s | "
            f"TTFT-R: {gm.get('ttft', 0.0):.2f}s | "
            f"TTFT-C: {gm.get('ttft_content', 0.0):.2f}s | "
            f"Speed: {tps:.1f} tok/s"
        )

        return {
            "score": score_result.score,
            "passed": score_result.passed,
            "task_id": task.task_id,
        }
    except Exception as exc:
        logger.error(f"任务 {getattr(task, 'task_id', task_idx)} 失败: {exc}")
        status_msg = f"[red]ERROR: {type(exc).__name__}: {exc}[/red]"
        console.print(f"  [{task_idx + 1}/{total}] {getattr(task, 'task_id', '?')} | {status_msg}")
        return {
            "error": exc,
            "task_id": getattr(task, "task_id", str(task_idx)),
            "passed": False,
            "score": 0.0,
        }
```

- [ ] **Step 3: 更新 _run_evaluation 解包 3-tuple**

将 `_run_evaluation` 中的解包改为:

```python
    adapter_cls, scorer_cls, evaluator_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    scorer = scorer_cls()
    evaluator = evaluator_cls()
```

以及 coros 构造:

```python
        coros = [
            _evaluate_task(i, task, model, llm, scorer, evaluator, db, run_id, len(tasks), debug)
            for i, task in enumerate(tasks)
        ]
```

- [ ] **Step 4: 运行全量测试确认无回归**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_llm_adapter.py --ignore=tests/test_app_integration.py`
Expected: ALL PASSED

- [ ] **Step 5: 提交**

```bash
git add benchmark/cli.py
git commit -m "refactor: CLI 集成 Evaluator，registry 改 3-tuple"
```

---

## Phase 4: MATH 数据集

> 前置: Phase 3 完成（CLI 已集成新接口）

### Task 9: prompt_builder 按 dataset 判断格式 + response_parser 新增 \boxed{} 提取

> **重要**: 不再按 dimension 全局替换 reasoning 的 prompt 格式，而是按 dataset 判断。
> 这样 GSM8K（dataset="gsm8k"）仍用 JSON 格式，MATH（dataset="math"）用 \boxed{} 格式。

**Files:**
- Modify: `benchmark/core/prompt_builder.py:35-39` (_SCHEMAS dict)
- Modify: `benchmark/core/response_parser.py:81-89` + 新增 extract_boxed 函数

- [ ] **Step 1: 在 prompt_builder.py 中添加 MATH schema**

保留 `_REASONING_SCHEMA`（GSM8K 仍使用），新增 `_MATH_SCHEMA`:

```python
_MATH_SCHEMA = {
    "instruction": (
        "请先展示解题过程，然后将最终答案放在 \\boxed{} 中。\n"
        "例如：如果答案是 42，请写 \\boxed{42}；如果答案是 3/5，请写 \\boxed{\\frac{3}{5}}。\n"
        "不要使用 JSON 格式回答。"
    ),
}
```

更新 `build_structured_prompt` 函数签名，增加 `dataset` 参数:

```python
def build_structured_prompt(task_prompt: str, dimension: str, dataset: str = "") -> str:
    # MATH 数据集：追加 \boxed{} 指令
    if dataset == "math":
        return f"{task_prompt}\n\n---\n{_MATH_SCHEMA['instruction']}"

    schema = _SCHEMAS.get(dimension)
    if not schema:
        return task_prompt

    # 其他维度：追加 JSON 格式要求（原有逻辑不变）
    example = json.dumps(schema["example"], ensure_ascii=False, indent=2)
    parts = [
        task_prompt,
        "",
        "---",
        "请严格按照以下 JSON 格式返回结果，不要在 JSON 之外包含任何其他文本：",
        "```json",
        example,
        "```",
        "",
        "字段说明：",
    ]
    for field, desc in schema["fields"].items():
        parts.append(f"- {field}: {desc}")
    parts.append("")
    parts.append("注意：只返回这个 JSON 对象，不要包含任何其他内容。")
    return "\n".join(parts)
```

**注意**: 所有现有 adapter 调用 `build_structured_prompt(prompt, dimension)` 不传 dataset 参数，走原有逻辑不受影响。只有 MATHAdapter 调用时传 `dataset="math"`。

- [ ] **Step 2: 在 response_parser.py 中添加 \boxed{} 提取**

在 `response_parser.py` 中添加:

```python
# 匹配 \boxed{...}，支持嵌套花括号
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str:
    """从 LaTeX 文本中提取 \\boxed{...} 内的内容.

    处理嵌套花括号，如 \\boxed{\\frac{14}{3}}.
    处理双重嵌套，如 \\boxed{\\boxed{42}} -> 42.

    Returns:
        提取的内容，如果未找到则返回空字符串.
    """
    match = _BOXED_RE.search(text)
    if not match:
        return ""

    # 从 \boxed{ 后面开始，逐字符平衡匹配花括号
    start = match.end()  # 指向 { 后面的第一个字符
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        result = text[start : pos - 1].strip()
    else:
        result = text[start:].strip()

    # 处理双重嵌套: \boxed{\boxed{42}} -> 递归提取
    if result.startswith("\\boxed"):
        inner = extract_boxed(result)
        if inner:
            return inner

    return result
```

更新 `_extract_answer_from_json`:

```python
def _extract_answer_from_json(data: dict, dimension: str) -> str:
    if dimension == "reasoning":
        return str(data.get("answer", data.get("result", "")))
    elif dimension in ("backend-dev", "frontend-dev"):
        return str(data.get("code", ""))
    return ""
```

更新 `parse_response`，在 JSON 解析之前先尝试 `\boxed{}` 提取:

```python
def parse_response(raw: str, dimension: str) -> ParsedResponse:
    if not raw:
        return ParsedResponse(think="", answer="")

    # Step 0: reasoning 维度先尝试 \boxed{} 提取（MATH 数据集）
    if dimension == "reasoning":
        boxed = extract_boxed(raw)
        if boxed:
            return ParsedResponse(think="", answer=boxed)

    # Step 1: 尝试 JSON 解析
    json_data = extract_json_object(raw)
    if json_data:
        answer = _extract_answer_from_json(json_data, dimension)
        return ParsedResponse(think="", answer=answer)

    # Step 2: JSON 解析失败的 fallback
    if dimension == "backend-dev":
        code = extract_python_code(raw)
        if code:
            return ParsedResponse(think="", answer=code)

    # 最终 fallback：原文整体作为 answer
    return ParsedResponse(think="", answer=raw)
```

- [ ] **Step 3: 写 extract_boxed 测试**

```python
# tests/test_response_parser.py (新建)
from benchmark.core.response_parser import extract_boxed, parse_response


def test_extract_boxed_simple():
    assert extract_boxed(r"The answer is \boxed{42}") == "42"


def test_extract_boxed_fraction():
    assert extract_boxed(r"\boxed{\frac{14}{3}}") == r"\frac{14}{3}"


def test_extract_boxed_degree():
    assert extract_boxed(r"\boxed{90^\circ}") == r"90^\circ"


def test_extract_boxed_nested():
    assert extract_boxed(r"\boxed{\frac{3\sqrt{3}}{4}}") == r"\frac{3\sqrt{3}}{4}"


def test_extract_boxed_none():
    assert extract_boxed("no boxed answer here") == ""


def test_parse_response_math_boxed():
    result = parse_response(
        r"Let me solve this... The answer is \boxed{\frac{14}{3}}",
        "reasoning",
    )
    assert result.answer == r"\frac{14}{3}"


def test_parse_response_reasoning_json_fallback():
    """如果无 \\boxed{}，仍回退到 JSON 提取."""
    result = parse_response('{"answer": "42"}', "reasoning")
    assert result.answer == "42"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_response_parser.py -v`
Expected: ALL PASSED

- [ ] **Step 5: 提交**

```bash
git add benchmark/core/prompt_builder.py benchmark/core/response_parser.py tests/test_response_parser.py
git commit -m "feat: MATH 维度 prompt 用 \\boxed{} 格式 + parser 提取 \\boxed{}"
```

---

### Task 10: MATHAdapter 数据集适配器

**Files:**
- Create: `benchmark/adapters/math_adapter.py`
- Create: `tests/test_math_adapter.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_math_adapter.py
from benchmark.adapters.math_adapter import MATHAdapter


def test_math_adapter_loads_tasks():
    adapter = MATHAdapter()
    tasks = adapter.load()

    assert len(tasks) == 15
    assert tasks[0].dimension == "reasoning"
    assert tasks[0].dataset == "math"
    assert tasks[0].expected_output  # answer 不为空
    assert tasks[0].metadata["level"] >= 3
    assert tasks[0].metadata["subject"]


def test_math_adapter_covers_multiple_subjects():
    adapter = MATHAdapter()
    tasks = adapter.load()
    subjects = {t.metadata["subject"] for t in tasks}
    assert len(subjects) >= 4  # 至少覆盖 4 个学科


def test_math_adapter_includes_hard_levels():
    adapter = MATHAdapter()
    tasks = adapter.load()
    levels = [t.metadata["level"] for t in tasks]
    assert max(levels) >= 4  # 包含 Level 4 或 5


def test_math_adapter_validate():
    adapter = MATHAdapter()
    tasks = adapter.load()
    for task in tasks:
        assert adapter.validate(task)


def test_math_adapter_get_dimension():
    adapter = MATHAdapter()
    assert adapter.get_dimension() == "reasoning"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_math_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.adapters.math_adapter'`

- [ ] **Step 3: 实现 MATHAdapter**

```python
# benchmark/adapters/math_adapter.py
"""MATH 数据集适配器。加载 Level 4-5 数学题，覆盖多学科."""

from __future__ import annotations

import os
import random
from typing import List

from datasets import load_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class MATHAdapter(DatasetAdapter):
    """MATH 适配器，选择 Level 3-5 的题目，覆盖多个学科，共 15 题."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        cache_dir = path or os.path.join("benchmark", "datasets", "math")
        dataset = load_dataset(
            "nlile/hendrycks-MATH-benchmark",
            split="test",
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

        # 筛选 Level 3-5 的题目
        eligible = [
            item for item in dataset
            if item["level"] >= 3
        ]

        # 按学科分组，确保覆盖度
        rng = random.Random(42)
        by_subject: dict[str, list] = {}
        for item in eligible:
            subj = item["subject"]
            by_subject.setdefault(subj, []).append(item)

        selected = []
        subjects = list(by_subject.keys())
        rng.shuffle(subjects)

        # 每个学科至少选 1 题，剩余随机分配
        for subj in subjects:
            pool = by_subject[subj]
            rng.shuffle(pool)
            selected.append(pool[0])

        # 补足到 15 题
        remaining = []
        for subj in subjects:
            remaining.extend(by_subject[subj][1:])
        rng.shuffle(remaining)

        for item in remaining:
            if len(selected) >= 15:
                break
            selected.append(item)

        tasks = []
        for idx, item in enumerate(selected[:15]):
            task = TaskDefinition(
                task_id=f"math_{item['unique_id']}",
                dimension="reasoning",
                dataset="math",
                prompt=build_structured_prompt(item["problem"], "reasoning"),
                expected_output=item["answer"],
                metadata={
                    "level": item["level"],
                    "subject": item["subject"],
                    "source": "nlile/hendrycks-MATH-benchmark",
                },
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        return bool(
            task.task_id
            and task.prompt
            and task.expected_output
            and task.dimension == "reasoning"
        )

    def get_dimension(self) -> str:
        return "reasoning"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_math_adapter.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/adapters/math_adapter.py tests/test_math_adapter.py
git commit -m "feat: 新增 MATHAdapter 从 HuggingFace 加载 MATH 数据集"
```

---

### Task 11: MathScorer 数学表达式评分器

**Files:**
- Create: `benchmark/scorers/math_scorer.py`
- Create: `tests/test_math_scorer.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_math_scorer.py
import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.math_scorer import MathScorer


def _make_ctx(predicted: str, expected: str) -> ScoringContext:
    return ScoringContext(
        model_answer=predicted,
        raw_output=predicted,
        expected=expected,
        task=TaskDefinition(
            task_id="math_test",
            dimension="reasoning",
            dataset="math",
            prompt="test",
            expected_output=expected,
        ),
    )


def test_pure_number_match():
    scorer = MathScorer()
    assert scorer.score(_make_ctx("42", "42")).passed is True


def test_numeric_close():
    scorer = MathScorer()
    assert scorer.score(_make_ctx("42.0", "42")).passed is True


def test_fraction_vs_decimal():
    scorer = MathScorer()
    # \frac{14}{3} ≈ 4.666...
    result = scorer.score(_make_ctx(r"\frac{14}{3}", r"\frac{14}{3}"))
    assert result.passed is True


def test_fraction_equivalent():
    scorer = MathScorer()
    # 14/3 和 \frac{14}{3} 应该等价
    result = scorer.score(_make_ctx("14/3", r"\frac{14}{3}"))
    assert result.passed is True


def test_degree_match():
    scorer = MathScorer()
    result = scorer.score(_make_ctx(r"90^\circ", r"90^\circ"))
    assert result.passed is True


def test_algebraic_expression_match():
    scorer = MathScorer()
    result = scorer.score(_make_ctx("p - q", "p - q"))
    assert result.passed is True


def test_wrong_answer():
    scorer = MathScorer()
    result = scorer.score(_make_ctx("99", "42"))
    assert result.passed is False
    assert result.score == 0.0


def test_get_metric_name():
    scorer = MathScorer()
    assert scorer.get_metric_name() == "math_match"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_math_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 MathScorer**

```python
# benchmark/scorers/math_scorer.py
"""数学表达式评分器。支持数值比较和 sympy 符号比较."""

from __future__ import annotations

import math
import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


def _extract_balanced_braces(text: str, start: int) -> tuple[str, int]:
    """从 start 位置（指向 { 后第一个字符）开始，平衡匹配花括号.

    Returns:
        (内容字符串, 结束位置)
    """
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip(), pos


def _normalize_latex(expr: str) -> str:
    """将 LaTeX 表达式转换为可解析的 Python/sympy 表达式."""
    s = expr.strip()

    # 去掉 \left \right
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\left[", "[")
    s = s.replace("\\right]", "]")
    s = s.replace("\\left\\{", "{")
    s = s.replace("\\right\\}", "}")

    # \frac{a}{b} -> (a)/(b) — 用花括号平衡而非正则
    while "\\frac" in s:
        idx = s.find("\\frac")
        # 跳过 \frac 后的空白
        pos = idx + len("\\frac")
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s) or s[pos] != '{':
            break
        # 提取第一个参数
        arg1, pos = _extract_balanced_braces(s, pos + 1)
        # 跳过空白
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s) or s[pos] != '{':
            break
        # 提取第二个参数
        arg2, pos = _extract_balanced_braces(s, pos + 1)
        # 替换
        s = s[:idx] + f"({arg1})/({arg2})" + s[pos:]

    # \sqrt{expr} -> sqrt(expr)  （有花括号）
    while "\\sqrt{" in s:
        idx = s.find("\\sqrt{")
        pos = idx + len("\\sqrt{") - 1  # 指向 {
        arg, pos = _extract_balanced_braces(s, pos + 1)
        s = s[:idx] + f"sqrt({arg})" + s[pos:]

    # \sqrt 后直接跟数字（无花括号）: \sqrt2 -> sqrt(2)
    s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
    # \sqrt 后直接跟字母（无花括号）: \sqrta -> sqrt(a)
    s = re.sub(r"\\sqrt([a-z])", r"sqrt(\1)", s)
    # 裸 \sqrt -> sqrt
    s = s.replace("\\sqrt", "sqrt")

    # ^\circ -> 去掉（角度与纯数值等价）
    s = s.replace("^\\circ", "")

    # \pi -> pi
    s = s.replace("\\pi", "pi")
    # \cdot -> *
    s = s.replace("\\cdot", "*")
    # 去掉多余反斜杠
    s = s.replace("\\", "")
    return s.strip()


def _strip_equals(s: str) -> str:
    """预处理含 = 的表达式: 尝试取等号右侧的值.

    如 "x=5" -> "5", "p - q" -> "p - q"（不变）
    """
    if "=" not in s:
        return s
    parts = s.split("=", 1)
    right = parts[1].strip()
    if right:
        return right
    return s


def _try_numeric_match(a: str, b: str) -> bool:
    """尝试将两个表达式解析为数值并比较."""
    a_norm = _normalize_latex(_strip_equals(a))
    b_norm = _normalize_latex(_strip_equals(b))
    # 安全 eval：限制内置函数
    safe_globals = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        val_a = float(eval(a_norm, safe_globals, {}))  # noqa: S307
        val_b = float(eval(b_norm, safe_globals, {}))  # noqa: S307
        return math.isclose(val_a, val_b, rel_tol=1e-6)
    except Exception:
        return False


def _try_sympy_match(a: str, b: str) -> bool:
    """尝试 sympy 符号比较."""
    try:
        import sympy

        a_norm = _normalize_latex(_strip_equals(a))
        b_norm = _normalize_latex(_strip_equals(b))
        expr_a = sympy.sympify(a_norm)
        expr_b = sympy.sympify(b_norm)
        diff = sympy.simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        return False


class MathScorer(BaseScorer):
    """数学题评分器.

    支持三种匹配模式:
    1. 字符串精确匹配
    2. 数值比较（normalize 后安全 eval）
    3. sympy 符号比较
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        predicted = ctx.model_answer.strip()
        expected = ctx.expected.strip()

        # 快速路径: 字符串精确匹配
        if predicted == expected:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted, "expected": expected, "method": "string"},
                reasoning=f"Correct: {predicted}",
            )

        # 数值比较
        if _try_numeric_match(predicted, expected):
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted, "expected": expected, "method": "numeric"},
                reasoning=f"Correct (numeric): {predicted} == {expected}",
            )

        # sympy 符号比较
        if _try_sympy_match(predicted, expected):
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted, "expected": expected, "method": "sympy"},
                reasoning=f"Correct (symbolic): {predicted} == {expected}",
            )

        return ScoreResult(
            score=0.0,
            passed=False,
            details={"predicted": predicted, "expected": expected},
            reasoning=f"Incorrect: predicted={predicted}, expected={expected}",
        )

    def get_metric_name(self) -> str:
        return "math_match"
```

- [ ] **Step 4: 添加 sympy 到 pyproject.toml 依赖**

在 `pyproject.toml` 的 `dependencies` 列表中添加:

```python
    "sympy>=1.12",
```

- [ ] **Step 5: 安装 sympy**

Run: `.venv/bin/pip install sympy>=1.12`

- [ ] **Step 6: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_math_scorer.py -v`
Expected: 8 passed

- [ ] **Step 7: 提交**

```bash
git add benchmark/scorers/math_scorer.py tests/test_math_scorer.py pyproject.toml
git commit -m "feat: 新增 MathScorer 支持数值+sympy 符号比较"
```

---

## Phase 5: MMLU-Pro + BigCodeBench 扩充

> 前置: Phase 3 完成

### Task 12: MMLUProAdapter 数据集适配器

**Files:**
- Create: `benchmark/adapters/mmlu_pro_adapter.py`
- Create: `tests/test_mmlu_pro_adapter.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_mmlu_pro_adapter.py
from benchmark.adapters.mmlu_pro_adapter import MMLUProAdapter


def test_mmlu_pro_loads_tasks():
    adapter = MMLUProAdapter()
    tasks = adapter.load()

    assert len(tasks) == 15
    assert tasks[0].dimension == "system-architecture"
    assert tasks[0].dataset == "mmlu-pro"
    assert tasks[0].expected_output  # 期望字母不为空
    assert len(tasks[0].expected_output) == 1  # 单个字母


def test_mmlu_pro_prompt_has_options():
    adapter = MMLUProAdapter()
    tasks = adapter.load()
    for task in tasks:
        # prompt 应包含选项
        assert "A." in task.prompt or "B." in task.prompt
        # prompt 应包含 "Answer with the letter"
        assert "Answer with the letter" in task.prompt


def test_mmlu_pro_covers_multiple_categories():
    adapter = MMLUProAdapter()
    tasks = adapter.load()
    categories = {t.metadata["category"] for t in tasks}
    assert len(categories) >= 2  # 至少 2 个学科


def test_mmlu_pro_validate():
    adapter = MMLUProAdapter()
    tasks = adapter.load()
    for task in tasks:
        assert adapter.validate(task)


def test_mmlu_pro_get_dimension():
    adapter = MMLUProAdapter()
    assert adapter.get_dimension() == "system-architecture"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_mmlu_pro_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 MMLUProAdapter**

```python
# benchmark/adapters/mmlu_pro_adapter.py
"""MMLU-Pro 数据集适配器。加载技术相关学科（CS、数学、物理），共 15 题."""

from __future__ import annotations

import os
import random
from typing import List

from datasets import load_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.models.schemas import TaskDefinition


class MMLUProAdapter(DatasetAdapter):
    """MMLU-Pro 适配器，选择技术学科各 5 题，共 15 题."""

    CATEGORIES = ["computer science", "math", "physics"]
    PER_CATEGORY = 5

    def load(self, path: str = "") -> List[TaskDefinition]:
        cache_dir = path or os.path.join("benchmark", "datasets", "mmlu_pro")
        dataset = load_dataset(
            "TIGER-Lab/MMLU-Pro",
            split="test",
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

        rng = random.Random(42)
        tasks = []

        for category in self.CATEGORIES:
            pool = [item for item in dataset if item["category"] == category]
            if not pool:
                continue
            selected = rng.sample(pool, min(self.PER_CATEGORY, len(pool)))

            for item in selected:
                options = item["options"]
                num_options = len(options)
                last_letter = chr(64 + num_options)  # 10 options -> 'J'

                # 构造选择题 prompt
                options_text = "\n".join(
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
                )
                prompt_text = (
                    f"{item['question']}\n\n{options_text}\n\n"
                    f"Answer with the letter (A-{last_letter}). "
                    f"Think step by step, then on the LAST LINE write ONLY the single letter."
                )

                task = TaskDefinition(
                    task_id=f"mmlu_pro_{item['question_id']}",
                    dimension="system-architecture",
                    dataset="mmlu-pro",
                    prompt=prompt_text,
                    expected_output=item["answer"],
                    metadata={
                        "category": category,
                        "source": "TIGER-Lab/MMLU-Pro",
                        "answer_index": item["answer_index"],
                        "num_options": num_options,
                    },
                )
                tasks.append(task)

        return tasks[:15]

    def validate(self, task: TaskDefinition) -> bool:
        return bool(
            task.task_id
            and task.prompt
            and task.expected_output
            and task.dimension == "system-architecture"
        )

    def get_dimension(self) -> str:
        return "system-architecture"
```

注意: MMLU-Pro 不使用 `build_structured_prompt`，因为选择题不需要 JSON 格式，直接在 adapter 中构造完整的 prompt。

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_mmlu_pro_adapter.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/adapters/mmlu_pro_adapter.py tests/test_mmlu_pro_adapter.py
git commit -m "feat: 新增 MMLUProAdapter 加载 MMLU-Pro 数据集"
```

---

### Task 13: BigCodeBench 题量增大 (5→15)

**Files:**
- Modify: `benchmark/adapters/bigcodebench_adapter.py:57` (min(5,...) → min(15,...))

- [ ] **Step 1: 修改题量**

将 `benchmark/adapters/bigcodebench_adapter.py:57` 的:
```python
        selected = rng.sample(lightweight, min(5, len(lightweight)))
```
改为:
```python
        n = min(15, len(lightweight))
        if n < 15:
            logger.warning(f"BigCodeBench: only {n}/{len(dataset)} lightweight tasks available (wanted 15)")
        selected = rng.sample(lightweight, n)
```

同时更新类 docstring 中的 "5 题" → "15 题"。

需要在文件头部加 `import logging` 和 `logger = logging.getLogger(__name__)`（如果还没有的话）。

- [ ] **Step 2: 验证**

Run: `.venv/bin/python -c "
from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
tasks = BigCodeBenchAdapter().load()
print(f'Loaded {len(tasks)} tasks')
assert len(tasks) > 0 and len(tasks) <= 15, f'Expected 1-15, got {len(tasks)}'
print('OK')
"`

- [ ] **Step 3: 提交**

```bash
git add benchmark/adapters/bigcodebench_adapter.py
git commit -m "feat: BigCodeBench-Hard 题量从 5 增加到 15"
```

---

### Task 14: CLI 更新维度注册表 + config 更新

**Files:**
- Modify: `benchmark/cli.py:37-49` (registry + imports)
- Modify: `benchmark/configs/default.yaml`

- [ ] **Step 1: 更新 CLI imports 和 registry**

在 `benchmark/cli.py` 中更新 imports:
```python
from benchmark.adapters.math_adapter import MATHAdapter
from benchmark.adapters.mmlu_pro_adapter import MMLUProAdapter
from benchmark.scorers.math_scorer import MathScorer
```

更新 `DIMENSION_REGISTRY`:
```python
DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (MATHAdapter, MathScorer, SingleTurnEvaluator),
    "backend-dev": (BigCodeBenchAdapter, ExecutionScorer, SingleTurnEvaluator),
    "system-architecture": (MMLUProAdapter, ChoiceMatchScorer, SingleTurnEvaluator),
    "frontend-dev": (FrontCodeAdapter, KeywordMatchScorer, SingleTurnEvaluator),
}
```

更新 `DATASET_REGISTRY`:
```python
DATASET_REGISTRY: dict[str, str] = {
    "reasoning": "math",
    "backend-dev": "bigcodebench",
    "system-architecture": "mmlu-pro",
    "frontend-dev": "frontcode",
}
```

更新 `list-datasets` 命令的描述文本。

- [ ] **Step 2: 更新 default.yaml**

```yaml
dimensions:
  reasoning:
    adapter: "math"
    auto_weight: 0.8
    judge_weight: 0.2

  backend-dev:
    adapter: "bigcodebench"
    auto_weight: 0.8
    judge_weight: 0.2

  system-architecture:
    adapter: "mmlu-pro"
    auto_weight: 0.8
    judge_weight: 0.2

  frontend-dev:
    adapter: "frontcode"
    auto_weight: 0.8
    judge_weight: 0.2
```

- [ ] **Step 3: 运行全量测试**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/test_llm_adapter.py --ignore=tests/test_app_integration.py`
Expected: ALL PASSED

- [ ] **Step 4: 提交**

```bash
git add benchmark/cli.py benchmark/configs/default.yaml
git commit -m "feat: CLI 维度注册表更新为 MATH + MMLU-Pro + BigCodeBench 15题"
```

---

## Phase 6: 高级统计模块

> 前置: 无（独立模块，可与 Phase 4-5 并行）

### Task 15: Bootstrap 置信区间 + t-test 显著性检验

**Files:**
- Create: `benchmark/core/advanced_statistics.py`
- Create: `tests/test_advanced_statistics.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_advanced_statistics.py
import pytest
from benchmark.core.advanced_statistics import (
    bootstrap_confidence_interval,
    ttest_significance,
    pairwise_comparison,
)


def test_bootstrap_ci_basic():
    scores = [80, 90, 100, 70, 60, 85, 95, 75, 80, 90]
    lower, upper = bootstrap_confidence_interval(scores, confidence=0.95)
    assert lower < upper
    assert lower < 82.5 < upper  # 均值应在区间内


def test_bootstrap_ci_single_value():
    """所有值相同时，CI 退化为该值."""
    scores = [50.0] * 10
    lower, upper = bootstrap_confidence_interval(scores)
    assert abs(lower - 50.0) < 1.0
    assert abs(upper - 50.0) < 1.0


def test_ttest_significant():
    a = [80, 90, 100, 70, 60, 85, 95, 75, 80, 90]
    b = [60, 70, 80, 50, 40, 65, 75, 55, 60, 70]
    result = ttest_significance(a, b)
    assert result["is_significant"] is True
    assert result["p_value"] < 0.05
    assert result["effect_size"] > 0  # Cohen's d > 0


def test_ttest_not_significant():
    a = [80, 82, 78, 81, 79]
    b = [80, 81, 79, 82, 78]
    result = ttest_significance(a, b)
    assert result["p_value"] > 0.05


def test_ttest_too_few_samples():
    with pytest.raises(ValueError):
        ttest_significance([1], [2])


def test_pairwise_comparison():
    model_scores = {
        "model_a": [80, 85, 90, 75, 95],
        "model_b": [60, 65, 70, 55, 75],
        "model_c": [78, 82, 88, 72, 90],
    }
    results = pairwise_comparison(model_scores)
    assert len(results) == 3  # C(3,2) = 3 pairs
    # model_a vs model_b 应该显著
    ab = [r for r in results if set([r["model_a"], r["model_b"]]) == {"model_a", "model_b"}]
    assert len(ab) == 1
    assert ab[0]["is_significant"] is True
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_advanced_statistics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 advanced_statistics.py**

```python
# benchmark/core/advanced_statistics.py
"""高级统计分析模块：Bootstrap 置信区间 + t-test 显著性检验."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import scipy.stats


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Bootstrap 重采样置信区间.

    通过有放回抽样计算均值置信区间，
    对样本量小（如 15 题）的情况更稳健。
    """
    arr = np.array(scores)
    rng = np.random.default_rng(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = float(np.percentile(bootstrap_means, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(bootstrap_means, (1 + confidence) / 2 * 100))
    return (lower, upper)


def ttest_significance(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """两模型 t-test 显著性检验."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        raise ValueError("Each group needs at least 2 samples for t-test")

    t_stat, p_value = scipy.stats.ttest_ind(scores_a, scores_b)

    # Cohen's d
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    pooled_std = np.sqrt(
        (np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2
    )
    effect_size = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0

    is_significant = p_value < alpha

    if is_significant:
        if mean_a > mean_b:
            conclusion = f"model_a 显著优于 model_b (p={p_value:.4f}, d={effect_size:.2f})"
        else:
            conclusion = f"model_b 显著优于 model_a (p={p_value:.4f}, d={effect_size:.2f})"
    else:
        conclusion = f"无显著差异 (p={p_value:.4f})"

    return {
        "p_value": float(p_value),
        "is_significant": is_significant,
        "effect_size": effect_size,
        "conclusion": conclusion,
    }


def pairwise_comparison(
    model_scores: dict[str, list[float]],
    alpha: float = 0.05,
) -> list[dict]:
    """多模型两两 t-test 比较."""
    models = list(model_scores.keys())
    results = []
    for model_a, model_b in itertools.combinations(models, 2):
        test_result = ttest_significance(
            model_scores[model_a], model_scores[model_b], alpha
        )
        results.append({
            "model_a": model_a,
            "model_b": model_b,
            **test_result,
        })
    return results
```

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_advanced_statistics.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add benchmark/core/advanced_statistics.py tests/test_advanced_statistics.py
git commit -m "feat: 新增 advanced_statistics (Bootstrap CI + t-test)"
```

---

## Phase 7: 报告生成

> 前置: Phase 6 完成

### Task 16: Reporter + HTML 模板

**Files:**
- Create: `benchmark/core/reporter.py`
- Create: `benchmark/templates/report.html`

- [ ] **Step 1: 创建 Jinja2 HTML 模板**

创建 `benchmark/templates/` 目录（如不存在）。

```html
<!-- benchmark/templates/report.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>LLM Benchmark Report</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 16px 0; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: center; }
        th { background: #f5f5f5; font-weight: 600; }
        .pass { color: #16a34a; font-weight: bold; }
        .fail { color: #dc2626; }
        .section { margin: 32px 0; }
        .section h2 { border-bottom: 2px solid #e5e5e5; padding-bottom: 8px; }
        .significant { background: #fef3c7; }
        img.chart { max-width: 100%; height: auto; margin: 16px 0; }
    </style>
</head>
<body>
    <h1>LLM Benchmark Report</h1>
    <p>Generated: {{ generated_at }}</p>

    <div class="section">
        <h2>1. Overview</h2>
        <p>Models: {{ models | join(', ') }}</p>
        <p>Dimensions: {{ dimensions | join(', ') }}</p>
        <p>Date range: {{ date_range }}</p>
    </div>

    <div class="section">
        <h2>2. Score Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    {% for dim in dimensions %}
                    <th>{{ dim }}</th>
                    {% endfor %}
                    <th>Average</th>
                </tr>
            </thead>
            <tbody>
                {% for row in score_table %}
                <tr>
                    <td>{{ row.model }}</td>
                    {% for dim in dimensions %}
                    <td>{{ row.scores[dim].mean | round(1) }} ({{ row.scores[dim].passed }}/{{ row.scores[dim].total }})</td>
                    {% endfor %}
                    <td><strong>{{ row.average | round(1) }}</strong></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>3. Statistical Tests</h2>
        <table>
            <thead>
                <tr>
                    <th>Model A</th>
                    <th>Model B</th>
                    <th>p-value</th>
                    <th>Effect Size (d)</th>
                    <th>Significant?</th>
                    <th>Conclusion</th>
                </tr>
            </thead>
            <tbody>
                {% for test in stat_tests %}
                <tr class="{{ 'significant' if test.is_significant else '' }}">
                    <td>{{ test.model_a }}</td>
                    <td>{{ test.model_b }}</td>
                    <td>{{ test.p_value | round(4) }}</td>
                    <td>{{ test.effect_size | round(2) }}</td>
                    <td>{{ 'Yes' if test.is_significant else 'No' }}</td>
                    <td>{{ test.conclusion }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>4. Detailed Results</h2>
        {% for dim in dimensions %}
        <h3>{{ dim }}</h3>
        <table>
            <thead>
                <tr>
                    <th>Task ID</th>
                    {% for model in models %}
                    <th>{{ model }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for task_id, results in detailed[dim].items() %}
                <tr>
                    <td>{{ task_id }}</td>
                    {% for model in models %}
                    <td class="{{ 'pass' if results[model].passed else 'fail' }}">
                        {{ results[model].score | round(0) }}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endfor %}
    </div>
</body>
</html>
```

- [ ] **Step 2: 实现 reporter.py**

```python
# benchmark/core/reporter.py
"""报告生成器。从数据库读取结果，生成 HTML 报告."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2

from benchmark.core.advanced_statistics import bootstrap_confidence_interval, pairwise_comparison
from benchmark.models.database import Database


def generate_html_report(
    run_ids: list[str] | None = None,
    models: list[str] | None = None,
    dimensions: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    output_path: str = "report.html",
) -> str:
    """生成 HTML 报告."""
    db = Database()
    try:
        results = db.get_results(model=None, dimension=None)
        if not results:
            raise ValueError("No results found in database")

        # 过滤
        rows = [dict(r) for r in results]
        if models:
            rows = [r for r in rows if r["model"] in models]
        if dimensions:
            rows = [r for r in rows if r["dimension"] in dimensions]
        if date_range:
            start, end = date_range
            rows = [r for r in rows if start <= str(r.get("created_at", ""))[:10] <= end]

        # 提取模型和维度列表
        model_list = sorted(set(r["model"] for r in rows))
        dim_list = sorted(set(r["dimension"] for r in rows))

        # 构建得分表
        score_table = _build_score_table(rows, model_list, dim_list)

        # 统计检验
        stat_tests = []
        if len(model_list) >= 2:
            model_scores = {}
            for model in model_list:
                model_rows = [r for r in rows if r["model"] == model]
                model_scores[model] = [float(r.get("final_score", 0)) for r in model_rows]
            stat_tests = pairwise_comparison(model_scores)

        # 详细结果
        detailed = _build_detailed(rows, model_list, dim_list)

        # 渲染
        template_dir = Path(__file__).parent.parent / "templates"
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        template = env.get_template("report.html")

        html = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            models=model_list,
            dimensions=dim_list,
            date_range=date_range if date_range else "All time",
            score_table=score_table,
            stat_tests=stat_tests,
            detailed=detailed,
        )

        output = Path(output_path)
        output.write_text(html, encoding="utf-8")
        return str(output)
    finally:
        db.close()


def _build_score_table(
    rows: list[dict], models: list[str], dimensions: list[str]
) -> list[dict]:
    """构建得分汇总表."""
    table = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        scores = {}
        for dim in dimensions:
            dim_rows = [r for r in model_rows if r["dimension"] == dim]
            if dim_rows:
                score_vals = [float(r.get("final_score", 0)) for r in dim_rows]
                passed_count = sum(1 for r in dim_rows if r.get("passed"))
                scores[dim] = {
                    "mean": sum(score_vals) / len(score_vals) if score_vals else 0,
                    "passed": passed_count,
                    "total": len(dim_rows),
                }
            else:
                scores[dim] = {"mean": 0, "passed": 0, "total": 0}

        all_means = [s["mean"] for s in scores.values() if s["total"] > 0]
        avg = sum(all_means) / len(all_means) if all_means else 0
        table.append({"model": model, "scores": scores, "average": avg})
    return table


def _build_detailed(
    rows: list[dict], models: list[str], dimensions: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """构建详细结果."""
    detailed = {}
    for dim in dimensions:
        dim_rows = [r for r in rows if r["dimension"] == dim]
        tasks: dict[str, dict[str, Any]] = {}
        for r in dim_rows:
            tid = r.get("task_id", "")
            tasks.setdefault(tid, {})
            tasks[tid][r["model"]] = {
                "score": float(r.get("final_score", 0)),
                "passed": bool(r.get("passed")),
            }
        detailed[dim] = tasks
    return detailed
```

- [ ] **Step 3: 添加 jinja2 到 pyproject.toml 依赖**

在 `pyproject.toml` 的 `dependencies` 中添加:
```python
    "jinja2>=3.1",
```

- [ ] **Step 4: 安装 jinja2**

Run: `.venv/bin/pip install jinja2>=3.1`

- [ ] **Step 5: 验证模板渲染**

Run: `.venv/bin/python -c "
from benchmark.core.reporter import generate_html_report
# 仅测试模板渲染，不依赖实际数据
print('Reporter module loaded OK')
"`

- [ ] **Step 6: 提交**

```bash
git add benchmark/core/reporter.py benchmark/templates/report.html pyproject.toml
git commit -m "feat: 新增 reporter.py + HTML 报告模板"
```

---

### Task 17: CLI report 命令

**Files:**
- Modify: `benchmark/cli.py` (新增 report 子命令)

- [ ] **Step 1: 在 cli.py 中添加 report 命令**

在 `cli.py` 文件末尾（`export` 命令之后）添加:

```python
@cli.command()
@click.option("--models", default=None, help="逗号分隔的模型列表")
@click.option("--dimensions", default=None, help="逗号分隔的维度列表")
@click.option("--date-range", default=None, help="日期范围，格式: 2026-04-01,2026-04-30")
@click.option("--output", default="report.html", help="输出文件路径")
def report(models: str | None, dimensions: str | None, date_range: str | None, output: str) -> None:
    """生成 HTML 评测报告."""
    from benchmark.core.reporter import generate_html_report

    model_list = models.split(",") if models else None
    dim_list = dimensions.split(",") if dimensions else None
    dr = tuple(date_range.split(",")) if date_range else None

    try:
        path = generate_html_report(
            models=model_list,
            dimensions=dim_list,
            date_range=dr,
            output_path=output,
        )
        console.print(f"[green]Report generated: {path}[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
```

- [ ] **Step 2: 验证 CLI 命令注册**

Run: `.venv/bin/python -m benchmark --help`
Expected: 输出中包含 `report` 命令

- [ ] **Step 3: 提交**

```bash
git add benchmark/cli.py
git commit -m "feat: CLI 新增 report 子命令生成 HTML 报告"
```

---

## Phase 6 备注: Web-Bench（暂缓实施）

Web-Bench 数据集的 `project` 字段仅为名称字符串，不含项目源代码和测试用例。完整实施需要:
1. 从 Web-Bench GitHub repo 获取项目脚手架
2. 搭建 Node.js + Playwright 测试环境
3. 实现 PlaywrightScorer

当前 frontend-dev 维度继续使用 FrontCode + KeywordMatchScorer。Web-Bench 留待 Stage 4 或单独的 Phase 实施。

---

## 自检

### Review 修复记录（2026-04-03）

以下问题在 review 后已修复:

| # | 问题 | 修复方式 |
|---|------|---------|
| 1 | ScoringContext 缺少 metrics 字段 | 新增 `reasoning_content` 和 `gen_metrics` 字段 |
| 2 | SingleTurnEvaluator 丢弃 GenerateResponse 指标 | evaluate() 中提取 tokens/ttft/speed 到 gen_metrics |
| 3 | _evaluate_task 中 metrics 和 think 全部丢失 | 从 ctx.gen_metrics 和 ctx.reasoning_content 恢复 |
| 4 | Task 9 和 Task 14 非原子切换破坏 GSM8K | prompt_builder 改为按 dataset 而非 dimension 判断格式 |
| 5 | `_normalize_latex` 的 `\frac` 正则无法处理嵌套花括号 | 改用花括号平衡匹配 `_extract_balanced_braces` |
| 6 | `eval()` 无安全限制 | 限制 globals `{"__builtins__": {}}` + 只暴露 sqrt/pi |
| 7 | `11\sqrt2` 无花括号格式未处理 | 新增 `re.sub(r"\\sqrt(\d+)", ...)` 处理无花括号情况 |
| 8 | 代数表达式含 `=` 号无法匹配 | 新增 `_strip_equals` 预处理，提取等号右侧的值 |
| 9 | `\boxed{\boxed{42}}` 双重嵌套 | extract_boxed 递归检查，提取到最内层 |
| 10 | ChoiceMatchScorer 对长文本易误判 | MMLUProAdapter prompt 加 "LAST LINE write ONLY the letter" |
| 11 | BigCodeBench 过滤后可能不足 15 题 | adapter 加 warning 日志 + 测试改为 `1 <= n <= 15` |

### Spec 覆盖

| Spec 要求 | 对应 Task |
|-----------|----------|
| ScoringContext 数据结构 | Task 1 |
| BaseEvaluator + SingleTurnEvaluator | Task 2 |
| BaseScorer 接口改为 score(ctx) | Task 3 |
| ExactMatchScorer 迁移 | Task 4 |
| ExecutionScorer 迁移 | Task 5 |
| ChoiceMatchScorer 迁移 | Task 6 |
| KeywordMatchScorer 迁移 | Task 7 |
| CLI registry 3-tuple | Task 8 |
| MATH \boxed{} prompt + parser | Task 9 |
| MATHAdapter | Task 10 |
| MathScorer (数值+符号) | Task 11 |
| MMLUProAdapter | Task 12 |
| BigCodeBench 15 题 | Task 13 |
| CLI 维度注册表更新 | Task 14 |
| Bootstrap CI + t-test | Task 15 |
| Reporter + HTML 模板 | Task 16 |
| CLI report 命令 | Task 17 |
| Web-Bench（Playwright） | 暂缓 |

### 类型一致性

- `ScoringContext` 在 Task 1 定义，Task 2-11 全部引用同一类型
- `score(ctx: ScoringContext) -> ScoreResult` 签名在 Task 3 定义，Task 4-7 全部遵循
- `DIMENSION_REGISTRY` 的 3-tuple 解包 `(Adapter, Scorer, Evaluator)` 在 Task 8 和 Task 14 一致
- `MathScorer.get_metric_name()` 返回 `"math_match"` — Task 11 和 Task 14 注册表一致
