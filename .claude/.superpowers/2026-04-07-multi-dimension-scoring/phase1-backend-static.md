# Phase 1: Backend Static Analysis Scorers (Task 5)

> 依赖: [主文档 Task 1](./phase1-infra-backend-sysarch.md) — CompositeScorer

---

## Task 5: Backend 静态分析 Scorers（权重合计 35%）

5 个 scorer，工具不可用时均返回默认 100 分（不惩罚）。

| Scorer | 文件 | 权重 | 工具 |
|--------|------|------|------|
| CodeStyleScorer | `code_style.py` | 15% | pylint + flake8 |
| RobustnessScorer | `robustness.py` | 10% | AST + bandit |
| ArchitectureScorer | `architecture.py` | 5% | radon |
| SecurityScorer | `security.py` | 3% | bandit + semgrep |
| ExtensibilityScorer | `extensibility.py` | 2% | AST |

---

### Task 5a: CodeStyleScorer (15%)

**Files:**
- Create: `benchmark/scorers/backend/code_style.py`
- Test: `tests/test_backend_code_style.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_code_style.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.code_style import CodeStyleScorer


def _make_ctx(code: str) -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={"test": "", "entry_point": ""},
        ),
    )


class TestCodeStyleScorer:
    def test_clean_code(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        scorer = CodeStyleScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score == 100.0

    def test_code_with_warnings(self):
        code = """
def f(x,y,z):
    a=1
    b=2
    return x+y+z+a+b
"""
        scorer = CodeStyleScorer()
        result = scorer.score(_make_ctx(code))
        # 不会满分，但也不应 0 分
        assert 0 < result.score < 100

    def test_empty_output(self):
        scorer = CodeStyleScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert CodeStyleScorer().get_metric_name() == "code_style"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_code_style.py -v
```

- [ ] **Step 3: 实现 CodeStyleScorer**

```python
# benchmark/scorers/backend/code_style.py
"""代码风格评分器。基于 pylint 和 flake8 静态分析。"""

from __future__ import annotations

import subprocess
import tempfile
import os

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


def _tool_available(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_lint(code: str, cmd: list[str]) -> tuple[int, str]:
    """运行 lint 工具，返回 (warning_count, stderr_output)。"""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="bench_lint_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        result = subprocess.run(cmd + [path], capture_output=True, text=True, timeout=30)
        # 统计输出行数作为问题数量的近似
        output = result.stdout + result.stderr
        lines = [l for l in output.strip().split("\n") if l.strip()]
        return len(lines), output[:1000]
    except Exception:
        return 0, ""
    finally:
        if os.path.exists(path):
            os.unlink(path)


class CodeStyleScorer(BaseScorer):
    """pylint + flake8 代码风格评分。"""

    def __init__(self) -> None:
        self._has_pylint = _tool_available(["pylint", "--version"])
        self._has_flake8 = _tool_available(["flake8", "--version"])

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty code, skip")

        if not self._has_pylint and not self._has_flake8:
            return ScoreResult(score=100.0, passed=True, reasoning="pylint/flake8 not available")

        issues = 0
        details = {}

        if self._has_flake8:
            count, output = _run_lint(ctx.model_answer, ["flake8", "--max-line-length=120"])
            issues += count
            details["flake8_issues"] = count

        if self._has_pylint:
            count, output = _run_lint(ctx.model_answer, [
                "pylint", "--disable=all", "--enable=E,F,W",
                "--max-line-length=120", "--score=n",
            ])
            issues += count
            details["pylint_issues"] = count

        # 基础扣分: 每 2 个 issue 扣 5 分，最低 0 分
        score = max(0.0, 100.0 - issues * 2.5)
        score = round(score, 1)

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"代码风格: {issues} issues, score={score}",
        )

    def get_metric_name(self) -> str:
        return "code_style"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_code_style.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/code_style.py tests/test_backend_code_style.py
git commit -m "feat(scorer): add CodeStyleScorer (pylint+flake8)"
```

---

### Task 5b: RobustnessScorer (10%)

**Files:**
- Create: `benchmark/scorers/backend/robustness.py`
- Test: `tests/test_backend_robustness.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_robustness.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.robustness import RobustnessScorer


def _make_ctx(code: str) -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={},
        ),
    )


class TestRobustnessScorer:
    def test_robust_code(self):
        code = """
def safe_divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
"""
        scorer = RobustnessScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score > 50

    def test_bare_except(self):
        code = "def f():\n    try:\n        pass\n    except:\n        pass\n"
        scorer = RobustnessScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score < 100

    def test_empty_code(self):
        scorer = RobustnessScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert RobustnessScorer().get_metric_name() == "robustness"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_robustness.py -v
```

- [ ] **Step 3: 实现 RobustnessScorer**

```python
# benchmark/scorers/backend/robustness.py
"""健壮性评分器。基于 AST 分析 + bandit 安全检查。"""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
import os

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 不良模式正则
_BARE_EXCEPT_RE = re.compile(r"except\s*:", re.MULTILINE)
_BARE_EXCEPT_AST = (ast.ExceptHandler,)  # handler.type is None -> bare except


def _count_bare_excepts(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            count += 1
    return count


def _has_bandit() -> bool:
    try:
        subprocess.run(["bandit", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_bandit(code: str) -> int:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="bench_bandit_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["bandit", "-f", "txt", path],
            capture_output=True, text=True, timeout=30,
        )
        lines = [l for l in result.stdout.strip().split("\n") if "Issue:" in l]
        return len(lines)
    except Exception:
        return 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


class RobustnessScorer(BaseScorer):
    """AST + bandit 健壮性评分。"""

    def __init__(self) -> None:
        self._has_bandit = _has_bandit()

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty code, skip")

        deductions = 0
        details = {}

        bare = _count_bare_excepts(ctx.model_answer)
        if bare:
            deductions += bare * 10
            details["bare_except_count"] = bare

        if self._has_bandit:
            issues = _run_bandit(ctx.model_answer)
            deductions += issues * 5
            details["bandit_issues"] = issues

        score = max(0.0, round(100.0 - deductions, 1))

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"健壮性: deductions={deductions}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "robustness"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_robustness.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/robustness.py tests/test_backend_robustness.py
git commit -m "feat(scorer): add RobustnessScorer (AST+bandit)"
```

---

### Task 5c: ArchitectureScorer (5%)

**Files:**
- Create: `benchmark/scorers/backend/architecture.py`
- Test: `tests/test_backend_architecture.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_architecture.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.architecture import ArchitectureScorer


def _make_ctx(code: str) -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={},
        ),
    )


class TestArchitectureScorer:
    def test_simple_function(self):
        code = "def add(a, b):\n    return a + b\n"
        scorer = ArchitectureScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score == 100.0

    def test_high_complexity(self):
        # 深度嵌套的 if-else 制造高圈复杂度
        code = "def f(x):\n"
        for i in range(10):
            code += f"    if x == {i}:\n        return {i}\n"
        code += "    return -1\n"
        scorer = ArchitectureScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score < 100.0

    def test_empty_code(self):
        scorer = ArchitectureScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert ArchitectureScorer().get_metric_name() == "architecture"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_architecture.py -v
```

- [ ] **Step 3: 实现 ArchitectureScorer**

```python
# benchmark/scorers/backend/architecture.py
"""架构评分器。基于 radon 圈复杂度分析。"""

from __future__ import annotations

import subprocess

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_COMPLEXITY_THRESHOLD = 10  # 超过此值的函数扣分


def _has_radon() -> bool:
    try:
        subprocess.run(["radon", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _get_max_complexity(code: str) -> int | None:
    """返回最大圈复杂度，radon 不可用时返回 None。"""
    try:
        import radon.complexity as cc
        from io import StringIO

        results = cc.cc_visit(code)
        if not results:
            return 0
        return max(item.complexity for item in results)
    except Exception:
        return None


class ArchitectureScorer(BaseScorer):
    """radon 圈复杂度评分。"""

    def __init__(self) -> None:
        self._has_radon = _has_radon()

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty code, skip")

        if not self._has_radon:
            return ScoreResult(score=100.0, passed=True, reasoning="radon not available")

        max_cc = _get_max_complexity(ctx.model_answer)
        if max_cc is None:
            return ScoreResult(score=100.0, passed=True, reasoning="Failed to parse code")

        # 复杂度 <= 5: 满分; 5-10: 线性扣分; >10: 更陡的扣分
        if max_cc <= 5:
            score = 100.0
        elif max_cc <= _COMPLEXITY_THRESHOLD:
            score = 100.0 - (max_cc - 5) * 5
        else:
            score = max(0.0, 75.0 - (max_cc - _COMPLEXITY_THRESHOLD) * 10)
        score = round(score, 1)

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details={"max_complexity": max_cc},
            reasoning=f"圈复杂度: {max_cc}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "architecture"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_architecture.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/architecture.py tests/test_backend_architecture.py
git commit -m "feat(scorer): add ArchitectureScorer (radon complexity)"
```

---

### Task 5d: SecurityScorer (3%)

**Files:**
- Create: `benchmark/scorers/backend/security.py`
- Test: `tests/test_backend_security.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_security.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.security import SecurityScorer


def _make_ctx(code: str) -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={},
        ),
    )


class TestSecurityScorer:
    def test_safe_code(self):
        code = "def add(a, b):\n    return a + b\n"
        scorer = SecurityScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score == 100.0

    def test_insecure_code(self):
        code = 'import subprocess\nsubprocess.call("rm -rf /", shell=True)\n'
        scorer = SecurityScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score < 100.0

    def test_empty_code(self):
        scorer = SecurityScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert SecurityScorer().get_metric_name() == "security"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_security.py -v
```

- [ ] **Step 3: 实现 SecurityScorer**

```python
# benchmark/scorers/backend/security.py
"""安全性评分器。基于 bandit + semgrep 安全扫描。"""

from __future__ import annotations

import subprocess
import tempfile
import os

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# AST 级别的危险模式（即使 bandit 不可用也能检测）
_DANGEROUS_PATTERNS = [
    ("shell=True", 15),      # shell 注入风险
    ("eval(", 20),           # eval 注入
    ("exec(", 20),           # exec 注入
    ("pickle.loads(", 15),   # 反序列化漏洞
    ("__import__(", 5),      # 动态导入
]


def _has_tool(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_bandit(code: str) -> int:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="bench_sec_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["bandit", "-f", "txt", "-ll", path],
            capture_output=True, text=True, timeout=30,
        )
        lines = [l for l in result.stdout.strip().split("\n") if "Issue:" in l]
        return len(lines)
    except Exception:
        return 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_semgrep(code: str) -> int:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="bench_semgrep_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["semgrep", "scan", "--config", "auto", "--json", path],
            capture_output=True, text=True, timeout=60,
        )
        import json
        data = json.loads(result.stdout)
        return len(data.get("results", []))
    except Exception:
        return 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _ast_danger_check(code: str) -> tuple[int, list[str]]:
    """AST 级别的危险模式检测（不依赖外部工具）。"""
    deductions = 0
    found = []
    for pattern, penalty in _DANGEROUS_PATTERNS:
        if pattern in code:
            deductions += penalty
            found.append(pattern)
    return deductions, found


class SecurityScorer(BaseScorer):
    """bandit + semgrep + AST 安全性评分。"""

    def __init__(self) -> None:
        self._has_bandit = _has_tool(["bandit", "--version"])
        self._has_semgrep = _has_tool(["semgrep", "--version"])

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty code, skip")

        deductions, found = _ast_danger_check(ctx.model_answer)
        details: dict = {"ast_dangerous": found}

        if self._has_bandit:
            issues = _run_bandit(ctx.model_answer)
            deductions += issues * 10
            details["bandit_high"] = issues

        if self._has_semgrep:
            issues = _run_semgrep(ctx.model_answer)
            deductions += issues * 10
            details["semgrep_issues"] = issues

        score = max(0.0, round(100.0 - deductions, 1))

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"安全性: deductions={deductions}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "security"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_security.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/security.py tests/test_backend_security.py
git commit -m "feat(scorer): add SecurityScorer (bandit+semgrep+AST)"
```

---

### Task 5e: ExtensibilityScorer (2%)

**Files:**
- Create: `benchmark/scorers/backend/extensibility.py`
- Test: `tests/test_backend_extensibility.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_extensibility.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.extensibility import ExtensibilityScorer


def _make_ctx(code: str) -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={},
        ),
    )


class TestExtensibilityScorer:
    def test_clean_code(self):
        code = "def process(data: list) -> list:\n    return [x * 2 for x in data]\n"
        scorer = ExtensibilityScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score == 100.0

    def test_hardcoded_values(self):
        code = 'def get_url():\n    return "https://api.example.com/v1"\n'
        scorer = ExtensibilityScorer()
        result = scorer.score(_make_ctx(code))
        # 硬编码 URL 应扣分
        assert result.score < 100.0

    def test_magic_numbers(self):
        code = "def calc(x):\n    return x * 86400\n"
        scorer = ExtensibilityScorer()
        result = scorer.score(_make_ctx(code))
        assert result.score < 100.0

    def test_empty_code(self):
        scorer = ExtensibilityScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert ExtensibilityScorer().get_metric_name() == "extensibility"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_backend_extensibility.py -v
```

- [ ] **Step 3: 实现 ExtensibilityScorer**

```python
# benchmark/scorers/backend/extensibility.py
"""可扩展性评分器。AST 级别的硬编码检测。"""

from __future__ import annotations

import ast
import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 硬编码 URL 模式
_HARDCODED_URL_RE = re.compile(r'"https?://[^\"]+"|\'https?://[^\']+\'')

# AST 级别: 检测字符串常量中的 URL
# 硬编码的大数字（非 0, 1, -1, 2 的整数）
_MAGIC_NUMBER_THRESHOLD = 100


def _check_hardcoded_urls(code: str) -> int:
    return len(_HARDCODED_URL_RE.findall(code))


def _check_magic_numbers(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if abs(node.value) >= _MAGIC_NUMBER_THRESHOLD and node.value not in (0, 1, -1):
                count += 1
    return count


class ExtensibilityScorer(BaseScorer):
    """AST 硬编码检测，纯 Python 无外部依赖。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty code, skip")

        urls = _check_hardcoded_urls(ctx.model_answer)
        magic = _check_magic_numbers(ctx.model_answer)
        deductions = urls * 10 + magic * 5
        score = max(0.0, round(100.0 - deductions, 1))

        details = {}
        if urls:
            details["hardcoded_urls"] = urls
        if magic:
            details["magic_numbers"] = magic

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"可扩展性: urls={urls}, magic={magic}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "extensibility"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_backend_extensibility.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/backend/extensibility.py tests/test_backend_extensibility.py
git commit -m "feat(scorer): add ExtensibilityScorer (AST hardcoded detection)"
```
