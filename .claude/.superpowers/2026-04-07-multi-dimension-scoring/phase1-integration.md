# Phase 1: Integration (Task 7)

> 依赖: [Task 1](./phase1-infra-backend-sysarch.md), [Task 2-4](./phase1-backend-execution.md), [Task 5](./phase1-backend-static.md), [Task 6](./phase1-system-architecture.md)

---

## Task 7: DIMENSION_REGISTRY 更新 + 集成

**Files:**
- Modify: `benchmark/cli.py:38-43`
- Test: `tests/test_integration_composite.py`

**Backend-dev CompositeScorer 权重分配:**
| Sub-scorer | 权重 |
|------------|------|
| TestCoverageScorer | 0.40 |
| PerformanceScorer | 0.25 |
| CodeStyleScorer | 0.15 |
| RobustnessScorer | 0.10 |
| ArchitectureScorer | 0.05 |
| SecurityScorer | 0.03 |
| ExtensibilityScorer | 0.02 |

**System-architecture CompositeScorer 权重分配:**
| Sub-scorer | 权重 |
|------------|------|
| AnswerCorrectnessScorer | 0.30 |
| ReasoningCompletenessScorer | 0.25 |
| OptionAnalysisScorer | 0.20 |
| ReasoningConfidenceScorer | 0.15 |
| SubjectAdaptationScorer | 0.10 |

- [ ] **Step 1: 写集成测试**

```python
# tests/test_integration_composite.py
"""集成测试: 验证 CompositeScorer 与 DIMENSION_REGISTRY 的配合。"""
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.composite import CompositeScorer


def _make_backend_ctx(code: str, test: str = "", canonical: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=code, raw_output=code, expected="",
        task=TaskDefinition(
            task_id="test", dimension="backend-dev",
            dataset="bigcodebench", prompt="test", expected_output="",
            metadata={"test": test, "entry_point": "", "canonical_solution": canonical},
        ),
    )


def _make_sysarch_ctx(answer: str = "B", expected: str = "B", reasoning: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=answer, raw_output=answer, expected=expected,
        task=TaskDefinition(
            task_id="test", dimension="system-architecture",
            dataset="mmlu-pro", prompt="test", expected_output=expected,
            metadata={"category": "computer science", "num_options": 4},
        ),
        reasoning_content=reasoning,
    )


class TestBackendCompositeIntegration:
    def test_creates_backend_composite(self):
        """验证 backend composite scorer 可以正确实例化和执行。"""
        from benchmark.scorers.backend.test_coverage import TestCoverageScorer
        from benchmark.scorers.backend.performance import PerformanceScorer
        from benchmark.scorers.backend.code_style import CodeStyleScorer
        from benchmark.scorers.backend.robustness import RobustnessScorer
        from benchmark.scorers.backend.architecture import ArchitectureScorer
        from benchmark.scorers.backend.security import SecurityScorer
        from benchmark.scorers.backend.extensibility import ExtensibilityScorer

        scorer = CompositeScorer([
            (0.40, TestCoverageScorer()),
            (0.25, PerformanceScorer()),
            (0.15, CodeStyleScorer()),
            (0.10, RobustnessScorer()),
            (0.05, ArchitectureScorer()),
            (0.03, SecurityScorer()),
            (0.02, ExtensibilityScorer()),
        ])

        code = "def add(a, b):\n    return a + b\n"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
        result = scorer.score(_make_backend_ctx(code, test))
        assert result.score > 0
        assert "composite.weights" in result.details
        assert "composite.scores" in result.details
        assert len(result.details["composite.weights"]) == 7

    def test_backend_passed_threshold(self):
        from benchmark.scorers.backend.test_coverage import TestCoverageScorer
        from benchmark.scorers.backend.performance import PerformanceScorer
        from benchmark.scorers.backend.code_style import CodeStyleScorer
        from benchmark.scorers.backend.robustness import RobustnessScorer
        from benchmark.scorers.backend.architecture import ArchitectureScorer
        from benchmark.scorers.backend.security import SecurityScorer
        from benchmark.scorers.backend.extensibility import ExtensibilityScorer

        scorer = CompositeScorer([
            (0.40, TestCoverageScorer()),
            (0.25, PerformanceScorer()),
            (0.15, CodeStyleScorer()),
            (0.10, RobustnessScorer()),
            (0.05, ArchitectureScorer()),
            (0.03, SecurityScorer()),
            (0.02, ExtensibilityScorer()),
        ])

        code = "def add(a, b):\n    return a + b\n"
        test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
        result = scorer.score(_make_backend_ctx(code, test))
        # 测试通过 + 简洁代码 -> 应该 pass
        assert result.passed is True


class TestSysArchCompositeIntegration:
    def test_creates_sysarch_composite(self):
        from benchmark.scorers.system_architecture.answer_correctness import AnswerCorrectnessScorer
        from benchmark.scorers.system_architecture.reasoning_completeness import ReasoningCompletenessScorer
        from benchmark.scorers.system_architecture.option_analysis import OptionAnalysisScorer
        from benchmark.scorers.system_architecture.reasoning_confidence import ReasoningConfidenceScorer
        from benchmark.scorers.system_architecture.subject_adaptation import SubjectAdaptationScorer

        scorer = CompositeScorer([
            (0.30, AnswerCorrectnessScorer()),
            (0.25, ReasoningCompletenessScorer()),
            (0.20, OptionAnalysisScorer()),
            (0.15, ReasoningConfidenceScorer()),
            (0.10, SubjectAdaptationScorer()),
        ])

        reasoning = (
            "Let me analyze each option:\n"
            "A is incorrect because it violates the principle.\n"
            "C can be ruled out.\n"
            "D is wrong.\n"
            "Therefore, B is clearly the correct answer."
        )
        result = scorer.score(_make_sysarch_ctx("B", "B", reasoning))
        assert result.score > 0
        assert "composite.weights" in result.details
        assert len(result.details["composite.weights"]) == 5

    def test_sysarch_correct_answer_with_reasoning(self):
        from benchmark.scorers.system_architecture.answer_correctness import AnswerCorrectnessScorer
        from benchmark.scorers.system_architecture.reasoning_completeness import ReasoningCompletenessScorer
        from benchmark.scorers.system_architecture.option_analysis import OptionAnalysisScorer
        from benchmark.scorers.system_architecture.reasoning_confidence import ReasoningConfidenceScorer
        from benchmark.scorers.system_architecture.subject_adaptation import SubjectAdaptationScorer

        scorer = CompositeScorer([
            (0.30, AnswerCorrectnessScorer()),
            (0.25, ReasoningCompletenessScorer()),
            (0.20, OptionAnalysisScorer()),
            (0.15, ReasoningConfidenceScorer()),
            (0.10, SubjectAdaptationScorer()),
        ])

        reasoning = (
            "Let me analyze each option:\n"
            "A is incorrect because it violates the principle.\n"
            "C can be ruled out since it doesn't apply.\n"
            "D is wrong because it contradicts the premise.\n"
            "Therefore, B is clearly the correct answer."
        )
        result = scorer.score(_make_sysarch_ctx("B", "B", reasoning))
        assert result.passed is True
```

- [ ] **Step 2: 运行集成测试验证失败**

```bash
pytest tests/test_integration_composite.py -v
```
Expected: PASS (所有子 scorer 已实现后)

- [ ] **Step 3: 创建 factory 函数并更新 cli.py**

在 `benchmark/scorers/backend/__init__.py` 中添加:

```python
# benchmark/scorers/backend/__init__.py
"""Backend composite scorer factory."""

from benchmark.scorers.composite import CompositeScorer
from benchmark.scorers.backend.test_coverage import TestCoverageScorer
from benchmark.scorers.backend.performance import PerformanceScorer
from benchmark.scorers.backend.code_style import CodeStyleScorer
from benchmark.scorers.backend.robustness import RobustnessScorer
from benchmark.scorers.backend.architecture import ArchitectureScorer
from benchmark.scorers.backend.security import SecurityScorer
from benchmark.scorers.backend.extensibility import ExtensibilityScorer


def create_backend_composite() -> CompositeScorer:
    return CompositeScorer([
        (0.40, TestCoverageScorer()),
        (0.25, PerformanceScorer()),
        (0.15, CodeStyleScorer()),
        (0.10, RobustnessScorer()),
        (0.05, ArchitectureScorer()),
        (0.03, SecurityScorer()),
        (0.02, ExtensibilityScorer()),
    ])
```

在 `benchmark/scorers/system_architecture/__init__.py` 中添加:

```python
# benchmark/scorers/system_architecture/__init__.py
"""System-architecture composite scorer factory."""

from benchmark.scorers.composite import CompositeScorer
from benchmark.scorers.system_architecture.answer_correctness import AnswerCorrectnessScorer
from benchmark.scorers.system_architecture.reasoning_completeness import ReasoningCompletenessScorer
from benchmark.scorers.system_architecture.option_analysis import OptionAnalysisScorer
from benchmark.scorers.system_architecture.reasoning_confidence import ReasoningConfidenceScorer
from benchmark.scorers.system_architecture.subject_adaptation import SubjectAdaptationScorer


def create_sysarch_composite() -> CompositeScorer:
    return CompositeScorer([
        (0.30, AnswerCorrectnessScorer()),
        (0.25, ReasoningCompletenessScorer()),
        (0.20, OptionAnalysisScorer()),
        (0.15, ReasoningConfidenceScorer()),
        (0.10, SubjectAdaptationScorer()),
    ])
```

更新 `benchmark/cli.py` 中的 DIMENSION_REGISTRY:

```python
# benchmark/cli.py 顶部新增 import
from benchmark.scorers.backend import create_backend_composite
from benchmark.scorers.system_architecture import create_sysarch_composite

# 替换 DIMENSION_REGISTRY
DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (MATHAdapter, MathScorer, SingleTurnEvaluator),
    "backend-dev": (BigCodeBenchAdapter, create_backend_composite, SingleTurnEvaluator),
    "system-architecture": (MMLUProAdapter, create_sysarch_composite, SingleTurnEvaluator),
    "frontend-dev": (FrontCodeAdapter, KeywordMatchScorer, SingleTurnEvaluator),
}
```

注意: `create_backend_composite` 和 `create_sysarch_composite` 是工厂函数而非类。`_run_evaluation` 中 `scorer = scorer_cls()` 调用需要适配:

修改 `benchmark/cli.py` 第 258 行:

```python
    adapter_cls, scorer_factory, evaluator_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    scorer = scorer_factory()  # 适配: 可能是类也可能是工厂函数
    evaluator = evaluator_cls()
```

将变量名从 `scorer_cls` 改为 `scorer_factory`，调用 `scorer_factory()` 统一处理类和工厂函数。

- [ ] **Step 4: 运行全部测试**

```bash
pytest tests/ -v
```
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/cli.py benchmark/scorers/backend/__init__.py benchmark/scorers/system_architecture/__init__.py tests/test_integration_composite.py
git commit -m "feat: integrate CompositeScorer into DIMENSION_REGISTRY for backend-dev and system-architecture"
```
