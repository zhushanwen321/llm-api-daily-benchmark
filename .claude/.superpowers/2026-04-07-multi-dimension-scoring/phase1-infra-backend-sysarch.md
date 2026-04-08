# Multi-Dimension Scoring Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 binary 评分（0/100）升级为多维度细粒度评分，Phase 1 完成 CompositeScorer 基础设施 + Backend + System-Architecture 两个维度的多维评分。

**Architecture:** CompositeScorer 聚合多个 BaseScorer 子评分器，按权重计算加权总分。每个子 scorer 保持 `score(ctx) -> ScoreResult` 同步接口。CompositeScorer.ascore() 用 asyncio.gather 并行执行子 scorer。工具不可用时返回默认 100 分（不惩罚）。

**Tech Stack:** Python 3.12+, pytest, asyncio, pylint, flake8, bandit, radon, semgrep, timeit

**子文档索引:**
- [Task 2-4: Backend Adapter + TestCoverage + Performance](./phase1-backend-execution.md)
- [Task 5: Backend 静态分析 Scorers](./phase1-backend-static.md)
- [Task 6: System-Architecture Scorers](./phase1-system-architecture.md)
- [Task 7: 集成](./phase1-integration.md)

---

## 文件结构总览

```
benchmark/
  scorers/
    composite.py                    # Task 1: CompositeScorer
    backend/
      __init__.py
      test_coverage.py              # Task 3: TestCoverageScorer
      performance.py                # Task 4: PerformanceScorer
      code_style.py                 # Task 5: CodeStyleScorer
      robustness.py                 # Task 5: RobustnessScorer
      architecture.py               # Task 5: ArchitectureScorer
      security.py                   # Task 5: SecurityScorer
      extensibility.py              # Task 5: ExtensibilityScorer
    system_architecture/
      __init__.py
      answer_correctness.py         # Task 6: AnswerCorrectnessScorer
      reasoning_completeness.py     # Task 6: ReasoningCompletenessScorer
      option_analysis.py            # Task 6: OptionAnalysisScorer
      reasoning_confidence.py       # Task 6: ReasoningConfidenceScorer
      subject_adaptation.py         # Task 6: SubjectAdaptationScorer
  adapters/
    bigcodebench_adapter.py         # Task 2: 添加 canonical_solution
  cli.py                            # Task 7: DIMENSION_REGISTRY 更新
tests/
  test_composite_scorer.py          # Task 1
  test_backend_test_coverage.py     # Task 3
  test_backend_performance.py       # Task 4
  test_backend_code_style.py        # Task 5
  test_backend_robustness.py        # Task 5
  test_backend_architecture.py      # Task 5
  test_backend_security.py          # Task 5
  test_backend_extensibility.py     # Task 5
  test_system_arch_correctness.py   # Task 6
  test_system_arch_reasoning.py     # Task 6
  test_system_arch_option.py        # Task 6
  test_system_arch_confidence.py    # Task 6
  test_system_arch_subject.py       # Task 6
```

---

## Task 1: CompositeScorer 基础设施

**Files:**
- Create: `benchmark/scorers/composite.py`
- Test: `tests/test_composite_scorer.py`

**设计要点:**
- 构造时接收 `list[tuple[float, BaseScorer]]`，权重之和校验为 1.0
- `score()` 同步串行执行子 scorer，`ascore()` 用 asyncio.gather 并行
- 子 scorer 异常时该维度默认 100 分
- `reasoning_content` 为空时，依赖推理的 scorer 返回默认 100 分（通过 scorer 自身处理，CompositeScorer 不感知）
- `passed = score >= 60`
- `details` 包含 `composite.weights`（权重映射）和 `composite.scores`（各维度得分映射）

- [ ] **Step 1: 写失败测试 — CompositeScorer 基本加权计算**

```python
# tests/test_composite_scorer.py
import asyncio

import pytest

from benchmark.models.schemas import ScoreResult, ScoringContext, TaskDefinition
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.composite import CompositeScorer


class _FixedScorer(BaseScorer):
    """固定分数的测试用 scorer."""

    def __init__(self, score_val: float, name: str = "fixed"):
        self._score = score_val
        self._name = name

    def score(self, ctx: ScoringContext) -> ScoreResult:
        return ScoreResult(score=self._score, passed=self._score >= 60)

    def get_metric_name(self) -> str:
        return self._name


def _make_ctx() -> ScoringContext:
    return ScoringContext(
        model_answer="test",
        raw_output="test",
        expected="",
        task=TaskDefinition(
            task_id="t1", dimension="test", dataset="test",
            prompt="test", expected_output="",
        ),
    )


class TestCompositeScorerInit:
    def test_valid_weights(self):
        scorer = CompositeScorer([
            (0.4, _FixedScorer(100)),
            (0.6, _FixedScorer(50)),
        ])
        assert scorer.get_metric_name() == "composite"

    def test_weights_not_sum_to_one(self):
        with pytest.raises(ValueError, match="权重之和必须等于 1.0"):
            CompositeScorer([
                (0.5, _FixedScorer(100)),
                (0.6, _FixedScorer(50)),
            ])

    def test_empty_scorers(self):
        with pytest.raises(ValueError, match="至少需要一个子评分器"):
            CompositeScorer([])

    def test_single_scorer_weight_one(self):
        scorer = CompositeScorer([(1.0, _FixedScorer(80))])
        result = scorer.score(_make_ctx())
        assert result.score == 80.0
        assert result.passed is True


class TestCompositeScorerScore:
    def test_weighted_score(self):
        scorer = CompositeScorer([
            (0.4, _FixedScorer(100, "a")),
            (0.6, _FixedScorer(50, "b")),
        ])
        result = scorer.score(_make_ctx())
        assert result.score == pytest.approx(70.0)  # 0.4*100 + 0.6*50

    def test_passed_threshold_60(self):
        scorer = CompositeScorer([
            (1.0, _FixedScorer(59, "x")),
        ])
        result = scorer.score(_make_ctx())
        assert result.passed is False

        scorer2 = CompositeScorer([(1.0, _FixedScorer(60, "x"))])
        result2 = scorer2.score(_make_ctx())
        assert result2.passed is True

    def test_details_contains_weights_and_scores(self):
        scorer = CompositeScorer([
            (0.3, _FixedScorer(100, "a")),
            (0.7, _FixedScorer(0, "b")),
        ])
        result = scorer.score(_make_ctx())
        assert result.details["composite.weights"] == {"a": 0.3, "b": 0.7}
        assert result.details["composite.scores"] == {"a": 100.0, "b": 0.0}

    def test_sub_scorer_exception_defaults_to_100(self):
        class _BrokenScorer(BaseScorer):
            def score(self, ctx: ScoringContext) -> ScoreResult:
                raise RuntimeError("boom")
            def get_metric_name(self) -> str:
                return "broken"

        scorer = CompositeScorer([
            (0.5, _BrokenScorer()),
            (0.5, _FixedScorer(0, "ok")),
        ])
        result = scorer.score(_make_ctx())
        # broken 默认 100, ok=0 -> 0.5*100 + 0.5*0 = 50
        assert result.score == pytest.approx(50.0)
        assert result.details["composite.scores"]["broken"] == 100.0
        assert result.details["composite.errors"]["broken"] == "RuntimeError: boom"


class TestCompositeScorerAsync:
    def test_ascore_parallel_execution(self):
        import time

        class _SlowScorer(BaseScorer):
            def __init__(self, score_val: float, name: str, delay: float):
                self._score = score_val
                self._name = name
                self._delay = delay

            def score(self, ctx: ScoringContext) -> ScoreResult:
                time.sleep(self._delay)
                return ScoreResult(score=self._score, passed=self._score >= 60)

            def get_metric_name(self) -> str:
                return self._name

        scorer = CompositeScorer([
            (0.5, _SlowScorer(100, "a", 0.2)),
            (0.5, _SlowScorer(0, "b", 0.2)),
        ])
        t0 = time.monotonic()
        result = asyncio.run(scorer.ascore(_make_ctx()))
        elapsed = time.monotonic() - t0
        # 串行需要 0.4s+，并行应 < 0.4s
        assert elapsed < 0.35
        assert result.score == pytest.approx(50.0)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_composite_scorer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.scorers.composite'`

- [ ] **Step 3: 实现 CompositeScorer**

```python
# benchmark/scorers/composite.py
"""组合评分器。按权重聚合多个子评分器的分数。"""

from __future__ import annotations

import asyncio
import logging
import time

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class CompositeScorer(BaseScorer):
    """按权重聚合多个子评分器，计算加权总分。

    子 scorer 异常时该维度默认 100 分。
    """

    def __init__(self, scorers: list[tuple[float, BaseScorer]]) -> None:
        if not scorers:
            raise ValueError("至少需要一个子评分器")
        total_weight = sum(w for w, _ in scorers)
        if not (abs(total_weight - 1.0) < 1e-9):
            raise ValueError(f"权重之和必须等于 1.0，当前为 {total_weight}")
        self._scorers = scorers

    def score(self, ctx: ScoringContext) -> ScoreResult:
        weights: dict[str, float] = {}
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}

        weighted_sum = 0.0
        for weight, scorer in self._scorers:
            name = scorer.get_metric_name()
            weights[name] = weight
            try:
                result = scorer.score(ctx)
                scores[name] = result.score
            except Exception as exc:
                logger.warning("子评分器 %s 异常，默认 100 分: %s", name, exc)
                scores[name] = 100.0
                errors[name] = f"{type(exc).__name__}: {exc}"
            weighted_sum += weight * scores[name]

        score = round(weighted_sum, 2)
        details: dict = {
            "composite.weights": weights,
            "composite.scores": scores,
        }
        if errors:
            details["composite.errors"] = errors

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"加权总分={score}",
        )

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """并行执行所有子 scorer。"""
        weights: dict[str, float] = {}
        errors: dict[str, str] = {}

        async def _run_one(weight: float, scorer: BaseScorer) -> tuple[str, float]:
            name = scorer.get_metric_name()
            weights[name] = weight
            try:
                result = await scorer.ascore(ctx)
                return name, result.score
            except Exception as exc:
                logger.warning("子评分器 %s 异常，默认 100 分: %s", name, exc)
                errors[name] = f"{type(exc).__name__}: {exc}"
                return name, 100.0

        coros = [_run_one(w, s) for w, s in self._scorers]
        results = await asyncio.gather(*coros)

        scores = dict(results)
        weighted_sum = sum(weights[name] * score for name, score in scores.items())
        score = round(weighted_sum, 2)

        details: dict = {
            "composite.weights": weights,
            "composite.scores": scores,
        }
        if errors:
            details["composite.errors"] = errors

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"加权总分={score}",
        )

    def get_metric_name(self) -> str:
        return "composite"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_composite_scorer.py -v
```
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/composite.py tests/test_composite_scorer.py
git commit -m "feat(scorer): add CompositeScorer for weighted multi-dimension scoring"
```

---

## 后续 Tasks

- [Task 2-4: Backend Adapter + TestCoverage + Performance](./phase1-backend-execution.md)
- [Task 5: Backend 静态分析 Scorers](./phase1-backend-static.md)
- [Task 6: System-Architecture Scorers](./phase1-system-architecture.md)
- [Task 7: 集成](./phase1-integration.md)
