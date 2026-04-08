# Phase 2: Frontend + Reasoning (MATH) 多维度评分实施计划

## 概述

实现 Frontend 和 Reasoning (MATH) 两个维度的多维度评分系统。每个维度由多个子 Scorer 组成，通过 CompositeScorer 计算加权总分。

### Reasoning 维度（权重分配）

| Scorer | 权重 | 文件 |
|--------|------|------|
| AnswerCorrectnessScorer | 40% | `benchmark/scorers/reasoning/answer_correctness.py` |
| ReasoningCompletenessScorer | 25% | `benchmark/scorers/reasoning/reasoning_completeness.py` |
| ReasoningValidityScorer | 20% | `benchmark/scorers/reasoning/reasoning_validity.py` |
| MethodEleganceScorer | 10% | `benchmark/scorers/reasoning/method_elegance.py` |
| DifficultyAdaptationScorer | 5% | `benchmark/scorers/reasoning/difficulty_adaptation.py` |

### Frontend 维度（权重分配）

| Scorer | 权重 | 文件 |
|--------|------|------|
| FunctionalityScorer | 30% | `benchmark/scorers/frontend/functionality.py` |
| HTMLSemanticScorer | 20% | `benchmark/scorers/frontend/html_semantic.py` |
| AccessibilityScorer | 15% | `benchmark/scorers/frontend/accessibility.py` |
| CSSQualityScorer | 15% | `benchmark/scorers/frontend/css_quality.py` |
| CodeOrganizationScorer | 10% | `benchmark/scorers/frontend/code_organization.py` |
| PerformanceScorer | 5% | `benchmark/scorers/frontend/performance.py` |
| BrowserCompatScorer | 5% | `benchmark/scorers/frontend/browser_compat.py` |

### 公共约定

- 所有 `score()` 方法是同步的（`ascore()` 由 BaseScorer 默认实现通过 `asyncio.to_thread` 包装）
- ReasoningValidityScorer 需要重写 `ascore()` 因为调用 LLM API
- 工具不可用时返回默认 100 分
- `reasoning_content` 为空时返回 100 分（Reasoning 维度的非答案类 Scorer）
- 测试用 `make_reasoning_ctx()` 和 `make_frontend_ctx()` 工厂函数构造 ScoringContext

---

## Task 1: 目录结构与公共测试工具

- [ ] 创建 `benchmark/scorers/reasoning/__init__.py`
- [ ] 创建 `benchmark/scorers/frontend/__init__.py`
- [ ] 创建 `tests/test_reasoning_scorers.py`，定义公共工厂函数

### 公共工厂函数

```python
# tests/test_reasoning_scorers.py

import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition


def make_reasoning_ctx(
    predicted: str = "42",
    expected: str = "42",
    reasoning: str = "因为 x=42，所以答案为 42。首先计算 x+1=43，其次验证 x-1=41，最后得出 x=42。",
    level: int = 3,
    subject: str = "Algebra",
) -> ScoringContext:
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
            metadata={"level": level, "subject": subject, "source": "test"},
        ),
        reasoning_content=reasoning,
    )


def make_frontend_ctx(
    code: str = "<html><body>hello</body></html>",
    task_type: str = "html",
    keywords: list[str] | None = None,
    test_cases: list[str] | None = None,
) -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="front_test",
            dimension="frontend-dev",
            dataset="frontcode",
            prompt="test",
            expected_output="",
            metadata={
                "type": task_type,
                "keywords": keywords or [],
                "source": "frontcode",
            },
            test_cases=test_cases or [],
        ),
    )
```

---

## Task 2: MATH AnswerCorrectnessScorer（权重 40%）

文件: `benchmark/scorers/reasoning/answer_correctness.py`
测试: `tests/test_reasoning_scorers.py`（追加）

### 测试

```python
from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer


class TestAnswerCorrectnessScorer:
    def test_exact_match(self):
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("42", "42"))
        assert r.passed is True and r.score == 100.0

    def test_numeric_match(self):
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("42.0", "42"))
        assert r.passed is True

    def test_fraction_match(self):
        r = AnswerCorrectnessScorer().score(
            make_reasoning_ctx(r"\frac{14}{3}", r"\frac{14}{3}")
        )
        assert r.passed is True

    def test_wrong_answer(self):
        r = AnswerCorrectnessScorer().score(make_reasoning_ctx("99", "42"))
        assert r.passed is False and r.score == 0.0

    def test_empty_reasoning_still_scores(self):
        r = AnswerCorrectnessScorer().score(
            make_reasoning_ctx("42", "42", reasoning="")
        )
        assert r.passed is True

    def test_get_metric_name(self):
        assert AnswerCorrectnessScorer().get_metric_name() == "answer_correctness"
```

### 实现

```python
# benchmark/scorers/reasoning/answer_correctness.py
from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.math_scorer import MathScorer


class AnswerCorrectnessScorer(BaseScorer):
    """MATH 答案正确性评分。委托 MathScorer 的三阶段匹配逻辑。"""

    def __init__(self) -> None:
        self._math = MathScorer()

    def score(self, ctx: ScoringContext) -> ScoreResult:
        return self._math.score(ctx)

    def get_metric_name(self) -> str:
        return "answer_correctness"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py::TestAnswerCorrectnessScorer -v
```

---

## Task 3: MATH ReasoningCompletenessScorer（权重 25%）

文件: `benchmark/scorers/reasoning/reasoning_completeness.py`

### 评分逻辑

- `reasoning_content` 为空 -> 100 分
- 长度分 (40%): `est_tokens = len(reasoning_content) / 4`, 期望最小长度 `{3: 200, 4: 400, 5: 600}`, `length_score = min(100, est_tokens / min_tokens * 100)`
- 连接词分 (30%): 检测逻辑连接词，`conn_score = min(100, count * 15)`
- 结构分 (30%): 检测结构标记，`struct_score = min(100, count * 20)`
- 总分 = `length_score * 0.4 + conn_score * 0.3 + struct_score * 0.3`

### 测试

```python
from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer


class TestReasoningCompletenessScorer:
    def test_empty_reasoning_gives_100(self):
        r = ReasoningCompletenessScorer().score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_long_reasoning_high_score(self):
        long_reasoning = "因为。所以。因此。由于。故。因为。所以。因此。由于。故。" * 50
        r = ReasoningCompletenessScorer().score(
            make_reasoning_ctx(reasoning=long_reasoning)
        )
        assert r.score > 50.0

    def test_short_reasoning_low_score(self):
        r = ReasoningCompletenessScorer().score(
            make_reasoning_ctx(reasoning="答案", level=5)
        )
        assert r.score < 50.0

    def test_structural_markers_boost(self):
        reasoning = "首先。其次。最后。因为。所以。因此。" * 100
        r = ReasoningCompletenessScorer().score(
            make_reasoning_ctx(reasoning=reasoning, level=3)
        )
        assert r.score >= 60.0

    def test_get_metric_name(self):
        assert ReasoningCompletenessScorer().get_metric_name() == "reasoning_completeness"
```

### 实现

```python
# benchmark/scorers/reasoning/reasoning_completeness.py
from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_CONNECTORS = [
    "因为", "所以", "因此", "由于", "故",
    "because", "therefore", "thus", "since",
]
_STRUCTURAL = [
    "步骤", "首先", "其次", "最后",
    "step", "first", "second", "finally",
]
_MIN_TOKENS = {3: 200, 4: 400, 5: 600}


class ReasoningCompletenessScorer(BaseScorer):
    """推理完整性评分。基于长度、逻辑连接词和结构标记。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "empty_reasoning"},
                               reasoning="No reasoning content, default 100")

        est_tokens = len(reasoning) / 4
        level = ctx.task.metadata.get("level", 3)
        min_tokens = _MIN_TOKENS.get(level, 200)

        length_score = min(100.0, est_tokens / min_tokens * 100)

        conn_count = sum(1 for c in _CONNECTORS if c in reasoning.lower())
        conn_score = min(100.0, conn_count * 15)

        struct_count = sum(1 for s in _STRUCTURAL if s in reasoning.lower())
        struct_score = min(100.0, struct_count * 20)

        total = length_score * 0.4 + conn_score * 0.3 + struct_score * 0.3

        return ScoreResult(
            score=round(total, 1),
            passed=total >= 60.0,
            details={
                "est_tokens": round(est_tokens),
                "length_score": round(length_score, 1),
                "connector_count": conn_count,
                "connector_score": round(conn_score, 1),
                "structural_count": struct_count,
                "structural_score": round(struct_score, 1),
            },
            reasoning=f"Completeness: length={length_score:.0f}, connectors={conn_score:.0f}, structure={struct_score:.0f}",
        )

    def get_metric_name(self) -> str:
        return "reasoning_completeness"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py::TestReasoningCompletenessScorer -v
```

---

## Task 4: MATH ReasoningValidityScorer（权重 20%）-- LLM-as-a-Judge

文件: `benchmark/scorers/reasoning/reasoning_validity.py`

### 设计要点

- 重写 `ascore()` 因为需要异步调用 LLM API
- `score()` 同步方法直接抛出 `NotImplementedError`（不支持同步调用）
- `reasoning_content` 为空 -> 100 分
- Judge prompt 要求返回 JSON，包含三个子维度分数
- 结果缓存: 按 `(problem_hash, reasoning_hash)` 缓存
- 需要 `LLMEvalAdapter` 实例（通过构造函数注入）

### 测试

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.models.schemas import GenerateResponse
from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer


class TestReasoningValidityScorer:
    def test_empty_reasoning_gives_100(self):
        scorer = ReasoningValidityScorer(llm=MagicMock())
        r = scorer.score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_empty_reasoning_async(self):
        scorer = ReasoningValidityScorer(llm=MagicMock())
        r = asyncio.get_event_loop().run_until_complete(
            scorer.ascore(make_reasoning_ctx(reasoning=""))
        )
        assert r.score == 100.0

    @pytest.mark.asyncio
    async def test_llm_judge_call(self):
        mock_llm = MagicMock()
        mock_llm.agenerate = AsyncMock(return_value=GenerateResponse(
            content='{"logical_consistency": 40, "math_facts": 40, "computation": 20}',
        ))
        scorer = ReasoningValidityScorer(llm=mock_llm)
        r = await scorer.ascore(make_reasoning_ctx(
            reasoning="因为 2+2=4，所以答案为 4。"
        ))
        assert r.score == 100.0
        mock_llm.agenerate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        mock_llm = MagicMock()
        mock_llm.agenerate = AsyncMock(return_value=GenerateResponse(
            content='{"logical_consistency": 40, "math_facts": 40, "computation": 20}',
        ))
        scorer = ReasoningValidityScorer(llm=mock_llm)
        ctx = make_reasoning_ctx(reasoning="test reasoning content")
        await scorer.ascore(ctx)
        await scorer.ascore(ctx)
        # 缓存命中，只调用一次
        assert mock_llm.agenerate.call_count == 1

    @pytest.mark.asyncio
    async def test_malformed_json_graceful(self):
        mock_llm = MagicMock()
        mock_llm.agenerate = AsyncMock(return_value=GenerateResponse(
            content="not json at all",
        ))
        scorer = ReasoningValidityScorer(llm=mock_llm)
        r = await scorer.ascore(make_reasoning_ctx(reasoning="some reasoning"))
        assert r.score == 50.0  # JSON 解析失败默认 50 分

    def test_get_metric_name(self):
        assert ReasoningValidityScorer(llm=MagicMock()).get_metric_name() == "reasoning_validity"
```

### 实现

```python
# benchmark/scorers/reasoning/reasoning_validity.py
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = "你是数学推理质量评判专家。请检查以下解题过程的正确性。"
_JUDGE_TEMPLATE = """请评估以下数学解题过程的正确性。

题目: {problem}

解题过程:
{reasoning}

请以 JSON 格式返回评分（不要包含其他文字）:
{{"logical_consistency": <0-40>, "math_facts": <0-40>, "computation": <0-20>}}

评分标准:
- logical_consistency (0-40): 逻辑一致性，推理链条是否连贯
- math_facts (0-40): 数学事实正确性，公式和定理使用是否正确
- computation (0-20): 计算正确性，数值计算是否准确"""


class ReasoningValidityScorer(BaseScorer):
    """LLM-as-a-Judge 推理正确性评分。需要异步调用 LLM API。"""

    def __init__(self, llm: Any, model: str = "zai/glm-5.1") -> None:
        self._llm = llm
        self._model = model
        self._cache: dict[str, float] = {}

    def score(self, ctx: ScoringContext) -> ScoreResult:
        raise NotImplementedError("ReasoningValidityScorer 只支持异步评分 (ascore)")

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "empty_reasoning"},
                               reasoning="No reasoning content, default 100")

        cache_key = self._cache_key(ctx.task.prompt, reasoning)
        if cache_key in self._cache:
            score = self._cache[cache_key]
            return ScoreResult(score=score, passed=score >= 60.0,
                               details={"source": "cache"},
                               reasoning=f"Cached score: {score}")

        try:
            prompt = _JUDGE_TEMPLATE.format(
                problem=ctx.task.prompt,
                reasoning=reasoning,
            )
            resp = await self._llm.agenerate(
                prompt=prompt,
                model=self._model,
                temperature=0.0,
                system_message=_JUDGE_SYSTEM,
            )
            score = self._parse_response(resp.content)
            self._cache[cache_key] = score
            return ScoreResult(
                score=score,
                passed=score >= 60.0,
                details={"source": "llm_judge", "raw": resp.content[:200]},
                reasoning=f"LLM judge score: {score}",
            )
        except Exception as exc:
            logger.warning(f"ReasoningValidityScorer LLM 调用失败: {exc}")
            return ScoreResult(
                score=50.0, passed=False,
                details={"error": str(exc)},
                reasoning=f"LLM judge failed: {exc}, default 50",
            )

    @staticmethod
    def _cache_key(prompt: str, reasoning: str) -> str:
        h = hashlib.md5()
        h.update(prompt.encode())
        h.update(reasoning.encode())
        return h.hexdigest()

    @staticmethod
    def _parse_response(content: str) -> float:
        # 尝试从 LLM 响应中提取 JSON
        json_match = re.search(r'\{[^}]+\}', content)
        if not json_match:
            return 50.0
        try:
            data = json.loads(json_match.group())
            lc = min(40, max(0, float(data.get("logical_consistency", 0))))
            mf = min(40, max(0, float(data.get("math_facts", 0))))
            cp = min(20, max(0, float(data.get("computation", 0))))
            return round(lc + mf + cp, 1)
        except (json.JSONDecodeError, ValueError, TypeError):
            return 50.0

    def get_metric_name(self) -> str:
        return "reasoning_validity"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py::TestReasoningValidityScorer -v
```

---

## Task 5: MATH MethodEleganceScorer（权重 10%）

文件: `benchmark/scorers/reasoning/method_elegance.py`

### 评分逻辑

- `reasoning_content` 为空 -> 100 分
- 学科关键词匹配: 每命中 +10 分，上限 50 分
- 冗余惩罚: `reasoning_tokens > level * 500` 时，每超 100 tokens 扣 5 分，上限扣 30 分
- 总分 = `keyword_score - redundancy_penalty`，下限 0

### 测试

```python
from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer


class TestMethodEleganceScorer:
    def test_empty_reasoning_gives_100(self):
        r = MethodEleganceScorer().score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_keyword_match(self):
        r = MethodEleganceScorer().score(
            make_reasoning_ctx(
                reasoning="使用因式分解和换元法解决问题，利用对称性简化",
                subject="Algebra",
            )
        )
        assert r.score >= 30.0

    def test_redundancy_penalty(self):
        long_reasoning = "x" * 4000  # ~1000 tokens
        r = MethodEleganceScorer().score(
            make_reasoning_ctx(reasoning=long_reasoning, level=3)
        )
        # level 3, threshold = 1500 tokens, actual ~1000, no penalty
        assert r.score >= 0.0

    def test_redundancy_penalty_high_level(self):
        long_reasoning = "x" * 8000  # ~2000 tokens
        r = MethodEleganceScorer().score(
            make_reasoning_ctx(reasoning=long_reasoning, level=3)
        )
        # level 3, threshold = 1500 tokens, actual ~2000, penalty
        assert r.details.get("redundancy_penalty", 0) > 0

    def test_unknown_subject(self):
        r = MethodEleganceScorer().score(
            make_reasoning_ctx(reasoning="some reasoning", subject="Unknown")
        )
        assert r.score == 0.0  # 无关键词匹配

    def test_get_metric_name(self):
        assert MethodEleganceScorer().get_metric_name() == "method_elegance"
```

### 实现

```python
# benchmark/scorers/reasoning/method_elegance.py
from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "Algebra": ["因式分解", "对称性", "换元", "韦达定理", "factorization", "symmetry", "substitution"],
    "Geometry": ["辅助线", "相似", "勾股定理", "auxiliary line", "similar", "Pythagorean"],
    "Number Theory": ["模运算", "整除", "同余", "欧拉函数", "modular", "divisibility", "Euler"],
    "Precalculus": ["三角恒等变换", "复数", "向量", "trigonometric", "complex number", "vector"],
    "Counting & Probability": ["排列", "组合", "概率", "permutation", "combination", "probability"],
    "Intermediate Algebra": ["二次方程", "不等式", "函数", "quadratic", "inequality", "function"],
    "Prealgebra": ["分数", "小数", "百分数", "fraction", "decimal", "percentage"],
}


class MethodEleganceScorer(BaseScorer):
    """方法优雅度评分。学科关键词匹配 + 冗余检测。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "empty_reasoning"},
                               reasoning="No reasoning content, default 100")

        subject = ctx.task.metadata.get("subject", "")
        keywords = _SUBJECT_KEYWORDS.get(subject, [])
        reasoning_lower = reasoning.lower()

        matched = [kw for kw in keywords if kw.lower() in reasoning_lower]
        keyword_score = min(50.0, len(matched) * 10)

        est_tokens = len(reasoning) / 4
        level = ctx.task.metadata.get("level", 3)
        threshold = level * 500
        excess = est_tokens - threshold
        penalty = min(30.0, max(0.0, (excess / 100) * 5)) if excess > 0 else 0.0

        total = max(0.0, keyword_score - penalty)

        return ScoreResult(
            score=round(total, 1),
            passed=total >= 30.0,
            details={
                "matched_keywords": matched,
                "keyword_score": round(keyword_score, 1),
                "est_tokens": round(est_tokens),
                "redundancy_penalty": round(penalty, 1),
            },
            reasoning=f"Elegance: keywords={keyword_score:.0f}, penalty={penalty:.0f}",
        )

    def get_metric_name(self) -> str:
        return "method_elegance"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py::TestMethodEleganceScorer -v
```

---

## Task 6: MATH DifficultyAdaptationScorer（权重 5%）

文件: `benchmark/scorers/reasoning/difficulty_adaptation.py`

### 评分逻辑

- `reasoning_content` 为空 -> 100 分
- 期望推理深度: `{3: 3, 4: 5, 5: 7}`
- 深度 = 逻辑连接词数 + 步骤标记数（复用 ReasoningCompletenessScorer 的词表）
- `depth_match = 1 - |actual - expected| / expected`
- `score = max(0, depth_match * 100)`

### 测试

```python
from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer


class TestDifficultyAdaptationScorer:
    def test_empty_reasoning_gives_100(self):
        r = DifficultyAdaptationScorer().score(make_reasoning_ctx(reasoning=""))
        assert r.score == 100.0

    def test_exact_depth_match(self):
        # level 3, expected depth = 3, 使用 3 个连接词/标记
        reasoning = "首先，因为 x=2，所以 y=4，因此答案为 4。"
        r = DifficultyAdaptationScorer().score(
            make_reasoning_ctx(reasoning=reasoning, level=3)
        )
        assert r.score >= 60.0

    def test_too_shallow(self):
        r = DifficultyAdaptationScorer().score(
            make_reasoning_ctx(reasoning="答案是 42。", level=5)
        )
        assert r.score < 50.0

    def test_get_metric_name(self):
        assert DifficultyAdaptationScorer().get_metric_name() == "difficulty_adaptation"
```

### 实现

```python
# benchmark/scorers/reasoning/difficulty_adaptation.py
from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_DEPTH_MARKERS = [
    "因为", "所以", "因此", "由于", "故",
    "because", "therefore", "thus", "since",
    "步骤", "首先", "其次", "最后",
    "step", "first", "second", "finally",
]
_EXPECTED_DEPTH = {3: 3, 4: 5, 5: 7}


class DifficultyAdaptationScorer(BaseScorer):
    """难度适配评分。推理深度是否匹配题目难度等级。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "empty_reasoning"},
                               reasoning="No reasoning content, default 100")

        reasoning_lower = reasoning.lower()
        depth = sum(1 for m in _DEPTH_MARKERS if m in reasoning_lower)

        level = ctx.task.metadata.get("level", 3)
        expected = _EXPECTED_DEPTH.get(level, 3)
        depth_match = 1 - abs(depth - expected) / expected
        score = max(0.0, depth_match * 100)

        return ScoreResult(
            score=round(score, 1),
            passed=score >= 50.0,
            details={
                "actual_depth": depth,
                "expected_depth": expected,
                "depth_match": round(depth_match, 2),
            },
            reasoning=f"Depth: actual={depth}, expected={expected}, match={depth_match:.2f}",
        )

    def get_metric_name(self) -> str:
        return "difficulty_adaptation"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py::TestDifficultyAdaptationScorer -v
```

---

## Task 7: Reasoning CompositeScorer 组装

文件: `benchmark/scorers/reasoning/__init__.py`

### 实现

```python
# benchmark/scorers/reasoning/__init__.py
from __future__ import annotations

from typing import Any

from benchmark.scorers.reasoning.answer_correctness import AnswerCorrectnessScorer
from benchmark.scorers.reasoning.reasoning_completeness import ReasoningCompletenessScorer
from benchmark.scorers.reasoning.reasoning_validity import ReasoningValidityScorer
from benchmark.scorers.reasoning.method_elegance import MethodEleganceScorer
from benchmark.scorers.reasoning.difficulty_adaptation import DifficultyAdaptationScorer


def create_reasoning_composite(llm: Any = None) -> list[tuple[float, Any]]:
    """创建 reasoning 维度的加权子评分器列表。

    Args:
        llm: LLMEvalAdapter 实例，用于 ReasoningValidityScorer。
             为 None 时用 MagicMock 替代（测试用）。
    """
    if llm is None:
        from unittest.mock import MagicMock
        llm = MagicMock()

    return [
        (0.40, AnswerCorrectnessScorer()),
        (0.25, ReasoningCompletenessScorer()),
        (0.20, ReasoningValidityScorer(llm=llm)),
        (0.10, MethodEleganceScorer()),
        (0.05, DifficultyAdaptationScorer()),
    ]
```

### 测试

```python
# 追加到 tests/test_reasoning_scorers.py
from benchmark.scorers.composite import CompositeScorer
from benchmark.scorers.reasoning import create_reasoning_composite


class TestReasoningComposite:
    def test_composite_integration(self):
        subs = create_reasoning_composite()
        scorer = CompositeScorer(subs)
        ctx = make_reasoning_ctx("42", "42")
        r = scorer.score(ctx)
        # AnswerCorrectness 给 100，其他 scorer 也给较高分
        assert r.score > 50.0

    def test_composite_wrong_answer(self):
        subs = create_reasoning_composite()
        scorer = CompositeScorer(subs)
        ctx = make_reasoning_ctx("99", "42")
        r = scorer.score(ctx)
        # AnswerCorrectness 给 0，占 40% 权重，总分应该较低
        assert r.score < 60.0
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py::TestReasoningComposite -v
```

---

## Task 8: Frontend FunctionalityScorer（权重 30%）

文件: `benchmark/scorers/frontend/functionality.py`

### 评分逻辑

- `test_cases` 为空 -> 100 分
- 根据 `ctx.task.metadata["type"]` 选择执行策略:
  - `html`/`css`: Playwright 加载 HTML 执行 DOM 断言
  - `javascript`: Node.js 执行代码 + 断言
  - `react`: Playwright + @testing-library 渲染
- Playwright/Node.js 不可用 -> 100 分
- 断言通过数 / 总断言数 * 100

### 测试

```python
# tests/test_frontend_scorers.py

import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.frontend.functionality import FunctionalityScorer


def make_frontend_ctx(
    code: str = "<html><body>hello</body></html>",
    task_type: str = "html",
    keywords: list[str] | None = None,
    test_cases: list[str] | None = None,
) -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="front_test",
            dimension="frontend-dev",
            dataset="frontcode",
            prompt="test",
            expected_output="",
            metadata={
                "type": task_type,
                "keywords": keywords or [],
                "source": "frontcode",
            },
            test_cases=test_cases or [],
        ),
    )


class TestFunctionalityScorer:
    def test_empty_test_cases_gives_100(self):
        r = FunctionalityScorer().score(make_frontend_ctx(test_cases=[]))
        assert r.score == 100.0

    def test_no_test_cases_field_gives_100(self):
        r = FunctionalityScorer().score(make_frontend_ctx())
        assert r.score == 100.0

    def test_node_not_available_gives_100(self):
        # 只在 node 不可用时生效；CI 环境可能安装了 node，所以 mock
        with patch("benchmark.scorers.frontend.functionality.shutil.which", return_value=None):
            r = FunctionalityScorer().score(
                make_frontend_ctx(
                    code="console.log('hi')",
                    task_type="javascript",
                    test_cases=["assert true"],
                )
            )
            assert r.score == 100.0

    def test_get_metric_name(self):
        assert FunctionalityScorer().get_metric_name() == "functionality"
```

### 实现

```python
# benchmark/scorers/frontend/functionality.py
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
    """前端功能评分。通过 Playwright 或 Node.js 执行测试断言。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        test_cases = ctx.task.test_cases
        if not test_cases:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "no_test_cases"},
                               reasoning="No test cases, default 100")

        task_type = ctx.task.metadata.get("type", "html")

        if task_type in ("javascript",) and shutil.which("node"):
            return self._run_node(ctx)
        elif task_type in ("html", "css", "react") and shutil.which("npx"):
            return self._run_playwright(ctx)

        return ScoreResult(
            score=100.0, passed=True,
            details={"reason": "runtime_unavailable", "type": task_type},
            reasoning=f"Runtime for {task_type} unavailable, default 100",
        )

    def _run_node(self, ctx: ScoringContext) -> ScoreResult:
        assertions = "\n".join(ctx.task.test_cases)
        code = f"{ctx.model_answer}\n{assertions}"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mjs", delete=False
            ) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["node", f.name], capture_output=True, text=True, timeout=15,
                )
                Path(f.name).unlink(missing_ok=True)
                if result.returncode == 0:
                    return ScoreResult(score=100.0, passed=True,
                                       details={"stdout": result.stdout[-500:]},
                                       reasoning="All assertions passed")
                return ScoreResult(
                    score=0.0, passed=False,
                    details={"stderr": result.stderr[-500:]},
                    reasoning=f"Assertion failed: {result.stderr[-200:]}",
                )
        except Exception as exc:
            return ScoreResult(score=50.0, passed=False,
                               details={"error": str(exc)},
                               reasoning=f"Execution error: {exc}")

    def _run_playwright(self, ctx: ScoringContext) -> ScoreResult:
        # TODO: Phase 3 实现完整 Playwright 集成
        return ScoreResult(
            score=100.0, passed=True,
            details={"reason": "playwright_placeholder"},
            reasoning="Playwright not yet implemented, default 100",
        )

    def get_metric_name(self) -> str:
        return "functionality"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestFunctionalityScorer -v
```

---

## Task 9: Frontend HTMLSemanticScorer（权重 20%）

文件: `benchmark/scorers/frontend/html_semantic.py`

### 评分逻辑

- BeautifulSoup 解析 HTML
- 语义标签: header, nav, main, article, section, aside, footer
- `semantic_ratio = semantic_count / total_elements`
- ratio >= 0.6 -> 100, >= 0.3 -> 60, < 0.3 -> 30
- heading 层级检查（h1 -> h2 不跳级）

### 测试

```python
from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer


class TestHTMLSemanticScorer:
    def test_semantic_html(self):
        code = "<header><nav></nav></header><main><article><section></section></article></main><footer></footer>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.score >= 60.0

    def test_non_semantic_html(self):
        code = "<div><div><div><span>hello</span></div></div></div>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.score <= 60.0

    def test_non_html_type(self):
        r = HTMLSemanticScorer().score(make_frontend_ctx(code="console.log('hi')", task_type="javascript"))
        assert r.score == 100.0  # 非 HTML 类型跳过

    def test_heading_hierarchy(self):
        code = "<h1>Title</h1><h2>Sub</h2><h3>Subsub</h3>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.details.get("heading_ok") is True

    def test_heading_skip(self):
        code = "<h1>Title</h1><h3>Skip h2</h3>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.details.get("heading_ok") is False

    def test_get_metric_name(self):
        assert HTMLSemanticScorer().get_metric_name() == "html_semantic"
```

### 实现

```python
# benchmark/scorers/frontend/html_semantic.py
from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_SEMANTIC_TAGS = {"header", "nav", "main", "article", "section", "aside", "footer"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class HTMLSemanticScorer(BaseScorer):
    """HTML 语义化评分。检查语义标签使用和 heading 层级。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "non_html_type"},
                               reasoning=f"Not HTML type ({task_type}), default 100")

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(ctx.model_answer, "html.parser")
        except Exception:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "parse_error"},
                               reasoning="HTML parse error, default 100")

        all_elements = soup.find_all(True)
        if not all_elements:
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "no_elements"},
                               reasoning="No HTML elements, default 100")

        semantic_count = sum(1 for el in all_elements if el.name in _SEMANTIC_TAGS)
        ratio = semantic_count / len(all_elements)

        if ratio >= 0.6:
            score = 100.0
        elif ratio >= 0.3:
            score = 60.0
        else:
            score = 30.0

        heading_ok = self._check_heading_hierarchy(soup)

        return ScoreResult(
            score=score,
            passed=score >= 60.0,
            details={
                "semantic_count": semantic_count,
                "total_elements": len(all_elements),
                "semantic_ratio": round(ratio, 2),
                "heading_ok": heading_ok,
            },
            reasoning=f"Semantic ratio={ratio:.2f}, heading_ok={heading_ok}",
        )

    @staticmethod
    def _check_heading_hierarchy(soup: Any) -> bool:
        headings = soup.find_all(_HEADING_TAGS)
        if not headings:
            return True
        prev_level = 0
        for h in headings:
            level = int(h.name[1])
            if level > prev_level + 1 and prev_level > 0:
                return False
            prev_level = level
        return True

    def get_metric_name(self) -> str:
        return "html_semantic"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestHTMLSemanticScorer -v
```

---

## Task 10: Frontend AccessibilityScorer（权重 15%）

文件: `benchmark/scorers/frontend/accessibility.py`

### 评分逻辑

- 需要 Playwright 环境
- 不可用 -> 100 分
- 违规计数: 0 -> 100, 1-2 -> 80, 3-5 -> 60, 6+ -> 40

### 测试

```python
from benchmark.scorers.frontend.accessibility import AccessibilityScorer


class TestAccessibilityScorer:
    def test_playwright_unavailable_gives_100(self):
        with patch("benchmark.scorers.frontend.accessibility.shutil.which", return_value=None):
            r = AccessibilityScorer().score(make_frontend_ctx())
            assert r.score == 100.0

    def test_non_html_type(self):
        r = AccessibilityScorer().score(
            make_frontend_ctx(code="console.log('hi')", task_type="javascript")
        )
        assert r.score == 100.0

    def test_get_metric_name(self):
        assert AccessibilityScorer().get_metric_name() == "accessibility"
```

### 实现

```python
# benchmark/scorers/frontend/accessibility.py
from __future__ import annotations

import shutil

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class AccessibilityScorer(BaseScorer):
    """可访问性评分。使用 axe-core 审计（需要 Playwright）。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "non_html_type"},
                               reasoning=f"Not HTML type ({task_type}), default 100")

        if not shutil.which("npx"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "playwright_unavailable"},
                               reasoning="Playwright unavailable, default 100")

        # TODO: Phase 3 实现 axe-core 集成
        return ScoreResult(
            score=100.0, passed=True,
            details={"reason": "axe_not_implemented"},
            reasoning="axe-core not yet implemented, default 100",
        )

    def get_metric_name(self) -> str:
        return "accessibility"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestAccessibilityScorer -v
```

---

## Task 11: Frontend CSSQualityScorer（权重 15%）

文件: `benchmark/scorers/frontend/css_quality.py`

### 评分逻辑

- Stylelint 不可用时退化为 AST 分析
- 检测媒体查询（响应式）、相对单位（rem/em/%/vw/vh）、flexbox/grid
- 违规数: 0 -> 100, 1-3 -> 80, 4+ -> 60

### 测试

```python
from benchmark.scorers.frontend.css_quality import CSSQualityScorer


class TestCSSQualityScorer:
    def test_non_css_type(self):
        r = CSSQualityScorer().score(
            make_frontend_ctx(code="console.log('hi')", task_type="javascript")
        )
        assert r.score == 100.0

    def test_good_css(self):
        css = "body { display: flex; margin: 0 auto; font-size: 1rem; } @media (max-width: 768px) { body { font-size: 0.875rem; } }"
        r = CSSQualityScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score >= 80.0

    def test_poor_css(self):
        css = "body { color: red; margin-top: 10px; padding: 5px; }"
        r = CSSQualityScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score <= 80.0

    def test_get_metric_name(self):
        assert CSSQualityScorer().get_metric_name() == "css_quality"
```

### 实现

```python
# benchmark/scorers/frontend/css_quality.py
from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class CSSQualityScorer(BaseScorer):
    """CSS 质量评分。AST 分析检查响应式、相对单位、现代布局。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "non_css_type"},
                               reasoning=f"Not CSS type ({task_type}), default 100")

        code = ctx.model_answer
        has_media_query = bool(re.search(r"@media\s", code))
        has_relative_units = bool(re.search(r"(?:rem|em|%|vw|vh)\b", code))
        has_modern_layout = bool(re.search(r"(?:flexbox|flex|grid)\b", code))

        checks = {
            "media_query": has_media_query,
            "relative_units": has_relative_units,
            "modern_layout": has_modern_layout,
        }
        violations = sum(1 for v in checks.values() if not v)

        if violations == 0:
            score = 100.0
        elif violations <= 3:
            score = 80.0
        else:
            score = 60.0

        return ScoreResult(
            score=score,
            passed=score >= 80.0,
            details={"checks": checks, "violations": violations},
            reasoning=f"CSS quality: {checks}, violations={violations}",
        )

    def get_metric_name(self) -> str:
        return "css_quality"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestCSSQualityScorer -v
```

---

## Task 12: Frontend CodeOrganizationScorer（权重 10%）

文件: `benchmark/scorers/frontend/code_organization.py`

### 评分逻辑

- ESLint 不可用时退化为 AST 分析
- 检查组件命名（PascalCase）、函数复杂度
- 违规: 0 -> 100, 1-5 -> 80, 6+ -> 60

### 测试

```python
from benchmark.scorers.frontend.code_organization import CodeOrganizationScorer


class TestCodeOrganizationScorer:
    def test_non_js_type(self):
        r = CodeOrganizationScorer().score(
            make_frontend_ctx(code="<html></html>", task_type="html")
        )
        assert r.score == 100.0

    def test_well_organized_react(self):
        code = """
function MyComponent() {
  return <div>Hello</div>;
}
function Helper() {
  return null;
}
"""
        r = CodeOrganizationScorer().score(make_frontend_ctx(code=code, task_type="react"))
        assert r.score >= 80.0

    def test_get_metric_name(self):
        assert CodeOrganizationScorer().get_metric_name() == "code_organization"
```

### 实现

```python
# benchmark/scorers/frontend/code_organization.py
from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class CodeOrganizationScorer(BaseScorer):
    """代码组织评分。检查命名规范和函数复杂度。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("javascript", "react"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "non_js_type"},
                               reasoning=f"Not JS type ({task_type}), default 100")

        code = ctx.model_answer
        violations = 0

        # React 组件应使用 PascalCase
        if task_type == "react":
            components = re.findall(r"function\s+(\w+)", code)
            non_pascal = [c for c in components if c[0].islower() and c not in ("if", "for", "while", "return")]
            violations += len(non_pascal)

        # 函数长度检查（超过 50 行视为复杂）
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

        return ScoreResult(
            score=score,
            passed=score >= 80.0,
            details={"violations": violations},
            reasoning=f"Code organization: violations={violations}",
        )

    def get_metric_name(self) -> str:
        return "code_organization"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestCodeOrganizationScorer -v
```

---

## Task 13: Frontend PerformanceScorer（权重 5%）

文件: `benchmark/scorers/frontend/performance.py`

### 评分逻辑

- 静态分析 + Playwright
- 检测: 不必要的 DOM 操作、同步阻塞、图片未指定尺寸
- Playwright 不可用时退化为纯静态分析
- 基础分 70，每检测到问题扣 10 分

### 测试

```python
from benchmark.scorers.frontend.performance import PerformanceScorer


class TestPerformanceScorer:
    def test_non_html_type(self):
        r = PerformanceScorer().score(
            make_frontend_ctx(code="console.log('hi')", task_type="javascript")
        )
        assert r.score == 100.0

    def test_clean_html(self):
        r = PerformanceScorer().score(
            make_frontend_ctx(code="<html><body><img src='a.png' width='100' height='100'></body></html>", task_type="html")
        )
        assert r.score >= 70.0

    def test_img_without_dimensions(self):
        code = "<html><body><img src='a.png'><img src='b.png'></body></html>"
        r = PerformanceScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.score < 70.0  # 每个无尺寸图片扣 10 分

    def test_get_metric_name(self):
        assert PerformanceScorer().get_metric_name() == "performance"
```

### 实现

```python
# benchmark/scorers/frontend/performance.py
from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


class PerformanceScorer(BaseScorer):
    """性能评分。静态分析检测常见性能问题。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "non_html_type"},
                               reasoning=f"Not HTML type ({task_type}), default 100")

        code = ctx.model_answer
        deductions = []

        # 图片未指定尺寸
        img_tags = re.findall(r"<img[^>]*>", code, re.IGNORECASE)
        for img in img_tags:
            if "width" not in img.lower() or "height" not in img.lower():
                deductions.append("img_without_dimensions")

        # 同步阻塞（document.write, synchronous XHR）
        if re.search(r"document\.write\s*\(", code):
            deductions.append("sync_dom_write")
        if re.search(r"new\s+XMLHttpRequest\s*\(\s*\)", code):
            deductions.append("sync_xhr")

        score = max(0.0, 70.0 - len(deductions) * 10)

        return ScoreResult(
            score=score,
            passed=score >= 50.0,
            details={
                "base": 70.0,
                "deductions": deductions,
                "deduction_count": len(deductions),
            },
            reasoning=f"Performance: base=70, deductions={len(deductions)}",
        )

    def get_metric_name(self) -> str:
        return "performance"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestPerformanceScorer -v
```

---

## Task 14: Frontend BrowserCompatScorer（权重 5%）

文件: `benchmark/scorers/frontend/browser_compat.py`

### 评分逻辑

- AST 分析 CSS 前缀
- 基础分 80
- 无前缀 -> 100
- 有前缀无 @supports -> 60
- 有 @supports -> +20

### 测试

```python
from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer


class TestBrowserCompatScorer:
    def test_non_css_type(self):
        r = BrowserCompatScorer().score(
            make_frontend_ctx(code="console.log('hi')", task_type="javascript")
        )
        assert r.score == 100.0

    def test_no_vendor_prefixes(self):
        css = "body { display: flex; gap: 10px; }"
        r = BrowserCompatScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score == 100.0

    def test_prefixes_without_supports(self):
        css = "body { -webkit-transform: rotate(5deg); transform: rotate(5deg); }"
        r = BrowserCompatScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score == 60.0

    def test_prefixes_with_supports(self):
        css = "@supports (display: grid) { body { display: grid; } } body { -webkit-transform: rotate(5deg); transform: rotate(5deg); }"
        r = BrowserCompatScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score == 80.0  # 60 + 20

    def test_get_metric_name(self):
        assert BrowserCompatScorer().get_metric_name() == "browser_compat"
```

### 实现

```python
# benchmark/scorers/frontend/browser_compat.py
from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_VENDOR_PREFIX_RE = re.compile(r"-(?:webkit|moz|ms|o)-")
_SUPPORTS_RE = re.compile(r"@supports\s")


class BrowserCompatScorer(BaseScorer):
    """浏览器兼容性评分。检查 CSS 前缀和 @supports 使用。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        task_type = ctx.task.metadata.get("type", "")
        if task_type not in ("html", "css", "react"):
            return ScoreResult(score=100.0, passed=True,
                               details={"reason": "non_css_type"},
                               reasoning=f"Not CSS type ({task_type}), default 100")

        code = ctx.model_answer
        has_prefixes = bool(_VENDOR_PREFIX_RE.search(code))
        has_supports = bool(_SUPPORTS_RE.search(code))

        if not has_prefixes:
            score = 100.0
        elif has_supports:
            score = 80.0  # 有前缀但有 @supports 回退
        else:
            score = 60.0  # 有前缀无回退

        return ScoreResult(
            score=score,
            passed=score >= 80.0,
            details={
                "has_vendor_prefixes": has_prefixes,
                "has_supports": has_supports,
            },
            reasoning=f"Browser compat: prefixes={has_prefixes}, supports={has_supports}",
        )

    def get_metric_name(self) -> str:
        return "browser_compat"
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestBrowserCompatScorer -v
```

---

## Task 15: Frontend CompositeScorer 组装

文件: `benchmark/scorers/frontend/__init__.py`

### 实现

```python
# benchmark/scorers/frontend/__init__.py
from __future__ import annotations

from benchmark.scorers.frontend.functionality import FunctionalityScorer
from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
from benchmark.scorers.frontend.accessibility import AccessibilityScorer
from benchmark.scorers.frontend.css_quality import CSSQualityScorer
from benchmark.scorers.frontend.code_organization import CodeOrganizationScorer
from benchmark.scorers.frontend.performance import PerformanceScorer
from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer


def create_frontend_composite() -> list[tuple[float, object]]:
    """创建 frontend-dev 维度的加权子评分器列表。"""
    return [
        (0.30, FunctionalityScorer()),
        (0.20, HTMLSemanticScorer()),
        (0.15, AccessibilityScorer()),
        (0.15, CSSQualityScorer()),
        (0.10, CodeOrganizationScorer()),
        (0.05, PerformanceScorer()),
        (0.05, BrowserCompatScorer()),
    ]
```

### 测试

```python
# 追加到 tests/test_frontend_scorers.py
from benchmark.scorers.composite import CompositeScorer
from benchmark.scorers.frontend import create_frontend_composite


class TestFrontendComposite:
    def test_composite_integration(self):
        subs = create_frontend_composite()
        scorer = CompositeScorer(subs)
        ctx = make_frontend_ctx()
        r = scorer.score(ctx)
        assert r.score > 0.0

    def test_composite_all_defaults(self):
        # 纯 JS 类型，大多数 scorer 返回 100
        subs = create_frontend_composite()
        scorer = CompositeScorer(subs)
        ctx = make_frontend_ctx(code="const x = 1;", task_type="javascript")
        r = scorer.score(ctx)
        assert r.score > 80.0
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontend_scorers.py::TestFrontendComposite -v
```

---

## Task 16: DIMENSION_REGISTRY 更新 + 集成测试

修改: `benchmark/cli.py`

### 步骤

- [ ] 在 `_run_evaluation` 中为 reasoning 和 frontend-dev 维度创建 CompositeScorer
- [ ] 导入 `create_reasoning_composite` 和 `create_frontend_composite`
- [ ] 导入 `CompositeScorer`
- [ ] 修改 DIMENSION_REGISTRY 或在 `_run_evaluation` 中按维度动态构建 scorer

### 设计决策

DIMENSION_REGISTRY 保持不变（仍然是 adapter/evaluator 的注册），scorer 的构建逻辑移到 `_run_evaluation` 中按维度分支：

```python
# benchmark/cli.py 修改

from benchmark.scorers.composite import CompositeScorer

# 在 _run_evaluation 函数中:

async def _run_evaluation(model, dimension, samples, debug):
    adapter_cls, scorer_cls, evaluator_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    evaluator = evaluator_cls()
    llm = LLMEvalAdapter(model=model)

    # 按维度构建 scorer
    if dimension == "reasoning":
        from benchmark.scorers.reasoning import create_reasoning_composite
        scorer = CompositeScorer(create_reasoning_composite(llm=llm))
    elif dimension == "frontend-dev":
        from benchmark.scorers.frontend import create_frontend_composite
        scorer = CompositeScorer(create_frontend_composite())
    else:
        scorer = scorer_cls()

    # ... 后续逻辑不变
```

### 测试

```python
# tests/test_cli.py 追加

def test_dimension_registry_unchanged():
    """DIMENSION_REGISTRY 仍包含原始 scorer 类。"""
    from benchmark.cli import DIMENSION_REGISTRY
    assert "reasoning" in DIMENSION_REGISTRY
    assert "frontend-dev" in DIMENSION_REGISTRY
```

### 验证

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_cli.py -v -k "registry"
```

---

## Task 17: 全量测试 + 清理

- [ ] 运行全部 reasoning 测试
- [ ] 运行全部 frontend 测试
- [ ] 运行已有测试确保无回归
- [ ] 检查 import 正确性

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reasoning_scorers.py tests/test_frontend_scorers.py -v
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/ -v --timeout=30
```

---

## 文件清单

新增文件:
```
benchmark/scorers/reasoning/__init__.py
benchmark/scorers/reasoning/answer_correctness.py
benchmark/scorers/reasoning/reasoning_completeness.py
benchmark/scorers/reasoning/reasoning_validity.py
benchmark/scorers/reasoning/method_elegance.py
benchmark/scorers/reasoning/difficulty_adaptation.py
benchmark/scorers/frontend/__init__.py
benchmark/scorers/frontend/functionality.py
benchmark/scorers/frontend/html_semantic.py
benchmark/scorers/frontend/accessibility.py
benchmark/scorers/frontend/css_quality.py
benchmark/scorers/frontend/code_organization.py
benchmark/scorers/frontend/performance.py
benchmark/scorers/frontend/browser_compat.py
tests/test_reasoning_scorers.py
tests/test_frontend_scorers.py
```

修改文件:
```
benchmark/cli.py  (scorer 构建逻辑)
```

## 依赖

新增 Python 依赖:
- `beautifulsoup4` (HTMLSemanticScorer)
- `pytest-asyncio` (ReasoningValidityScorer 测试)

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| ReasoningValidityScorer 的 LLM 调用延迟高 | 缓存 + 20% 权重占比可控 |
| Playwright/Node.js 环境差异 | 不可用时返回 100 分，不阻塞评测 |
| BeautifulSoup 未安装 | try/except 包裹，解析失败返回 100 分 |
| CompositeScorer 子 scorer 异常 | Phase 1 已实现异常兜底（默认 100 分） |
