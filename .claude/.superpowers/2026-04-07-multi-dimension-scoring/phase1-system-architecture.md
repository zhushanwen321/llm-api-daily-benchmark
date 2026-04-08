# Phase 1: System-Architecture Scorers (Task 6)

> 依赖: [主文档 Task 1](./phase1-infra-backend-sysarch.md) — CompositeScorer

---

## Task 6: System-Architecture Scorers（全部正则匹配）

5 个 scorer，全部基于正则匹配分析 `reasoning_content`。`reasoning_content` 为空时全部返回 100 分。

| Scorer | 文件 | 权重 | 检测内容 |
|--------|------|------|----------|
| AnswerCorrectnessScorer | `answer_correctness.py` | 30% | 复用 ChoiceMatchScorer 逻辑 |
| ReasoningCompletenessScorer | `reasoning_completeness.py` | 25% | 推理长度+结构 |
| OptionAnalysisScorer | `option_analysis.py` | 20% | 排除法/对比分析 |
| ReasoningConfidenceScorer | `reasoning_confidence.py` | 15% | 确定性/不确定性表述 |
| SubjectAdaptationScorer | `subject_adaptation.py` | 10% | 学科期望长度 |

---

### Task 6a: AnswerCorrectnessScorer (30%)

**Files:**
- Create: `benchmark/scorers/system_architecture/__init__.py` (空文件)
- Create: `benchmark/scorers/system_architecture/answer_correctness.py`
- Test: `tests/test_system_arch_correctness.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_system_arch_correctness.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.system_architecture.answer_correctness import AnswerCorrectnessScorer


def _make_ctx(answer: str, expected: str = "B", reasoning: str = "") -> ScoringContext:
    return ScoringContext(
        model_answer=answer, raw_output=answer, expected=expected,
        task=TaskDefinition(
            task_id="test", dimension="system-architecture",
            dataset="mmlu-pro", prompt="test", expected_output=expected,
            metadata={"category": "computer science", "num_options": 4},
        ),
        reasoning_content=reasoning,
    )


class TestAnswerCorrectnessScorer:
    def test_correct_answer(self):
        scorer = AnswerCorrectnessScorer()
        result = scorer.score(_make_ctx("The answer is B", "B"))
        assert result.score == 100.0
        assert result.passed is True

    def test_wrong_answer(self):
        scorer = AnswerCorrectnessScorer()
        result = scorer.score(_make_ctx("The answer is A", "B"))
        assert result.score == 0.0
        assert result.passed is False

    def test_answer_with_reasoning(self):
        scorer = AnswerCorrectnessScorer()
        result = scorer.score(_make_ctx(
            "After analysis, B is correct because...", "B",
            reasoning="Let me analyze each option...",
        ))
        assert result.score == 100.0

    def test_empty_reasoning(self):
        scorer = AnswerCorrectnessScorer()
        result = scorer.score(_make_ctx("B", "B", reasoning=""))
        # 答案正确，reasoning 为空不影响正确性评分
        assert result.score == 100.0

    def test_get_metric_name(self):
        assert AnswerCorrectnessScorer().get_metric_name() == "answer_correctness"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_system_arch_correctness.py -v
```

- [ ] **Step 3: 实现 AnswerCorrectnessScorer**

```python
# benchmark/scorers/system_architecture/__init__.py
# 空

# benchmark/scorers/system_architecture/answer_correctness.py
"""答案正确性评分器。复用 ChoiceMatchScorer 逻辑。"""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_CHOICE_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


class AnswerCorrectnessScorer(BaseScorer):
    """从模型输出中提取选项字母，与期望答案比较。复用 ChoiceMatchScorer 逻辑。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected_letter = ctx.expected.strip().upper()
        matches = _CHOICE_RE.findall(ctx.model_answer)

        if not matches:
            return ScoreResult(
                score=0.0, passed=False,
                details={"error": "No choice letter found"},
                reasoning="Model output contains no choice letter",
            )

        predicted = matches[-1].upper()
        passed = predicted == expected_letter

        return ScoreResult(
            score=100.0 if passed else 0.0,
            passed=passed,
            details={"predicted": predicted, "expected": expected_letter},
            reasoning=f"{'Correct' if passed else 'Incorrect'}: predicted={predicted}",
        )

    def get_metric_name(self) -> str:
        return "answer_correctness"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_system_arch_correctness.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/system_architecture/__init__.py benchmark/scorers/system_architecture/answer_correctness.py tests/test_system_arch_correctness.py
git commit -m "feat(scorer): add AnswerCorrectnessScorer for system-architecture"
```

---

### Task 6b: ReasoningCompletenessScorer (25%)

**Files:**
- Create: `benchmark/scorers/system_architecture/reasoning_completeness.py`
- Test: `tests/test_system_arch_reasoning.py`

**评分逻辑:**
- `reasoning_content` 为空 -> 100 分
- 检测推理结构: 是否提到各选项（A/B/C/D）、是否逐步分析
- 推理长度: 短于 50 字符扣分
- 有明确推理步骤加分

- [ ] **Step 1: 写失败测试**

```python
# tests/test_system_arch_reasoning.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.system_architecture.reasoning_completeness import ReasoningCompletenessScorer


def _make_ctx(reasoning: str, num_options: int = 4) -> ScoringContext:
    return ScoringContext(
        model_answer="B", raw_output="B", expected="B",
        task=TaskDefinition(
            task_id="test", dimension="system-architecture",
            dataset="mmlu-pro", prompt="test", expected_output="B",
            metadata={"category": "computer science", "num_options": num_options},
        ),
        reasoning_content=reasoning,
    )


class TestReasoningCompletenessScorer:
    def test_empty_reasoning(self):
        scorer = ReasoningCompletenessScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_full_reasoning(self):
        reasoning = (
            "Let me analyze each option:\n"
            "A. This is incorrect because...\n"
            "B. This is correct because...\n"
            "C. This is incorrect because...\n"
            "D. This is incorrect because...\n"
            "Therefore, the answer is B."
        )
        scorer = ReasoningCompletenessScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert result.score >= 80.0

    def test_short_reasoning(self):
        scorer = ReasoningCompletenessScorer()
        result = scorer.score(_make_ctx("B"))
        assert result.score < 100.0

    def test_reasoning_with_options(self):
        reasoning = "Option A is wrong. Option C is wrong. Option D is wrong. So B is correct."
        scorer = ReasoningCompletenessScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert result.score > 50.0

    def test_get_metric_name(self):
        assert ReasoningCompletenessScorer().get_metric_name() == "reasoning_completeness"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_system_arch_reasoning.py -v
```

- [ ] **Step 3: 实现 ReasoningCompletenessScorer**

```python
# benchmark/scorers/system_architecture/reasoning_completeness.py
"""推理完整性评分器。正则匹配推理长度+结构。"""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 检测是否提到选项字母（排除最终答案行）
_OPTION_MENTION_RE = re.compile(r"\b([A-Z])\b")

# 检测推理步骤关键词
_STEP_KEYWORDS_RE = re.compile(
    r"(because|therefore|since|however|first|second|third|finally|"
    r"because|thus|hence|consequently|given that|it follows)",
    re.IGNORECASE,
)


class ReasoningCompletenessScorer(BaseScorer):
    """基于推理内容的长度和结构评分。"""

    def __init__(self, min_length: int = 50) -> None:
        self._min_length = min_length

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning or not reasoning.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty reasoning, skip")

        score = 100.0
        details = {}
        length = len(reasoning.strip())

        # 长度评分: < min_length 扣分
        if length < self._min_length:
            length_penalty = (self._min_length - length) / self._min_length * 30
            score -= length_penalty
            details["length_penalty"] = round(length_penalty, 1)

        details["reasoning_length"] = length

        # 结构评分: 提到选项字母数量
        num_options = ctx.task.metadata.get("num_options", 4)
        mentioned_options = set(_OPTION_MENTION_RE.findall(reasoning))
        # 排除最终答案字母（只计推理中提到的）
        expected = ctx.expected.strip().upper()
        reasoning_mentions = len(mentioned_options - {expected}) if expected else len(mentioned_options)
        option_coverage = min(1.0, reasoning_mentions / max(num_options - 1, 1))
        option_bonus = option_coverage * 20
        score += option_bonus
        details["option_coverage"] = round(option_coverage, 2)

        # 推理步骤关键词加分
        step_matches = _STEP_KEYWORDS_RE.findall(reasoning)
        step_bonus = min(15, len(step_matches) * 3)
        score += step_bonus
        details["step_keywords"] = len(step_matches)

        score = round(min(100.0, score), 1)

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"推理完整性: length={length}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "reasoning_completeness"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_system_arch_reasoning.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/system_architecture/reasoning_completeness.py tests/test_system_arch_reasoning.py
git commit -m "feat(scorer): add ReasoningCompletenessScorer"
```

---

### Task 6c: OptionAnalysisScorer (20%)

**Files:**
- Create: `benchmark/scorers/system_architecture/option_analysis.py`
- Test: `tests/test_system_arch_option.py`

**评分逻辑:**
- `reasoning_content` 为空 -> 100 分
- 检测排除法关键词: "eliminate", "rule out", "incorrect", "wrong"
- 检测对比分析: "compared to", "versus", "while", "whereas", "on the other hand"
- 检测逐项分析: 多次提到不同选项

- [ ] **Step 1: 写失败测试**

```python
# tests/test_system_arch_option.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.system_architecture.option_analysis import OptionAnalysisScorer


def _make_ctx(reasoning: str) -> ScoringContext:
    return ScoringContext(
        model_answer="B", raw_output="B", expected="B",
        task=TaskDefinition(
            task_id="test", dimension="system-architecture",
            dataset="mmlu-pro", prompt="test", expected_output="B",
            metadata={"category": "computer science", "num_options": 4},
        ),
        reasoning_content=reasoning,
    )


class TestOptionAnalysisScorer:
    def test_empty_reasoning(self):
        scorer = OptionAnalysisScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_elimination_reasoning(self):
        reasoning = (
            "Option A is incorrect because it violates the principle.\n"
            "Option C can be ruled out since it doesn't apply.\n"
            "Option D is wrong because it contradicts the premise.\n"
            "Therefore B is the correct answer."
        )
        scorer = OptionAnalysisScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert result.score >= 70.0

    def test_no_analysis(self):
        scorer = OptionAnalysisScorer()
        result = scorer.score(_make_ctx("B is correct."))
        assert result.score < 70.0

    def test_comparison_reasoning(self):
        reasoning = (
            "Comparing A and B, A is too slow while B is efficient.\n"
            "On the other hand, C and D have other issues."
        )
        scorer = OptionAnalysisScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert result.score > 50.0

    def test_get_metric_name(self):
        assert OptionAnalysisScorer().get_metric_name() == "option_analysis"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_system_arch_option.py -v
```

- [ ] **Step 3: 实现 OptionAnalysisScorer**

```python
# benchmark/scorers/system_architecture/option_analysis.py
"""选项分析评分器。检测排除法和对比分析推理模式。"""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 排除法关键词
_ELIMINATION_RE = re.compile(
    r"(eliminate|rule out|incorrect|wrong|not correct|not the answer|"
    r"can be excluded|does not apply|is not)",
    re.IGNORECASE,
)

# 对比分析关键词
_COMPARISON_RE = re.compile(
    r"(compared to|versus|while|whereas|on the other hand|in contrast|"
    r"however|unlike|differ|between)",
    re.IGNORECASE,
)

# 选项字母提及
_OPTION_RE = re.compile(r"\b([A-Z])\b")


class OptionAnalysisScorer(BaseScorer):
    """检测排除法和对比分析推理模式。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning or not reasoning.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty reasoning, skip")

        eliminations = len(_ELIMINATION_RE.findall(reasoning))
        comparisons = len(_COMPARISON_RE.findall(reasoning))
        options_mentioned = len(set(_OPTION_RE.findall(reasoning)))

        # 基础分 30，排除法最多 +40，对比最多 +20，选项覆盖最多 +10
        score = 30.0
        score += min(40, eliminations * 8)
        score += min(20, comparisons * 5)
        score += min(10, options_mentioned * 2)
        score = round(min(100.0, score), 1)

        details = {
            "eliminations": eliminations,
            "comparisons": comparisons,
            "options_mentioned": options_mentioned,
        }

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details=details,
            reasoning=f"选项分析: elim={eliminations}, comp={comparisons}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "option_analysis"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_system_arch_option.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/system_architecture/option_analysis.py tests/test_system_arch_option.py
git commit -m "feat(scorer): add OptionAnalysisScorer"
```

---

### Task 6d: ReasoningConfidenceScorer (15%)

**Files:**
- Create: `benchmark/scorers/system_architecture/reasoning_confidence.py`
- Test: `tests/test_system_arch_confidence.py`

**评分逻辑:**
- `reasoning_content` 为空 -> 100 分
- 检测确定性表述加分: "clearly", "definitely", "must be", "certainly"
- 检测不确定性表述扣分: "maybe", "perhaps", "not sure", "could be", "might be", "I think", "probably"

- [ ] **Step 1: 写失败测试**

```python
# tests/test_system_arch_confidence.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.system_architecture.reasoning_confidence import ReasoningConfidenceScorer


def _make_ctx(reasoning: str) -> ScoringContext:
    return ScoringContext(
        model_answer="B", raw_output="B", expected="B",
        task=TaskDefinition(
            task_id="test", dimension="system-architecture",
            dataset="mmlu-pro", prompt="test", expected_output="B",
            metadata={"category": "computer science", "num_options": 4},
        ),
        reasoning_content=reasoning,
    )


class TestReasoningConfidenceScorer:
    def test_empty_reasoning(self):
        scorer = ReasoningConfidenceScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_confident_reasoning(self):
        reasoning = "The correct answer is clearly B. This must be the case because..."
        scorer = ReasoningConfidenceScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert result.score > 70.0

    def test_uncertain_reasoning(self):
        reasoning = "Maybe B is the answer? I think perhaps it could be B, but I'm not sure."
        scorer = ReasoningConfidenceScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert result.score < 70.0

    def test_neutral_reasoning(self):
        reasoning = "Option B satisfies all the given conditions."
        scorer = ReasoningConfidenceScorer()
        result = scorer.score(_make_ctx(reasoning))
        assert 60.0 <= result.score <= 100.0

    def test_get_metric_name(self):
        assert ReasoningConfidenceScorer().get_metric_name() == "reasoning_confidence"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_system_arch_confidence.py -v
```

- [ ] **Step 3: 实现 ReasoningConfidenceScorer**

```python
# benchmark/scorers/system_architecture/reasoning_confidence.py
"""推理信心度评分器。检测确定性/不确定性表述。"""

from __future__ import annotations

import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

_CERTAINTY_RE = re.compile(
    r"(clearly|definitely|certainly|must be|undoubtedly|obviously|"
    r"unquestionably|without doubt|for sure|surely)",
    re.IGNORECASE,
)

_UNCERTAINTY_RE = re.compile(
    r"(maybe|perhaps|not sure|could be|might be|I think|probably|"
    r"uncertain|possibly|I guess|I believe|seems like|it appears)",
    re.IGNORECASE,
)


class ReasoningConfidenceScorer(BaseScorer):
    """检测推理过程中的确定性/不确定性表述。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning or not reasoning.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty reasoning, skip")

        certainty = len(_CERTAINTY_RE.findall(reasoning))
        uncertainty = len(_UNCERTAINTY_RE.findall(reasoning))

        # 基础分 60, 确定性 +5/个(最多+40), 不确定性 -8/个(最多-50)
        score = 60.0
        score += min(40, certainty * 5)
        score -= min(50, uncertainty * 8)
        score = round(max(0.0, min(100.0, score)), 1)

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details={"certainty": certainty, "uncertainty": uncertainty},
            reasoning=f"推理信心: cert={certainty}, uncert={uncertainty}, score={score}",
        )

    def get_metric_name(self) -> str:
        return "reasoning_confidence"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_system_arch_confidence.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/system_architecture/reasoning_confidence.py tests/test_system_arch_confidence.py
git commit -m "feat(scorer): add ReasoningConfidenceScorer"
```

---

### Task 6e: SubjectAdaptationScorer (10%)

**Files:**
- Create: `benchmark/scorers/system_architecture/subject_adaptation.py`
- Test: `tests/test_system_arch_subject.py`

**评分逻辑:**
- `reasoning_content` 为空 -> 100 分
- 根据学科 (category) 设定期望的推理长度范围
  - computer science: 100-500 字符
  - math: 80-400 字符
  - physics: 100-500 字符
- 推理长度在期望范围内满分，过短或过长扣分

- [ ] **Step 1: 写失败测试**

```python
# tests/test_system_arch_subject.py
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.system_architecture.subject_adaptation import SubjectAdaptationScorer


def _make_ctx(reasoning: str, category: str = "computer science") -> ScoringContext:
    return ScoringContext(
        model_answer="B", raw_output="B", expected="B",
        task=TaskDefinition(
            task_id="test", dimension="system-architecture",
            dataset="mmlu-pro", prompt="test", expected_output="B",
            metadata={"category": category, "num_options": 4},
        ),
        reasoning_content=reasoning,
    )


class TestSubjectAdaptationScorer:
    def test_empty_reasoning(self):
        scorer = SubjectAdaptationScorer()
        result = scorer.score(_make_ctx(""))
        assert result.score == 100.0

    def test_cs_appropriate_length(self):
        reasoning = "A" * 200  # 200 字符，在 CS 期望范围内
        scorer = SubjectAdaptationScorer()
        result = scorer.score(_make_ctx(reasoning, "computer science"))
        assert result.score >= 80.0

    def test_too_short_for_cs(self):
        reasoning = "B is correct."
        scorer = SubjectAdaptationScorer()
        result = scorer.score(_make_ctx(reasoning, "computer science"))
        assert result.score < 80.0

    def test_math_appropriate_length(self):
        reasoning = "Let x = 5, then y = x^2 + 3 = 28."  # ~40 字符，偏短
        scorer = SubjectAdaptationScorer()
        result = scorer.score(_make_ctx(reasoning, "math"))
        # math 范围 80-400，40 偏短但不会太低
        assert result.score > 0

    def test_physics_appropriate_length(self):
        reasoning = "F" * 200
        scorer = SubjectAdaptationScorer()
        result = scorer.score(_make_ctx(reasoning, "physics"))
        assert result.score >= 80.0

    def test_unknown_category(self):
        reasoning = "x" * 150
        scorer = SubjectAdaptationScorer()
        result = scorer.score(_make_ctx(reasoning, "history"))
        # 未知类别使用默认范围，不应报错
        assert 0 <= result.score <= 100

    def test_get_metric_name(self):
        assert SubjectAdaptationScorer().get_metric_name() == "subject_adaptation"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_system_arch_subject.py -v
```

- [ ] **Step 3: 实现 SubjectAdaptationScorer**

```python
# benchmark/scorers/system_architecture/subject_adaptation.py
"""学科适应性评分器。根据学科期望长度评分。"""

from __future__ import annotations

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer

# 各学科期望推理长度范围 (min_chars, max_chars)
_SUBJECT_RANGES: dict[str, tuple[int, int]] = {
    "computer science": (100, 500),
    "math": (80, 400),
    "physics": (100, 500),
}

_DEFAULT_RANGE = (50, 600)


class SubjectAdaptationScorer(BaseScorer):
    """基于学科期望推理长度评分。"""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        reasoning = ctx.reasoning_content
        if not reasoning or not reasoning.strip():
            return ScoreResult(score=100.0, passed=True, reasoning="Empty reasoning, skip")

        category = ctx.task.metadata.get("category", "")
        lo, hi = _SUBJECT_RANGES.get(category, _DEFAULT_RANGE)
        length = len(reasoning.strip())

        if lo <= length <= hi:
            score = 100.0
        elif length < lo:
            score = round(100.0 * (length / lo), 1)
        else:
            # 过长: 线性扣分，最多扣到 50
            overshoot = length - hi
            score = round(max(50.0, 100.0 - overshoot * 0.1), 1)

        return ScoreResult(
            score=score,
            passed=score >= 60,
            details={
                "category": category,
                "reasoning_length": length,
                "expected_range": [lo, hi],
            },
            reasoning=f"学科适应性: {category}, length={length}, range=[{lo},{hi}], score={score}",
        )

    def get_metric_name(self) -> str:
        return "subject_adaptation"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_system_arch_subject.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/scorers/system_architecture/subject_adaptation.py tests/test_system_arch_subject.py
git commit -m "feat(scorer): add SubjectAdaptationScorer"
```
