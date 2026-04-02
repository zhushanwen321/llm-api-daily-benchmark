# Stage 2: 可视化增强 + 新维度 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩展评测维度（system-architecture、frontend-dev），增强可视化（趋势图、基础统计）

**Architecture:**
- MMLUAdapter: 从 Hugging Face 加载 MMLU 数据集，选择 computer_science 和 abstract_algebra 学科
- FrontCodeAdapter: 自建前端题目，使用关键词/正则匹配评分
- Trends组件: 从 SQLite 查询时间序列数据，使用 matplotlib 绑制趋势图
- Statistics模块: 计算均值、标准差、95%置信区间

**Tech Stack:**
- `matplotlib>=3.7` - 趋势图绑定
- `scipy>=1.11` - 统计计算（scipy.stats.sem）
- `datasets` - Hugging Face 数据集加载

---

## 文件结构

```
benchmark/
├── adapters/
│   ├── mmlu_adapter.py          # 🆕 MMLU数据集适配器
│   └── frontcode_adapter.py      # 🆕 前端代码数据集适配器
├── scorers/
│   └── keyword_match_scorer.py   # 🆕 关键词匹配评分器（FrontCode用）
├── core/
│   └── statistics.py             # 🆕 统计计算模块
├── visualization/
│   ├── app.py                    # ✏️ 修改：添加统计卡片
│   └── components/
│       └── trends.py             # 🆕 趋势图组件
├── configs/
│   └── default.yaml              # ✏️ 修改：添加新维度配置
└── datasets/
    ├── mmlu/                     # 🆕 MMLU缓存目录
    └── frontcode/                # 🆕 FrontCode数据目录

tests/
├── test_mmlu_adapter.py          # 🆕 MMLU适配器测试
├── test_frontcode_adapter.py     # 🆕 FrontCode适配器测试
├── test_keyword_match_scorer.py  # 🆕 关键词匹配评分器测试
└── test_statistics.py            # 🆕 统计模块测试
```

---

## Task 1: 创建统计计算模块

**Files:**
- Create: `benchmark/core/statistics.py`
- Test: `tests/test_statistics.py`

- [ ] **Step 1: 编写统计模块的失败测试**

```python
# tests/test_statistics.py
import pytest
from benchmark.core.statistics import calculate_mean, calculate_std, calculate_confidence_interval

def test_calculate_mean():
    """计算均值."""
    scores = [80.0, 85.0, 90.0, 75.0, 95.0]
    result = calculate_mean(scores)
    assert result == 85.0

def test_calculate_std():
    """计算标准差."""
    scores = [80.0, 85.0, 90.0, 75.0, 95.0]
    result = calculate_std(scores)
    assert abs(result - 7.07) < 0.1  # 样本标准差

def test_calculate_confidence_interval():
    """计算95%置信区间."""
    scores = [80.0, 85.0, 90.0, 75.0, 95.0]
    lower, upper = calculate_confidence_interval(scores, confidence=0.95)
    assert lower < upper
    assert lower < 85.0 < upper

def test_empty_list_raises():
    """空列表应抛出异常."""
    with pytest.raises(ValueError):
        calculate_mean([])
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_statistics.py -v`
Expected: FAIL with "module 'benchmark.core.statistics' not found"

- [ ] **Step 3: 实现统计模块**

```python
# benchmark/core/statistics.py
"""统计计算模块：均值、标准差、置信区间."""

from __future__ import annotations

from statistics import mean, stdev

import scipy.stats


def calculate_mean(scores: list[float]) -> float:
    """计算分数的均值.

    Args:
        scores: 分数列表.

    Returns:
        均值.

    Raises:
        ValueError: 列表为空时.
    """
    if not scores:
        raise ValueError("Cannot calculate mean of empty list")
    return mean(scores)


def calculate_std(scores: list[float]) -> float:
    """计算分数的样本标准差.

    Args:
        scores: 分数列表.

    Returns:
        样本标准差.

    Raises:
        ValueError: 列表为空或只有一个元素时.
    """
    if len(scores) < 2:
        raise ValueError("Cannot calculate std with less than 2 samples")
    return stdev(scores)


def calculate_confidence_interval(
    scores: list[float],
    confidence: float = 0.95
) -> tuple[float, float]:
    """计算均值的置信区间（单次计算，非Bootstrap）.

    使用 t-distribution 计算置信区间。

    Args:
        scores: 分数列表.
        confidence: 置信水平（默认0.95）.

    Returns:
        (lower_bound, upper_bound).

    Raises:
        ValueError: 列表为空或只有一个元素时.
    """
    if len(scores) < 2:
        raise ValueError("Cannot calculate CI with less than 2 samples")

    import scipy.stats
    from statistics import mean, stdev

    n = len(scores)
    sample_mean = mean(scores)
    sample_std = stdev(scores)
    standard_error = sample_std / (n ** 0.5)

    # 使用 t-distribution
    t_critical = scipy.stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * standard_error

    return (sample_mean - margin_of_error, sample_mean + margin_of_error)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_statistics.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_statistics.py benchmark/core/statistics.py
git commit -m "feat(statistics): 添加统计计算模块（均值、标准差、置信区间）"
```

---

## Task 2: 创建关键词匹配评分器

**Files:**
- Create: `benchmark/scorers/keyword_match_scorer.py`
- Test: `tests/test_keyword_match_scorer.py`

- [ ] **Step 1: 编写评分器的失败测试**

```python
# tests/test_keyword_match_scorer.py
import pytest
from benchmark.models.schemas import TaskDefinition, ScoreResult
from benchmark.scorers.keyword_match_scorer import KeywordMatchScorer

def test_score_with_all_keywords():
    """包含所有关键词得满分."""
    scorer = KeywordMatchScorer(keywords=["div", "class", "button"])
    task = TaskDefinition(
        task_id="test_1",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="Create a div with class button",
        expected_output="",
        metadata={}
    )
    result = scorer.score('<div class="button">Click</div>', "", task)
    assert result.score == 100.0
    assert result.passed is True

def test_score_with_partial_keywords():
    """包含部分关键词按比例得分."""
    scorer = KeywordMatchScorer(keywords=["div", "class", "button", "onclick", "addEventListener"])
    task = TaskDefinition(
        task_id="test_2",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="",
        expected_output="",
        metadata={}
    )
    result = scorer.score('<div class="button">Click</div>', "", task)
    assert result.score == 60.0  # 3/5 = 60%

def test_score_with_regex_patterns():
    """支持正则表达式匹配."""
    scorer = KeywordMatchScorer(
        keywords=[r"function\s+\w+", r"const\s+\w+\s*="],
        use_regex=True
    )
    task = TaskDefinition(
        task_id="test_3",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="",
        expected_output="",
        metadata={}
    )
    result = scorer.score('function handleClick() { const x = 1; }', "", task)
    assert result.score == 100.0

def test_score_with_no_match():
    """没有匹配返回0分."""
    scorer = KeywordMatchScorer(keywords=["div", "span"])
    task = TaskDefinition(
        task_id="test_4",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="",
        expected_output="",
        metadata={}
    )
    result = scorer.score('p paragraph text', "", task)
    assert result.score == 0.0
    assert result.passed is False
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_keyword_match_scorer.py -v`
Expected: FAIL with "module 'benchmark.scorers.keyword_match_scorer' not found"

- [ ] **Step 3: 实现关键词匹配评分器**

```python
# benchmark/scorers/keyword_match_scorer.py
"""关键词匹配评分器。用于前端代码评测."""

from __future__ import annotations

import re
from typing import List

from benchmark.models.schemas import ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class KeywordMatchScorer(BaseScorer):
    """关键词匹配评分器。

    检查代码是否包含预期的关键词或正则表达式模式。
    根据匹配项数量计算得分（匹配数/总数 * 100）。
    """

    def __init__(
        self,
        keywords: List[str],
        use_regex: bool = False,
        case_sensitive: bool = False
    ):
        """初始化评分器.

        Args:
            keywords: 关键词列表或正则表达式列表.
            use_regex: 是否使用正则表达式匹配.
            case_sensitive: 是否区分大小写.
        """
        self.keywords = keywords
        self.use_regex = use_regex
        self.case_sensitive = case_sensitive

    def score(
        self,
        model_output: str,
        expected: str,  # noqa: ARG002 — 未使用
        task: TaskDefinition,  # noqa: ARG002 — 未使用
    ) -> ScoreResult:
        """对模型输出进行评分.

        Args:
            model_output: 模型生成的代码.
            expected: 未使用.
            task: 原始任务定义.

        Returns:
            ScoreResult 包含分数、是否通过、详情.
        """
        if not self.keywords:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "No keywords configured"},
                reasoning="No keywords to match"
            )

        search_text = model_output if self.case_sensitive else model_output.lower()
        matched = []
        matched_indices = []

        for idx, keyword in enumerate(self.keywords):
            search_keyword = keyword if self.case_sensitive else keyword.lower()

            if self.use_regex:
                if re.search(search_keyword, search_text):
                    matched.append(keyword)
                    matched_indices.append(idx)
            else:
                if search_keyword in search_text:
                    matched.append(keyword)
                    matched_indices.append(idx)

        score = len(matched) / len(self.keywords) * 100
        passed = score >= 50.0  # 至少匹配50%才算通过

        return ScoreResult(
            score=score,
            passed=passed,
            details={
                "matched": matched,
                "matched_indices": matched_indices,
                "total_keywords": len(self.keywords),
                "match_rate": f"{len(matched)}/{len(self.keywords)}"
            },
            reasoning=f"Matched {len(matched)}/{len(self.keywords)} keywords: {matched}"
        )

    def get_metric_name(self) -> str:
        return "keyword_match"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_keyword_match_scorer.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/test_keyword_match_scorer.py benchmark/scorers/keyword_match_scorer.py
git commit -m "feat(scorer): 添加关键词匹配评分器（用于前端代码评测）"
```

---

## Task 3: 创建 MMLU 适配器

**Files:**
- Create: `benchmark/adapters/mmlu_adapter.py`
- Create: `benchmark/datasets/mmlu/.gitkeep`
- Test: `tests/test_mmlu_adapter.py`

- [ ] **Step 1: 编写 MMLU 适配器的失败测试**

```python
# tests/test_mmlu_adapter.py
import pytest
from benchmark.adapters.mmlu_adapter import MMLUAdapter

def test_load_returns_5_tasks():
    """加载应返回5个任务."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    tasks = adapter.load("benchmark/datasets/mmlu")
    assert len(tasks) == 5

def test_all_tasks_have_required_fields():
    """所有任务应包含必需字段."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    tasks = adapter.load("benchmark/datasets/mmlu")
    for task in tasks:
        assert task.task_id
        assert task.dimension == "system-architecture"
        assert task.dataset == "mmlu"
        assert task.prompt
        assert task.expected_output  # MMLU是选择题，expected_output是正确选项

def test_validate_valid_task():
    """验证有效任务应返回True."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    from benchmark.models.schemas import TaskDefinition
    task = TaskDefinition(
        task_id="mmlu_test_1",
        dimension="system-architecture",
        dataset="mmlu",
        prompt="Test question",
        expected_output="A",
        metadata={}
    )
    assert adapter.validate(task) is True

def test_get_dimension():
    """get_dimension应返回system-architecture."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    assert adapter.get_dimension() == "system-architecture"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_mmlu_adapter.py -v`
Expected: FAIL with "module 'benchmark.adapters.mmlu_adapter' not found"

- [ ] **Step 3: 实现 MMLU 适配器**

```python
# benchmark/adapters/mmlu_adapter.py
"""MMLU 数据集适配器。加载 computer_science 和 abstract_algebra 学科."""

from __future__import annotations

import os
from typing import List

from datasets import load_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class MMLUAdapter(DatasetAdapter):
    """MMLU 适配器，选择指定学科的最难题目."""

    def __init__(self, subjects: List[str] | None = None):
        """初始化适配器.

        Args:
            subjects: 学科列表. 默认为 ["computer_science", "abstract_algebra"].
        """
        self.subjects = subjects or ["computer_science", "abstract_algebra"]

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 MMLU 题目.

        从每个学科选择一定数量的题目，总共返回 5 题.
        computer_science 选择 3 题，abstract_algebra 选择 2 题.

        Args:
            path: 数据集缓存路径.

        Returns:
            TaskDefinition 列表.
        """
        cache_dir = path or os.path.join("benchmark", "datasets", "mmlu")
        all_tasks = []

        # 计算每个学科的题目数量
        num_subjects = len(self.subjects)
        if num_subjects == 2:
            counts = {self.subjects[0]: 3, self.subjects[1]: 2}
        else:
            # 平均分配，多余给第一个学科
            base_count = 5 // num_subjects
            remainder = 5 % num_subjects
            counts = {s: base_count + (1 if i == 0 else 0) for i, s in enumerate(self.subjects)}

        for subject in self.subjects:
            count = counts.get(subject, 1)
            try:
                dataset = load_dataset(
                    "cais/mmlu",
                    subject,
                    split="test",
                    cache_dir=cache_dir,
                    download_mode="reuse_dataset_if_exists",
                )
            except Exception as e:
                # 如果学科不存在，尝试添加前缀
                try:
                    dataset = load_dataset(
                        "cais/mmlu",
                        f"mmlu_{subject}",
                        split="test",
                        cache_dir=cache_dir,
                        download_mode="reuse_dataset_if_exists",
                    )
                except Exception:
                    raise ValueError(
                        f"Failed to load MMLU subject '{subject}'. "
                        f"Available subjects can be found at https://huggingface.co/datasets/cais/mmlu"
                    ) from e

            # 简单策略：取前 count 个题目
            # MMLU 题目格式：{"question": ..., "choices": [...], "answer": 0}（answer是索引）
            for idx in range(min(count, len(dataset))):
                item = dataset[idx]
                question = item["question"]
                choices = item.get("choices", [])
                answer_idx = item["answer"]  # 正确答案的索引

                # 构造题目文本
                if choices:
                    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                    prompt_text = f"{question}\n\n{choices_text}\n\nAnswer with the letter (A, B, C, D)."
                    expected_answer = chr(65 + answer_idx)  # 索引转字母 (0->A, 1->B, etc.)
                else:
                    prompt_text = question
                    expected_answer = str(answer_idx)

                task = TaskDefinition(
                    task_id=f"mmlu_{subject}_{idx + 1}",
                    dimension="system-architecture",
                    dataset="mmlu",
                    prompt=build_structured_prompt(prompt_text, "system-architecture"),
                    expected_output=expected_answer,
                    metadata={
                        "subject": subject,
                        "source": "cais/mmlu",
                        "choices": choices,
                        "answer_idx": answer_idx
                    }
                )
                all_tasks.append(task)

        return all_tasks[:5]  # 确保最多返回5题

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式."""
        return bool(
            task.task_id and
            task.prompt and
            task.expected_output and
            task.dimension == "system-architecture"
        )

    def get_dimension(self) -> str:
        return "system-architecture"
```

- [ ] **Step 4: 创建数据集目录标记文件**

Run:
```bash
mkdir -p benchmark/datasets/mmlu
touch benchmark/datasets/mmlu/.gitkeep
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_mmlu_adapter.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add tests/test_mmlu_adapter.py benchmark/adapters/mmlu_adapter.py benchmark/datasets/mmlu/.gitkeep
git commit -m "feat(adapter): 添加MMLU适配器（computer_science + abstract_algebra）"
```

---

## Task 4: 创建 FrontCode 适配器

**Files:**
- Create: `benchmark/adapters/frontcode_adapter.py`
- Create: `benchmark/datasets/frontcode/tasks.json`
- Test: `tests/test_frontcode_adapter.py`

- [ ] **Step 1: 编写 FrontCode 适配器的失败测试**

```python
# tests/test_frontcode_adapter.py
import pytest
from benchmark.adapters.frontcode_adapter import FrontCodeAdapter

def test_load_returns_5_tasks():
    """加载应返回5个任务."""
    adapter = FrontCodeAdapter()
    tasks = adapter.load("benchmark/datasets/frontcode")
    assert len(tasks) == 5

def test_tasks_have_correct_types():
    """任务应包含不同类型的前端题目."""
    adapter = FrontCodeAdapter()
    tasks = adapter.load("benchmark/datasets/frontcode")
    task_types = [task.metadata.get("type") for task in tasks]
    expected_types = ["html", "css", "javascript", "react", "complex"]
    assert set(task_types) == set(expected_types)

def test_validate_valid_task():
    """验证有效任务应返回True."""
    adapter = FrontCodeAdapter()
    from benchmark.models.schemas import TaskDefinition
    task = TaskDefinition(
        task_id="frontcode_1",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="Create a button",
        expected_output="",
        metadata={"type": "html", "keywords": ["button"]}
    )
    assert adapter.validate(task) is True

def test_get_dimension():
    """get_dimension应返回frontend-dev."""
    adapter = FrontCodeAdapter()
    assert adapter.get_dimension() == "frontend-dev"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_frontcode_adapter.py -v`
Expected: FAIL with "module 'benchmark.adapters.frontcode_adapter' not found"

- [ ] **Step 3: 创建 FrontCode 任务数据文件**

```json
// benchmark/datasets/frontcode/tasks.json
{
  "tasks": [
    {
      "id": "frontcode_html_1",
      "type": "html",
      "prompt": "Create a semantic HTML structure for a blog post page. Include: a header with navigation, a main article area with heading and paragraphs, and a footer with copyright.",
      "keywords": ["header", "nav", "main", "article", "h1", "footer"]
    },
    {
      "id": "frontcode_css_1",
      "type": "css",
      "prompt": "Write CSS to create a responsive card layout with: white background, subtle shadow, rounded corners (8px), padding (20px), and flexbox center alignment for content.",
      "keywords": ["background", "box-shadow", "border-radius", "padding", "display", "flex", "justify-content", "align-items"]
    },
    {
      "id": "frontcode_js_1",
      "type": "javascript",
      "prompt": "Write JavaScript to implement a debounce function that takes a function and delay (ms), returns a debounced version that only executes after delay milliseconds have passed since the last invocation.",
      "keywords": ["function", "setTimeout", "clearTimeout", "return", "arguments"]
    },
    {
      "id": "frontcode_react_1",
      "type": "react",
      "prompt": "Create a React functional component called 'Button' that accepts: children (for label), onClick (callback), and variant ('primary' | 'secondary' props. Use Tailwind CSS classes for styling.",
      "keywords": ["React", "useState", "props", "onClick", "children", "variant"]
    },
    {
      "id": "frontcode_complex_1",
      "type": "complex",
      "prompt": "Build a complete Todo List component with: add task input, delete button for each item, and checkbox to toggle completion. Use React hooks (useState) and include basic CSS styling.",
      "keywords": ["useState", "useState", "map", "filter", "onChange", "onClick", "checkbox", "input"]
    }
  ]
}
```

- [ ] **Step 4: 实现 FrontCode 适配器**

```python
# benchmark/adapters/frontcode_adapter.py
"""FrontCode 数据集适配器。自建前端评测题目."""

from __future__ import annotations

import json
import os
from typing import List

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class FrontCodeAdapter(DatasetAdapter):
    """FrontCode 适配器，加载自建前端评测题目."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 FrontCode 题目.

        Args:
            path: 数据集路径.

        Returns:
            TaskDefinition 列表.
        """
        data_dir = path or os.path.join("benchmark", "datasets", "frontcode")
        tasks_file = os.path.join(data_dir, "tasks.json")

        if not os.path.exists(tasks_file):
            raise FileNotFoundError(
                f"FrontCode tasks file not found: {tasks_file}. "
                f"Please create tasks.json in {data_dir}"
            )

        with open(tasks_file) as f:
            data = json.load(f)

        tasks = []
        for item in data.get("tasks", []):
            task = TaskDefinition(
                task_id=item["id"],
                dimension="frontend-dev",
                dataset="frontcode",
                prompt=build_structured_prompt(item["prompt"], "frontend-dev"),
                expected_output="",  # FrontCode 使用关键词匹配，不需要 expected_output
                metadata={
                    "type": item["type"],
                    "keywords": item["keywords"],
                    "source": "frontcode"
                }
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式."""
        return bool(
            task.task_id and
            task.prompt and
            task.dimension == "frontend-dev" and
            "keywords" in task.metadata
        )

    def get_dimension(self) -> str:
        return "frontend-dev"
```

- [ ] **Step 5: 创建数据集目录和数据文件**

Run:
```bash
mkdir -p benchmark/datasets/frontcode
# tasks.json 已在 Step 3 创建
```

- [ ] **Step 6: 运行测试验证通过**

Run: `pytest tests/test_frontcode_adapter.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add tests/test_frontcode_adapter.py benchmark/adapters/frontcode_adapter.py benchmark/datasets/frontcode/tasks.json
git commit -m "feat(adapter): 添加FrontCode适配器（自建前端评测题目）"
```

---

## Task 5: 创建趋势图组件

**Files:**
- Create: `benchmark/visualization/components/trends.py`
- Create: `benchmark/visualization/components/__init__.py`

- [ ] **Step 1: 编写趋势图组件的失败测试**

```python
# tests/test_trends.py
import pytest
from datetime import datetime, timedelta
from benchmark.visualization.components.trends import (
    get_trend_data,
    create_trend_figure
)

def test_get_trend_data_returns_correct_structure():
    """获取趋势数据应返回正确的结构."""
    # 假设已有测试数据在数据库中
    data = get_trend_data(
        conn=None,  # 使用 mock connection
        model="glm-4.7",
        dimension="reasoning",
        days=30
    )
    assert "dates" in data
    assert "scores" in data
    assert len(data["dates"]) == len(data["scores"])

def test_create_trend_figure():
    """创建趋势图应返回 matplotlib Figure."""
    data = {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "scores": [80.0, 85.0, 82.0]
    }
    fig = create_trend_figure(data, title="Test Trend")
    assert fig is not None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_trends.py -v`
Expected: FAIL with "module 'benchmark.visualization.components.trends' not found"

- [ ] **Step 3: 实现趋势图组件**

```python
# benchmark/visualization/components/__init__.py
"""可视化组件."""

- [ ] **Step 4: 创建趋势图组件**

```python
# benchmark/visualization/components/trends.py
"""趋势图组件。展示分数随时间变化的趋势."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3


def get_trend_data(
    conn: sqlite3.Connection,
    model: str,
    dimension: str,
    days: int = 30
) -> dict[str, list]:
    """从数据库获取趋势数据.

    Args:
        conn: SQLite 连接.
        model: 模型名称.
        dimension: 评测维度.
        days: 天数范围.

    Returns:
        包含 dates 和 scores 的字典.
    """
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    query = """
        SELECT
            DATE(r.created_at) as date,
            AVG(r.final_score) as avg_score
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        WHERE e.model = ?
          AND e.dimension = ?
          AND DATE(r.created_at) >= ?
        GROUP BY DATE(r.created_at)
        ORDER BY date ASC
    """

    cursor = conn.execute(query, (model, dimension, cutoff_date))
    rows = cursor.fetchall()

    return {
        "dates": [row["date"] for row in rows],
        "scores": [row["avg_score"] for row in rows]
    }


def create_trend_figure(
    data: dict[str, list],
    title: str = "Score Trend"
) -> plt.Figure:
    """创建趋势图.

    Args:
        data: 包含 dates 和 scores 的字典.
        title: 图表标题.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if not data["dates"]:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    ax.plot(data["dates"], data["scores"], marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # 设置 Y 轴范围
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_multi_model_trend(
    conn: sqlite3.Connection,
    models: list[str],
    dimension: str,
    days: int = 30
) -> plt.Figure:
    """创建多模型对比趋势图.

    Args:
        conn: SQLite 连接.
        models: 模型名称列表.
        dimension: 评测维度.
        days: 天数范围.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in models:
        data = get_trend_data(conn, model, dimension, days)
        if data["dates"]:
            ax.plot(data["dates"], data["scores"], marker="o", label=model, linewidth=2, markersize=4)

    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title(f"{dimension} - Model Comparison (Last {days} days)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_trends.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add tests/test_trends.py benchmark/visualization/components/__init__.py benchmark/visualization/components/trends.py
git commit -m "feat(visualization): 添加趋势图组件"
```

---

## Task 6: 更新 Streamlit 应用（添加统计卡片和趋势图）

**Files:**
- Modify: `benchmark/visualization/app.py`

- [ ] **Step 1: 编写集成测试（验证新功能存在）**

```python
# tests/test_app_integration.py
import pytest

def test_app_imports_statistics_module():
    """验证 app 可以导入统计模块."""
    from benchmark.visualization import app
    from benchmark.core import statistics
    assert statistics is not None

def test_app_imports_trends_component():
    """验证 app 可以导入趋势图组件."""
    from benchmark.visualization import app
    from benchmark.visualization.components import trends
    assert trends is not None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_app_integration.py -v`
Expected: FAIL (imports 尚未添加)

- [ ] **Step 3: 更新 Streamlit 应用**

在 `benchmark/visualization/app.py` 中添加以下内容：

```python
# 在文件顶部的 import 区域添加:
from benchmark.core.statistics import calculate_mean, calculate_std, calculate_confidence_interval
from benchmark.visualization.components import trends

# 在 main() 函数中，st.title() 之后添加统计卡片展示:
def main():
    st.set_page_config(page_title="LLM Benchmark", layout="wide")
    st.title("LLM Benchmark Results")

    conn = get_connection()

    # ... 现有代码 ...

    # 添加统计卡片（在 filters 之后）
    if not df.empty:
        st.subheader("📊 Statistics Summary")

        # 计算统计数据
        scores = df["final_score"].tolist()
        if len(scores) >= 2:
            mean_score = calculate_mean(scores)
            std_score = calculate_std(scores)
            ci_lower, ci_upper = calculate_confidence_interval(scores)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Mean", f"{mean_score:.1f}")
            with col2:
                st.metric("Std Dev", f"±{std_score:.2f}")
            with col3:
                st.metric("95% CI", f"[{ci_lower:.1f}, {ci_upper:.1f}]")
            with col4:
                st.metric("Max", f"{max(scores):.1f}")
            with col5:
                st.metric("Min", f"{min(scores):.1f}")

    # ... 现有结果表格代码 ...

    # 添加趋势图标签页
    tab1, tab2, tab3 = st.tabs(["Results", "Trends", "Detail"])

    with tab1:
        # 现有的结果展示代码
        pass

    with tab2:
        st.subheader("Score Trends")

        # 时间范围选择
        time_range = st.selectbox("Time Range", ["7 days", "30 days", "90 days", "All"], index=1)
        days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "All": 365}
        selected_days = days_map[time_range]

        # 模型选择
        available_models = get_models(conn)
        if len(available_models) > 1:
            show_comparison = st.checkbox("Compare models", value=False)
            if show_comparison:
                selected_models = st.multiselect("Select models", available_models, default=available_models[:2])
                if selected_models and len(selected_models) >= 1:
                    fig = trends.create_multi_model_trend(conn, selected_models, selected_dimension or "reasoning", selected_days)
                    st.pyplot(fig)
            else:
                if selected_model and selected_model != "All":
                    fig = trends.create_multi_model_trend(conn, [selected_model], selected_dimension or "reasoning", selected_days)
                    st.pyplot(fig)
        else:
            if selected_model and selected_model != "All":
                data = trends.get_trend_data(conn, selected_model, selected_dimension or "reasoning", selected_days)
                fig = trends.create_trend_figure(data, title=f"{selected_model} - {selected_dimension or 'reasoning'} Trend")
                st.pyplot(fig)

    with tab3:
        # 现有的详情展示代码
        pass
```

注意：需要重构现有代码以适应新的标签页结构。完整的修改见实际文件。

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_app_integration.py -v`
Expected: PASS

- [ ] **Step 5: 手动验证（可选但推荐）**

Run: `streamlit run benchmark/visualization/app.py`
Expected: 可以看到统计卡片和趋势图标签页

- [ ] **Step 6: 提交**

```bash
git add tests/test_app_integration.py benchmark/visualization/app.py
git commit -m "feat(visualization): 添加统计卡片和趋势图到Streamlit应用"
```

---

## Task 7: 更新配置文件

**Files:**
- Modify: `benchmark/configs/default.yaml`

- [ ] **Step 1: 更新 default.yaml**

```yaml
# LLM Benchmark 默认配置
model: "glm-4.7"
temperature: 0.0
max_tokens: 4096
max_retries: 3
timeout: 300

# 数据集根目录
dataset_root: "benchmark/datasets"

# 维度权重配置
dimensions:
  reasoning:
    adapter: "gsm8k"
    auto_weight: 1.0

  backend-dev:
    adapter: "bigcodebench"
    auto_weight: 1.0

  system-architecture:  # 新增
    adapter: "mmlu"
    subjects: ["computer_science", "abstract_algebra"]
    auto_weight: 1.0

  frontend-dev:  # 新增
    adapter: "frontcode"
    auto_weight: 1.0

# 难度权重
difficulty_weights:
  easy: 1.0
  medium: 1.5
  hard: 2.0
```

- [ ] **Step 2: 提交**

```bash
git add benchmark/configs/default.yaml
git commit -m "chore(config): 添加 system-architecture 和 frontend-dev 维度配置"
```

---

## Task 8: 更新项目依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 matplotlib 和 scipy 到主依赖**

将 `matplotlib` 和 `scipy` 从 `benchmark` 可选依赖移到主依赖：

```toml
[project]
name = "llm-api-daily-benchmark"
version = "0.1.0"
description = "LLM API daily benchmark - track model performance over time"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
    "streamlit>=1.28",
    "datasets>=2.14",
    "pyyaml>=6.0",
    "requests>=2.31",
    "httpx>=0.27",
    "pandas>=2.0",
    "python-dotenv>=1.0",
    "matplotlib>=3.7",  # 新增
    "scipy>=1.11",     # 新增
]

# ... 其余保持不变 ...
```

- [ ] **Step 2: 提交**

```bash
git add pyproject.toml
git commit -m "chore(deps): 添加 matplotlib 和 scipy 到主依赖"
```

---

## 完成检查

- [ ] 所有任务已完成
- [ ] 所有测试通过: `pytest tests/ -v`
- [ ] 手动验证 Streamlit 应用: `streamlit run benchmark/visualization/app.py`
- [ ] 配置文件已更新
- [ ] 新增维度可以评测: `python -m benchmark evaluate --model glm-4.7 --dimension system-architecture --samples 5`
