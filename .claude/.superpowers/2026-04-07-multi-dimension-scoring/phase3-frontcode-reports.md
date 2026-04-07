# Phase 3: FrontCode 题目扩展 + 报告雷达图

> **依赖:** Phase 1 (CompositeScorer 基础设施), Phase 2 (Frontend + Reasoning Scorers)

---

## 概述

Phase 3 包含 4 个任务:
1. FrontCode 题目从 5 题扩展到 17 题（覆盖 7 个类型）
2. FrontCodeAdapter 支持 test_cases 字段映射
3. 报告中嵌入雷达图（SVG）
4. 排行榜展示各维度子分数

### 题目分布

| 类型 | 数量 | 覆盖内容 |
|------|------|----------|
| HTML | 2 | 语义化结构, 表单 |
| CSS | 3 | 响应式布局, Grid, 动画, Flexbox |
| JavaScript | 3 | debounce, throttle, 策略模式, 日期处理 |
| React | 3 | 组件, Hooks, 状态管理, 表单 |
| TypeScript | 2 | 类型定义, 泛型, 接口 |
| Accessibility | 2 | ARIA, 键盘导航, 屏幕阅读器 |
| 综合 | 2 | 完整页面（多技术栈组合） |

### 雷达图维度

| 维度 | 雷达轴 |
|------|--------|
| Backend | 正确性, 测试覆盖, 性能, 风格, 健壮性, 架构, 安全, 可扩展性 |
| Frontend | 功能, 语义化, 可访问性, CSS质量, 代码组织, 性能, 兼容性 |
| Reasoning | 答案正确性, 完整性, 推理正确性, 方法优雅度, 难度适配 |
| System-Architecture | 答案正确性, 推理完整性, 选项分析, 推理置信度, 学科适配 |

---

## Task 1: FrontCode 题目扩展

**Files:**
- Create: `scripts/generate_frontcode_tasks.py`
- Modify: `benchmark/datasets/frontcode/tasks.json`

### Step 1: 创建生成脚本

```python
# scripts/generate_frontcode_tasks.py
"""用 LLM 批量生成 FrontCode 题目，辅助人工审核。"""

from __future__ import annotations

import asyncio
import json
import sys

from benchmark.core.llm_adapter import LLMEvalAdapter

# 每批次的生成规格
_BATCHES = [
    {
        "type": "html",
        "count": 2,
        "difficulty": "easy",
        "description": "语义化 HTML 结构和表单",
    },
    {
        "type": "css",
        "count": 3,
        "difficulty": "medium",
        "description": "响应式布局、Grid、动画、Flexbox",
    },
    {
        "type": "javascript",
        "count": 3,
        "difficulty": "medium",
        "description": "debounce、throttle、策略模式、日期处理",
    },
    {
        "type": "react",
        "count": 3,
        "difficulty": "medium",
        "description": "组件、Hooks、状态管理、表单",
    },
    {
        "type": "typescript",
        "count": 2,
        "difficulty": "medium",
        "description": "类型定义、泛型、接口",
    },
    {
        "type": "accessibility",
        "count": 2,
        "difficulty": "hard",
        "description": "ARIA、键盘导航、屏幕阅读器",
    },
    {
        "type": "complex",
        "count": 2,
        "difficulty": "hard",
        "description": "完整页面，多技术栈组合",
    },
]

_SYSTEM_PROMPT = """你是一个前端评测题目出题专家。你需要为 LLM Benchmark 生成前端编码题目。
题目要求模型输出一段完整的前端代码。你需要:
1. 用中文描述题目要求
2. 列出代码中必须包含的关键词（用于关键词匹配评分）
3. 列出 3-5 个 DOM/逻辑断言（用于自动化功能测试）
4. 指定题目难度"""

_TASK_TEMPLATE = """请为前端评测生成 {count} 道「{type}」类型的题目。

要求:
- 难度: {difficulty}
- 覆盖范围: {description}

每道题返回以下 JSON 格式（不要包含其他文字）:
```json
[
  {{
    "id": "frontcode_{type}_{序号}",
    "type": "{type}",
    "prompt": "中文题目描述...",
    "keywords": ["keyword1", "keyword2", ...],
    "test_cases": ["断言1", "断言2", ...],
    "difficulty": "{difficulty}"
  }}
]
```

注意:
- prompt 必须用中文
- keywords 是代码中必须包含的技术关键词
- test_cases 是用自然语言描述的功能断言，后续会用于自动化评分
- 每道题的 id 序号从当前最大序号+1开始"""


async def generate_batch(llm: LLMEvalAdapter, batch: dict, existing_count: int) -> list[dict]:
    """生成一批题目。"""
    prompt = _TASK_TEMPLATE.format(**batch)
    resp = await llm.agenerate(
        prompt=prompt,
        model="zai/glm-5.1",
        temperature=0.7,
        system_message=_SYSTEM_PROMPT,
    )
    # 解析 JSON
    import re
    json_match = re.search(r'\[.*\]', resp.content, re.DOTALL)
    if not json_match:
        print(f"  [WARN] {batch['type']}: 无法从响应中提取 JSON")
        return []
    try:
        tasks = json.loads(json_match.group())
        # 修正 id 中的序号
        for i, task in enumerate(tasks):
            task["id"] = f"frontcode_{batch['type']}_{existing_count + i + 1}"
        return tasks
    except json.JSONDecodeError as e:
        print(f"  [WARN] {batch['type']}: JSON 解析失败: {e}")
        return []


async def main() -> None:
    tasks_file = "benchmark/datasets/frontcode/tasks.json"

    # 读取现有题目
    with open(tasks_file) as f:
        data = json.load(f)
    existing_tasks = data.get("tasks", [])
    print(f"现有题目数: {len(existing_tasks)}")

    llm = LLMEvalAdapter(model="zai/glm-5.1")
    all_new_tasks = []
    offset = len(existing_tasks)

    for batch in _BATCHES:
        print(f"\n生成 {batch['type']} ({batch['count']} 题)...")
        new_tasks = await generate_batch(llm, batch, offset)
        all_new_tasks.extend(new_tasks)
        offset += len(new_tasks)
        print(f"  成功生成: {len(new_tasks)} 题")

    # 合并并输出
    merged = existing_tasks + all_new_tasks
    print(f"\n总计: {len(merged)} 题")

    output_path = tasks_file.replace(".json", "_generated.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tasks": merged}, f, ensure_ascii=False, indent=2)
    print(f"\n生成结果已写入: {output_path}")
    print("请人工审核后替换 tasks.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: 运行生成脚本

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python scripts/generate_frontcode_tasks.py
```

### Step 3: 人工审核并更新 tasks.json

审核要点:
- prompt 描述是否清晰无歧义
- keywords 是否覆盖核心知识点（每题至少 4 个）
- test_cases 是否可验证（至少 3 个）
- difficulty 标注是否合理
- 题目之间无重复

审核通过后:

```bash
cp benchmark/datasets/frontcode/tasks.json benchmark/datasets/frontcode/tasks.json.bak
cp benchmark/datasets/frontcode/tasks_generated.json benchmark/datasets/frontcode/tasks.json
```

### Step 4: 验证题目可加载

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -c "
from benchmark.adapters.frontcode_adapter import FrontCodeAdapter
tasks = FrontCodeAdapter().load()
print(f'Loaded {len(tasks)} tasks')
for t in tasks:
    print(f'  {t.task_id}: {t.metadata[\"type\"]} keywords={len(t.metadata[\"keywords\"])} test_cases={len(t.test_cases)}')
"
```

---

## Task 2: FrontCodeAdapter 支持 test_cases

**Files:**
- Modify: `benchmark/adapters/frontcode_adapter.py`
- Test: `tests/test_frontcode_adapter.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_frontcode_adapter.py
import json
import tempfile
import os

from benchmark.adapters.frontcode_adapter import FrontCodeAdapter


class TestFrontCodeAdapter:
    def _write_tasks(self, tasks: list[dict]) -> str:
        """将 tasks 写入临时文件，返回目录路径。"""
        tmpdir = tempfile.mkdtemp()
        tasks_file = os.path.join(tmpdir, "tasks.json")
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump({"tasks": tasks}, f, ensure_ascii=False)
        return tmpdir

    def test_load_with_test_cases(self):
        tasks_data = [
            {
                "id": "fc_test_1",
                "type": "javascript",
                "prompt": "实现 debounce 函数",
                "keywords": ["setTimeout", "clearTimeout"],
                "test_cases": [
                    "typeof debounce === 'function'",
                    "debounce 返回一个函数",
                ],
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert len(result) == 1
        assert result[0].test_cases == tasks_data[0]["test_cases"]

    def test_load_without_test_cases(self):
        """向后兼容: 没有 test_cases 字段时不报错。"""
        tasks_data = [
            {
                "id": "fc_test_2",
                "type": "html",
                "prompt": "创建 header 元素",
                "keywords": ["header", "nav"],
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert len(result) == 1
        assert result[0].test_cases == []

    def test_load_with_difficulty_in_metadata(self):
        tasks_data = [
            {
                "id": "fc_test_3",
                "type": "css",
                "prompt": "实现响应式布局",
                "keywords": ["media", "flex"],
                "test_cases": ["存在 @media 查询"],
                "difficulty": "medium",
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert result[0].metadata["difficulty"] == "medium"

    def test_validate_with_test_cases(self):
        tasks_data = [
            {
                "id": "fc_test_4",
                "type": "javascript",
                "prompt": "实现 throttle",
                "keywords": ["setTimeout", "Date"],
                "test_cases": ["typeof throttle === 'function'"],
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert adapter.validate(result[0]) is True
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontcode_adapter.py -v
```

- [ ] **Step 3: 修改 FrontCodeAdapter**

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

        try:
            with open(tasks_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in FrontCode tasks file: {tasks_file}. "
                f"Parse error: {e}"
            ) from e

        tasks = []
        for idx, item in enumerate(data.get("tasks", [])):
            required_fields = ["id", "type", "prompt", "keywords"]
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                raise ValueError(
                    f"Task at index {idx} missing required fields: {missing_fields}"
                )

            if not isinstance(item["keywords"], list):
                raise ValueError(
                    f"Task {item['id']}: 'keywords' must be a list, got {type(item['keywords']).__name__}"
                )
            if not item["keywords"]:
                raise ValueError(
                    f"Task {item['id']}: 'keywords' cannot be empty"
                )

            # 构建 metadata，保留额外字段（difficulty, source 等）
            metadata = {
                "type": item["type"],
                "keywords": item["keywords"],
                "source": "frontcode",
            }
            # 透传可选字段
            for key in ("difficulty", "difficulty_level"):
                if key in item:
                    metadata[key] = item[key]

            task = TaskDefinition(
                task_id=item["id"],
                dimension="frontend-dev",
                dataset="frontcode",
                prompt=build_structured_prompt(item["prompt"], "frontend-dev"),
                expected_output="",
                test_cases=item.get("test_cases", []),
                metadata=metadata,
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式.

        验证要求:
        - task_id 非空
        - prompt 非空
        - dimension 为 frontend-dev
        - metadata 中存在 keywords 且为非空列表
        """
        keywords = task.metadata.get("keywords")
        return bool(
            task.task_id
            and task.prompt
            and task.dimension == "frontend-dev"
            and isinstance(keywords, list)
            and keywords
        )

    def get_dimension(self) -> str:
        return "frontend-dev"
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_frontcode_adapter.py -v
```

- [ ] **Step 5: 提交**

```bash
git add benchmark/adapters/frontcode_adapter.py tests/test_frontcode_adapter.py
git commit -m "feat(adapter): FrontCodeAdapter 支持 test_cases + difficulty 透传"
```

---

## Task 3: 报告 - 雷达图

**Files:**
- Modify: `benchmark/core/reporter.py`
- Modify: `benchmark/templates/report.html`
- Test: `tests/test_reporter.py`

### 设计决策

使用纯 SVG 生成雷达图，不引入 matplotlib 依赖。理由:
- 报告是 HTML，SVG 直接嵌入，不需要额外图片生成
- 零额外依赖
- SVG 可缩放、可样式化

### 维度轴定义

```python
# radar chart 的维度轴配置
_DIMENSION_AXES = {
    "backend-dev": [
        ("correctness", "正确性"),
        ("test_coverage", "测试覆盖"),
        ("performance", "性能"),
        ("code_style", "风格"),
        ("robustness", "健壮性"),
        ("architecture", "架构"),
        ("security", "安全"),
        ("extensibility", "可扩展性"),
    ],
    "frontend-dev": [
        ("functionality", "功能"),
        ("html_semantic", "语义化"),
        ("accessibility", "可访问性"),
        ("css_quality", "CSS质量"),
        ("code_organization", "代码组织"),
        ("performance", "性能"),
        ("browser_compat", "兼容性"),
    ],
    "reasoning": [
        ("answer_correctness", "答案正确性"),
        ("reasoning_completeness", "完整性"),
        ("reasoning_validity", "推理正确性"),
        ("method_elegance", "方法优雅度"),
        ("difficulty_adaptation", "难度适配"),
    ],
    "system-architecture": [
        ("answer_correctness", "答案正确性"),
        ("reasoning_completeness", "推理完整性"),
        ("option_analysis", "选项分析"),
        ("reasoning_confidence", "推理置信度"),
        ("subject_adaptation", "学科适配"),
    ],
}
```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_reporter.py
import math

from benchmark.core.reporter import _build_radar_svg, _extract_dimension_scores


class TestExtractDimensionScores:
    def test_backend_scores(self):
        rows = [
            {
                "model": "test-model",
                "dimension": "backend-dev",
                "details": json.dumps({
                    "composite": {
                        "weights": {"correctness": 0.3, "test_coverage": 0.2, "performance": 0.15},
                        "scores": {"correctness": 90.0, "test_coverage": 80.0, "performance": 70.0},
                    },
                    "correctness": {"method": "exact_match", "passed": True},
                }),
            }
        ]
        result = _extract_dimension_scores(rows, "backend-dev", "test-model")
        assert result["correctness"] == 90.0
        assert result["test_coverage"] == 80.0
        assert result["performance"] == 70.0

    def test_missing_details(self):
        rows = [
            {
                "model": "test-model",
                "dimension": "backend-dev",
                "details": "",
            }
        ]
        result = _extract_dimension_scores(rows, "backend-dev", "test-model")
        assert result == {}

    def test_no_composite_key(self):
        rows = [
            {
                "model": "test-model",
                "dimension": "backend-dev",
                "details": json.dumps({"some_key": "some_val"}),
            }
        ]
        result = _extract_dimension_scores(rows, "backend-dev", "test-model")
        assert result == {}


class TestBuildRadarSvg:
    def test_basic_svg_output(self):
        scores = {"correctness": 80, "test_coverage": 60, "performance": 90}
        axes = [
            ("correctness", "正确性"),
            ("test_coverage", "测试覆盖"),
            ("performance", "性能"),
        ]
        svg = _build_radar_svg(scores, axes)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "正确性" in svg

    def test_empty_scores(self):
        svg = _build_radar_svg({}, [("a", "A"), ("b", "B")])
        assert "<svg" in svg

    def test_polygon_points(self):
        scores = {"a": 100, "b": 0}
        axes = [("a", "A"), ("b", "B")]
        svg = _build_radar_svg(scores, axes, width=200, height=200)
        # 验证 SVG 中包含 polygon 元素
        assert "<polygon" in svg

    def test_single_axis(self):
        svg = _build_radar_svg({"a": 50}, [("a", "A")])
        assert "<svg" in svg
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reporter.py -v
```

- [ ] **Step 3: 实现 _extract_dimension_scores**

添加到 `benchmark/core/reporter.py`:

```python
# benchmark/core/reporter.py 新增函数

import json
import math
from typing import Any

_DIMENSION_AXES = {
    "backend-dev": [
        ("correctness", "正确性"),
        ("test_coverage", "测试覆盖"),
        ("performance", "性能"),
        ("code_style", "风格"),
        ("robustness", "健壮性"),
        ("architecture", "架构"),
        ("security", "安全"),
        ("extensibility", "可扩展性"),
    ],
    "frontend-dev": [
        ("functionality", "功能"),
        ("html_semantic", "语义化"),
        ("accessibility", "可访问性"),
        ("css_quality", "CSS质量"),
        ("code_organization", "代码组织"),
        ("performance", "性能"),
        ("browser_compat", "兼容性"),
    ],
    "reasoning": [
        ("answer_correctness", "答案正确性"),
        ("reasoning_completeness", "完整性"),
        ("reasoning_validity", "推理正确性"),
        ("method_elegance", "方法优雅度"),
        ("difficulty_adaptation", "难度适配"),
    ],
    "system-architecture": [
        ("answer_correctness", "答案正确性"),
        ("reasoning_completeness", "推理完整性"),
        ("option_analysis", "选项分析"),
        ("reasoning_confidence", "推理置信度"),
        ("subject_adaptation", "学科适配"),
    ],
}


def _extract_dimension_scores(
    rows: list[dict], dimension: str, model: str
) -> dict[str, float]:
    """从评测结果中提取指定模型在指定维度的子分数。

    Args:
        rows: 数据库查询结果（需包含 details 字段）
        dimension: 评测维度
        model: 模型名称

    Returns:
        {子维度名: 平均分} 的映射
    """
    model_dim_rows = [r for r in rows if r["model"] == model and r["dimension"] == dimension]
    if not model_dim_rows:
        return {}

    # 收集所有 composite.scores，计算各子维度的平均分
    all_sub_scores: dict[str, list[float]] = {}
    for row in model_dim_rows:
        details_str = row.get("details", "")
        if not details_str:
            continue
        try:
            details = json.loads(details_str) if isinstance(details_str, str) else details_str
        except (json.JSONDecodeError, TypeError):
            continue

        composite = details.get("composite", {})
        sub_scores = composite.get("scores", {})
        for key, val in sub_scores.items():
            if isinstance(val, (int, float)):
                all_sub_scores.setdefault(key, []).append(float(val))

    if not all_sub_scores:
        return {}

    return {k: sum(v) / len(v) for k, v in all_sub_scores.items()}
```

- [ ] **Step 4: 实现 _build_radar_svg**

```python
# benchmark/core/reporter.py 新增函数

def _build_radar_svg(
    scores: dict[str, float],
    axes: list[tuple[str, str]],
    width: int = 400,
    height: int = 400,
    radius: int = 140,
) -> str:
    """生成雷达图 SVG 字符串。

    Args:
        scores: {子维度key: 分数} 的映射
        axes: [(key, label), ...] 维度轴定义
        width: SVG 宽度
        height: SVG 高度
        radius: 雷达图半径

    Returns:
        SVG 字符串
    """
    n = len(axes)
    if n < 3:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle">数据不足（需要至少 3 个维度）</text></svg>'

    cx, cy = width / 2, height / 2

    def _point(index: int, value: float) -> tuple[float, float]:
        """计算第 index 个轴上 value (0-100) 对应的 SVG 坐标。"""
        angle = 2 * math.pi * index / n - math.pi / 2
        r = radius * (value / 100.0)
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    # 背景网格（3 层: 33%, 66%, 100%）
    grid_levels = [0.33, 0.66, 1.0]
    grid_paths = []
    for level in grid_levels:
        points = []
        for i in range(n):
            x, y = _point(i, level * 100)
            points.append(f"{x:.1f},{y:.1f}")
        grid_paths.append(
            f'<polygon points="{" ".join(points)}" '
            f'fill="none" stroke="#e5e7eb" stroke-width="1"/>'
        )

    # 轴线
    axis_lines = []
    for i in range(n):
        x, y = _point(i, 100)
        axis_lines.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1"/>')

    # 数据多边形
    if scores:
        data_points = []
        for i, (key, label) in enumerate(axes):
            val = scores.get(key, 0)
            x, y = _point(i, val)
            data_points.append(f"{x:.1f},{y:.1f}")
        data_polygon = (
            f'<polygon points="{" ".join(data_points)}" '
            f'fill="rgba(59,130,246,0.2)" stroke="#3b82f6" stroke-width="2"/>'
        )
    else:
        data_polygon = ""

    # 数据点圆点
    data_dots = []
    if scores:
        for i, (key, label) in enumerate(axes):
            val = scores.get(key, 0)
            x, y = _point(i, val)
            data_dots.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#3b82f6"/>'
            )

    # 标签
    labels = []
    for i, (key, label) in enumerate(axes):
        x, y = _point(i, 115)  # 放在轴外侧
        val = scores.get(key, 0) if scores else 0
        # 根据位置调整对齐
        anchor = "middle"
        dx, dy = 0, 0
        angle = 2 * math.pi * i / n - math.pi / 2
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        if cos_a > 0.3:
            anchor = "start"
            dx = 5
        elif cos_a < -0.3:
            anchor = "end"
            dx = -5
        if sin_a < -0.3:
            dy = -5
        elif sin_a > 0.3:
            dy = 10
        labels.append(
            f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
            f'font-size="11" fill="#374151">{label}</text>'
        )
        # 分数标注
        fx, fy = _point(i, val)
        labels.append(
            f'<text x="{fx:.1f}" y="{fy - 10:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#3b82f6" font-weight="bold">{val:.0f}</text>'
        )

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        *grid_paths,
        *axis_lines,
        data_polygon,
        *data_dots,
        *labels,
        "</svg>",
    ]
    return "\n".join(parts)
```

- [ ] **Step 5: 修改 Database.get_results 返回 details 字段**

`reporter.py` 需要从数据库获取 `details` 字段，当前 `get_results` 不包含它。在 reporter.py 中改用 `get_result_detail` 或直接写 SQL。

在 `benchmark/core/reporter.py` 的 `generate_html_report` 中，替换数据查询逻辑:

```python
# 替换原来的 db.get_results() 调用
# 在 generate_html_report 函数中:

def _query_results(db: Database, models=None, dimensions=None, date_range=None) -> list[dict]:
    """查询评测结果，包含 details 字段。"""
    conn = db._get_conn()
    query = """
        SELECT r.result_id, e.model, e.dimension,
               r.task_id, r.final_score, r.passed,
               r.execution_time, r.created_at,
               r.details
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        WHERE 1=1
    """
    params: list = []
    if models:
        placeholders = ",".join("?" for _ in models)
        query += f" AND e.model IN ({placeholders})"
        params.extend(models)
    if dimensions:
        placeholders = ",".join("?" for _ in dimensions)
        query += f" AND e.dimension IN ({placeholders})"
        params.extend(dimensions)
    if date_range:
        start, end = date_range
        query += " AND r.created_at >= ? AND r.created_at <= ?"
        params.extend([start, end + " 23:59:59"])
    query += " ORDER BY r.created_at DESC"

    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

- [ ] **Step 6: 修改 generate_html_report 集成雷达图**

在 `generate_html_report` 中新增雷达图数据生成:

```python
# 在 generate_html_report 函数中，构建 score_table 之后:

# 构建雷达图数据: {model: {dimension: svg_string}}
radar_charts: dict[str, dict[str, str]] = {}
for model in model_list:
    radar_charts[model] = {}
    for dim in dim_list:
        axes = _DIMENSION_AXES.get(dim, [])
        if not axes:
            continue
        scores = _extract_dimension_scores(rows, dim, model)
        if scores:
            svg = _build_radar_svg(scores, axes)
            radar_charts[model][dim] = svg
```

将 `radar_charts` 传入模板渲染:

```python
html = template.render(
    generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    models=model_list,
    dimensions=dim_list,
    date_range=date_range if date_range else "All time",
    score_table=score_table,
    stat_tests=stat_tests,
    detailed=detailed,
    radar_charts=radar_charts,
)
```

- [ ] **Step 7: 更新 report.html 模板**

在 `benchmark/templates/report.html` 中，Score Summary 之后添加雷达图 section:

```html
<div class="section">
    <h2>3. Dimension Radar Charts</h2>
    {% for model in models %}
    {% if radar_charts[model] %}
    <h3>{{ model }}</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 24px;">
        {% for dim, svg in radar_charts[model].items() %}
        <div style="text-align: center;">
            <p style="font-weight: 600; margin-bottom: 4px;">{{ dim }}</p>
            {{ svg | safe }}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    {% endfor %}
</div>
```

同时将 Statistical Tests 的 section 编号从 3 改为 4，Detailed Results 从 4 改为 5。

- [ ] **Step 8: 运行测试验证通过**

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reporter.py -v
```

- [ ] **Step 9: 提交**

```bash
git add benchmark/core/reporter.py benchmark/templates/report.html tests/test_reporter.py
git commit -m "feat(report): 雷达图 + 维度子分数提取"
```

---

## Task 4: 报告 - 维度分数表格

**Files:**
- Modify: `benchmark/core/reporter.py`
- Modify: `benchmark/templates/report.html`
- Test: `tests/test_reporter.py`（追加）

### 设计

在 Score Summary 表格中，每个维度列展开为多个子分数列。数据来源是 `details` JSON 中的 `composite.scores`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_reporter.py

from benchmark.core.reporter import _build_dimension_score_table


class TestBuildDimensionScoreTable:
    def test_basic(self):
        rows = [
            {
                "model": "model-a",
                "dimension": "backend-dev",
                "final_score": 82.5,
                "passed": 1,
                "details": json.dumps({
                    "composite": {
                        "weights": {"correctness": 0.3, "performance": 0.15, "code_style": 0.15},
                        "scores": {"correctness": 90.0, "performance": 75.0, "code_style": 85.0},
                    },
                }),
            },
            {
                "model": "model-a",
                "dimension": "backend-dev",
                "final_score": 70.0,
                "passed": 0,
                "details": json.dumps({
                    "composite": {
                        "weights": {"correctness": 0.3, "performance": 0.15, "code_style": 0.15},
                        "scores": {"correctness": 80.0, "performance": 60.0, "code_style": 70.0},
                    },
                }),
            },
        ]
        result = _build_dimension_score_table(rows)
        # 应包含 model-a 的 backend-dev 维度子分数
        assert "model-a" in result
        assert "backend-dev" in result["model-a"]
        sub = result["model-a"]["backend-dev"]
        # correctness 平均分 = (90+80)/2 = 85
        assert abs(sub["correctness"] - 85.0) < 0.1

    def test_empty_rows(self):
        result = _build_dimension_score_table([])
        assert result == {}

    def test_missing_details(self):
        rows = [
            {"model": "m", "dimension": "backend-dev", "final_score": 80, "passed": 1, "details": ""},
        ]
        result = _build_dimension_score_table(rows)
        assert result["m"]["backend-dev"] == {}
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reporter.py::TestBuildDimensionScoreTable -v
```

- [ ] **Step 3: 实现 _build_dimension_score_table**

```python
# benchmark/core/reporter.py 新增函数

def _build_dimension_score_table(
    rows: list[dict],
) -> dict[str, dict[str, dict[str, float]]]:
    """构建维度子分数表。

    Returns:
        {model: {dimension: {sub_dimension: avg_score}}}
    """
    result: dict[str, dict[str, dict[str, float]]] = {}

    for row in rows:
        model = row.get("model", "")
        dimension = row.get("dimension", "")
        details_str = row.get("details", "")

        if not details_str:
            continue

        try:
            details = json.loads(details_str) if isinstance(details_str, str) else details_str
        except (json.JSONDecodeError, TypeError):
            continue

        composite = details.get("composite", {})
        sub_scores = composite.get("scores", {})
        if not sub_scores:
            continue

        result.setdefault(model, {}).setdefault(dimension, {})
        for key, val in sub_scores.items():
            if isinstance(val, (int, float)):
                # 累加用于后续平均
                bucket = result[model][dimension]
                if key not in bucket:
                    bucket[key] = {"sum": 0.0, "count": 0}
                bucket[key]["sum"] += float(val)
                bucket[key]["count"] += 1

    # 计算平均值
    for model_dims in result.values():
        for dim_scores in model_dims.values():
            for key in list(dim_scores.keys()):
                bucket = dim_scores[key]
                dim_scores[key] = bucket["sum"] / bucket["count"] if bucket["count"] > 0 else 0.0

    return result
```

- [ ] **Step 4: 在 generate_html_report 中集成**

```python
# 在 generate_html_report 函数中，radar_charts 之后:

# 构建维度子分数表
dim_score_table = _build_dimension_score_table(rows)
```

传入模板:

```python
html = template.render(
    generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    models=model_list,
    dimensions=dim_list,
    date_range=date_range if date_range else "All time",
    score_table=score_table,
    stat_tests=stat_tests,
    detailed=detailed,
    radar_charts=radar_charts,
    dim_score_table=dim_score_table,
)
```

- [ ] **Step 5: 更新 report.html 模板**

在 Score Summary 表格之后、雷达图之前，添加维度子分数详情表格:

```html
<div class="section">
    <h2>3. Dimension Sub-Scores</h2>
    {% for dim in dimensions %}
    {% set dim_axes = dimension_axes.get(dim, []) %}
    {% if dim_axes and dim_score_table %}
    <h3>{{ dim }}</h3>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                {% for key, label in dim_axes %}
                <th>{{ label }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for model in models %}
            {% if dim_score_table.get(model, {}).get(dim) %}
            <tr>
                <td>{{ model }}</td>
                {% for key, label in dim_axes %}
                <td>{{ dim_score_table[model][dim].get(key, "-") | round(1) if dim_score_table[model][dim].get(key) is not none else "-" }}</td>
                {% endfor %}
            </tr>
            {% endif %}
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    {% endfor %}
</div>
```

在 `generate_html_report` 中传入 `dimension_axes`:

```python
html = template.render(
    ...
    dimension_axes=_DIMENSION_AXES,
    ...
)
```

调整 section 编号: Dimension Sub-Scores 为 3，Radar Charts 为 4，Statistical Tests 为 5，Detailed Results 为 6。

- [ ] **Step 6: 运行全部测试**

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_reporter.py -v
```

- [ ] **Step 7: 提交**

```bash
git add benchmark/core/reporter.py benchmark/templates/report.html tests/test_reporter.py
git commit -m "feat(report): 维度子分数表格"
```

---

## Task 5: 集成验证

- [ ] 运行全部已有测试确保无回归

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/ -v --timeout=30
```

- [ ] 验证 FrontCodeAdapter 加载扩展后的题目

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -c "
from benchmark.adapters.frontcode_adapter import FrontCodeAdapter
adapter = FrontCodeAdapter()
tasks = adapter.load()
print(f'Total tasks: {len(tasks)}')
assert len(tasks) >= 17, f'Expected >= 17, got {len(tasks)}'
for t in tasks:
    assert adapter.validate(t), f'Validation failed for {t.task_id}'
    if t.test_cases:
        print(f'  {t.task_id}: test_cases={len(t.test_cases)}')
print('All tasks validated OK')
"
```

- [ ] 验证报告生成（用 mock 数据）

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -c "
from benchmark.core.reporter import _build_radar_svg, _extract_dimension_scores, _build_dimension_score_table, _DIMENSION_AXES
import json

# 模拟数据
rows = [
    {
        'model': 'zai/glm-5.1',
        'dimension': 'backend-dev',
        'details': json.dumps({
            'composite': {
                'weights': {'correctness': 0.3, 'code_style': 0.15, 'performance': 0.15, 'robustness': 0.1, 'architecture': 0.05, 'security': 0.03, 'extensibility': 0.02, 'test_coverage': 0.2},
                'scores': {'correctness': 85, 'code_style': 90, 'performance': 70, 'robustness': 80, 'architecture': 75, 'security': 95, 'extensibility': 88, 'test_coverage': 60},
            },
        }),
    },
]

scores = _extract_dimension_scores(rows, 'backend-dev', 'zai/glm-5.1')
print(f'Sub-scores: {scores}')

axes = _DIMENSION_AXES['backend-dev']
svg = _build_radar_svg(scores, axes)
assert '<svg' in svg
print(f'SVG generated: {len(svg)} chars')

dim_table = _build_dimension_score_table(rows)
print(f'Dimension score table: {dim_table}')
print('Integration check OK')
"
```

---

## 文件清单

新增文件:
```
scripts/generate_frontcode_tasks.py
tests/test_frontcode_adapter.py
tests/test_reporter.py
```

修改文件:
```
benchmark/adapters/frontcode_adapter.py      # test_cases + difficulty 透传
benchmark/datasets/frontcode/tasks.json        # 题目扩展到 17 题
benchmark/core/reporter.py                     # 雷达图 + 维度子分数
benchmark/templates/report.html                # 雷达图 + 子分数表格
```

## 依赖

无新增 Python 依赖。雷达图使用纯 SVG 生成。

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| LLM 生成的题目质量不稳定 | 生成后必须人工审核，脚本输出到 _generated.json 不直接覆盖 |
| SVG 雷达图在边缘情况下渲染异常 | 测试覆盖: 空数据、单维度、极端分数值 |
| 数据库 details 字段为空或格式不一致 | _extract_dimension_scores 逐行 try/except，缺失时跳过 |
| get_results 不返回 details 字段 | 新增 _query_results 函数直接查询，不修改 Database 类接口 |
