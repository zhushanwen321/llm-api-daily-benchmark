# Stage 1: MVP 核心评测能力 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 LLM Benchmark 的最小可用版本，支持 reasoning 和 backend-dev 两个维度的手动评测，结果持久化到 SQLite 并通过 Streamlit 展示。

**Architecture:** 纯 Python 实现，使用 Pydantic 做数据验证、SQLite 做结果存储、Streamlit 做 Web 可视化。数据集适配器和评分器通过抽象基类解耦，CLI 使用 click 框架。每个评测维度只选最难的 5 道题目（GSM8K 按解答步骤数排序，BigCodeBench 使用官方 Hard 子集）。

**Tech Stack:** Python 3.11+, pydantic, click, rich, streamlit, datasets (HuggingFace), pyyaml, requests, sqlite3

---

## Spec Reference

**Spec 文件**: `.sisyphus/specs/llm-benchmark-stage1.md`

**范围**:
- 2 个维度：reasoning (GSM8K) + backend-dev (BigCodeBench-Hard)
- 每个维度 5 道最难的题目
- 手动评测，无定时调度
- 无 unit tests，Agent-Executed QA only

---

## File Structure

```
llm-api-daily-benchmark/
├── pyproject.toml                    # 项目依赖和元数据
├── .gitignore                        # 排除敏感文件
│
├── benchmark/
│   ├── __init__.py                   # 包初始化
│   ├── __main__.py                   # python -m benchmark 入口
│   ├── cli.py                        # CLI 命令（click）
│   ├── config.py                     # 配置加载
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                   # DatasetAdapter 抽象基类
│   │   ├── gsm8k_adapter.py          # GSM8K 适配器
│   │   └── bigcodebench_adapter.py   # BigCodeBench 适配器
│   │
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseScorer + ScoreResult
│   │   ├── exact_match_scorer.py     # 精确匹配评分器
│   │   └── execution_scorer.py       # 执行验证评分器
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py               # Pydantic 数据模型
│   │   └── database.py              # SQLite 操作
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── llm_adapter.py           # LLM API 调用适配
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── app.py                   # Streamlit 应用
│   │
│   ├── configs/
│   │   ├── default.yaml             # 默认配置
│   │   └── models.yaml.example      # 模型配置模板
│   │
│   ├── data/                        # SQLite 数据库（.gitignore）
│   └── logs/                        # 日志文件（.gitignore）
```

---

## Execution Waves

```
Wave 1 (Foundation — 3 tasks, parallel):
├── Task 1: Project scaffolding + pyproject.toml
├── Task 2: Data models (schemas.py)
└── Task 3: Config system (config.py + YAML)

Wave 2 (Base classes + Storage — 3 tasks, parallel):
├── Task 4: Dataset adapter base class
├── Task 5: Scorer base class + ScoreResult
└── Task 6: SQLite database

Wave 3 (Adapters + Scorers + LLM — 5 tasks, parallel):
├── Task 7: GSM8K adapter (depends: 4)
├── Task 8: ExactMatchScorer (depends: 5)
├── Task 9: BigCodeBench adapter (depends: 4)
├── Task 10: ExecutionScorer (depends: 5)
└── Task 11: LLM Adapter (depends: 3)

Wave 4 (Integration — 2 tasks):
├── Task 12: CLI commands (depends: all Wave 3)
└── Task 13: Streamlit interface (depends: 6)

Wave FINAL (Verification — 4 parallel reviews):
├── F1: Plan compliance audit
├── F2: Code quality review
├── F3: Real manual QA
└── F4: Scope fidelity check
```

---

## TODOs

- [ ] 1. Project scaffolding + pyproject.toml + .gitignore

  **What to do**:
  - 创建 `pyproject.toml`，定义项目元数据和所有 Stage 1 依赖
  - 创建 `.gitignore`，排除敏感文件和缓存目录
  - 创建 `benchmark/` 目录结构：`adapters/`, `scorers/`, `models/`, `core/`, `visualization/`, `configs/`, `data/`, `datasets/`, `logs/`
  - 创建所有 `__init__.py` 文件
  - 创建 `benchmark/__main__.py`（`python -m benchmark` 入口）
  - 创建空文件占位（后续任务填充）

  **Must NOT do**:
  - 不要实现任何业务逻辑
  - 不要创建测试文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4-13
  - **Blocked By**: None

  **References**:
  - `.sisyphus/specs/llm-benchmark-stage1.md` — "File Structure" 和 "依赖列表" 章节

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 pyproject.toml**

  创建文件 `pyproject.toml`：

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
      "pandas>=2.0",
  ]

  [project.scripts]
  benchmark = "benchmark.cli:cli"

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
  ```

  - [ ] **Step 2: 创建 .gitignore**

  创建文件 `.gitignore`：

  ```
  # Python
  __pycache__/
  *.py[cod]
  *.egg-info/
  .eggs/
  dist/
  build/
  .venv/

  # 敏感配置（API keys）
  benchmark/configs/models.yaml

  # 数据和日志
  benchmark/data/*.db
  benchmark/datasets/*/
  benchmark/logs/*.log

  # IDE
  .idea/
  .vscode/
  *.swp

  # OS
  .DS_Store
  Thumbs.db
  ```

  - [ ] **Step 3: 创建目录结构和 __init__.py 文件**

  ```bash
  mkdir -p benchmark/{adapters,scorers,models,core,visualization,configs,data,datasets,logs}
  touch benchmark/__init__.py
  touch benchmark/adapters/__init__.py
  touch benchmark/scorers/__init__.py
  touch benchmark/models/__init__.py
  touch benchmark/core/__init__.py
  touch benchmark/visualization/__init__.py
  ```

  - [ ] **Step 4: 创建 benchmark/__main__.py**

  创建文件 `benchmark/__main__.py`：

  ```python
  from benchmark.cli import cli

  if __name__ == "__main__":
      cli()
  ```

  - [ ] **Step 5: 创建 models.yaml.example（模板文件，不含真实 key）**

  创建文件 `benchmark/configs/models.yaml.example`：

  ```yaml
  # 模型配置模板
  # 复制为 models.yaml 并填入真实 API key
  # models.yaml 已加入 .gitignore，不会被提交

  models:
    glm-4.7:
      provider: "glm"
      api_key: "YOUR_GLM_API_KEY_HERE"
      api_base: "https://open.bigmodel.cn/api/paas/v4/"
      max_tokens: 4096

    gpt-4:
      provider: "openai"
      api_key: "YOUR_OPENAI_API_KEY_HERE"
      api_base: "https://api.openai.com/v1/"
      max_tokens: 4096
  ```

  - [ ] **Step 6: 安装依赖并验证**

  ```bash
  uv sync
  python -c "import pydantic; import click; import rich; import streamlit; import datasets; import yaml; import requests; import pandas; print('All dependencies OK')"
  ```

  Expected: `All dependencies OK`

  - [ ] **Step 7: 验证 python -m benchmark 入口**

  ```bash
  python -m benchmark
  ```

  Expected: 显示 click 错误或帮助信息（因为 cli.py 尚未实现）

  - [ ] **Step 8: Commit**

  ```bash
  git add -A
  git commit -m "feat(benchmark): stage 1 scaffolding and dependencies"
  ```

  **QA Scenarios**:
  ```
  Scenario: 目录结构验证
    Tool: Bash
    Steps:
      1. test -d benchmark/adapters && test -d benchmark/scorers && test -d benchmark/models && test -d benchmark/core && test -d benchmark/visualization && test -d benchmark/configs && test -d benchmark/data && test -d benchmark/datasets
      2. test -f benchmark/__init__.py && test -f benchmark/__main__.py
    Expected: 所有目录和文件存在，退出码 0
    Evidence: .sisyphus/evidence/task-1-dir-structure.txt

  Scenario: 依赖安装验证
    Tool: Bash
    Steps:
      1. python -c "import pydantic, click, rich, streamlit, datasets, yaml, requests, pandas; print('OK')"
    Expected: OK
    Evidence: .sisyphus/evidence/task-1-deps.txt

  Scenario: .gitignore 排除 models.yaml
    Tool: Bash
    Steps:
      1. cp benchmark/configs/models.yaml.example benchmark/configs/models.yaml
      2. echo "test_key_123" >> benchmark/configs/models.yaml
      3. git status benchmark/configs/models.yaml
    Expected: 文件不在 git 跟踪中（被 .gitignore 排除）
    Evidence: .sisyphus/evidence/task-1-gitignore.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): stage 1 scaffolding and dependencies`
  - Pre-commit: `python -c "import pydantic, click, rich, streamlit, datasets, yaml, requests, pandas"`

---

- [ ] 2. Data models (schemas.py)

  **What to do**:
  - 创建 `benchmark/models/schemas.py`，定义所有 Pydantic 数据模型

  **Must NOT do**:
  - 不要实现数据库操作（Task 6 负责）
  - 不要添加 Stage 2/3 才需要的字段

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4-6
  - **Blocked By**: None

  **References**:
  - `.sisyphus/specs/llm-benchmark-stage1.md` — "7. Data Models" 章节

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 schemas.py**

  创建文件 `benchmark/models/schemas.py`：

  ```python
  """LLM Benchmark 数据模型定义。"""

  from __future__ import annotations

  from datetime import datetime
  from typing import Any

  from pydantic import BaseModel, Field


  class TaskDefinition(BaseModel):
      """评测题目定义。"""

      task_id: str
      dimension: str  # reasoning, backend-dev
      dataset: str  # gsm8k, bigcodebench
      prompt: str
      expected_output: str
      test_cases: list[str] = Field(default_factory=list)
      metadata: dict[str, Any] = Field(default_factory=dict)


  class ScoreResult(BaseModel):
      """单题评分结果。"""

      score: float  # 0-100
      passed: bool
      details: dict[str, Any] = Field(default_factory=dict)
      reasoning: str = ""


  class EvalRun(BaseModel):
      """一次评测运行的记录。"""

      run_id: str
      model: str
      dimension: str
      dataset: str
      started_at: datetime
      finished_at: datetime | None = None
      status: str  # running, completed, failed
      config_snapshot: str = "{}"


  class EvalResult(BaseModel):
      """单题评测结果。"""

      result_id: str
      run_id: str
      task_id: str
      task_content: str
      model_output: str
      functional_score: float
      quality_score: float = 0.0
      final_score: float
      passed: bool
      details: dict[str, Any] = Field(default_factory=dict)
      execution_time: float
      created_at: datetime
  ```

  - [ ] **Step 2: 验证模型可以实例化**

  ```bash
  python -c "
  from benchmark.models.schemas import TaskDefinition, ScoreResult, EvalRun, EvalResult
  from datetime import datetime

  td = TaskDefinition(task_id='test-001', dimension='reasoning', dataset='gsm8k', prompt='2+3=?', expected_output='5')
  assert td.task_id == 'test-001'

  sr = ScoreResult(score=100, passed=True, reasoning='Correct')
  assert sr.score == 100

  run = EvalRun(run_id='r1', model='glm-4.7', dimension='reasoning', dataset='gsm8k', started_at=datetime.now(), status='running')
  assert run.status == 'running'

  result = EvalResult(result_id='res1', run_id='r1', task_id='test-001', task_content='2+3=?', model_output='5', functional_score=100, final_score=100, passed=True, execution_time=1.2, created_at=datetime.now())
  assert result.final_score == 100

  print('All models validated OK')
  "
  ```

  Expected: `All models validated OK`

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/models/schemas.py
  git commit -m "feat(benchmark): data models — TaskDefinition, ScoreResult, EvalRun, EvalResult"
  ```

  **QA Scenarios**:
  ```
  Scenario: Pydantic 模型验证
    Tool: Bash
    Steps:
      1. python -c "from benchmark.models.schemas import TaskDefinition; t = TaskDefinition(task_id='x', dimension='reasoning', dataset='gsm8k', prompt='q', expected_output='a'); print(t.model_dump_json())"
    Expected: 输出合法 JSON
    Evidence: .sisyphus/evidence/task-2-models.txt

  Scenario: 必填字段缺失报错
    Tool: Bash
    Steps:
      1. python -c "from benchmark.models.schemas import TaskDefinition; TaskDefinition()" 2>&1
    Expected: 抛出 ValidationError，提示缺少必填字段
    Evidence: .sisyphus/evidence/task-2-validation-error.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): data models — TaskDefinition, ScoreResult, EvalRun, EvalResult`

---

- [ ] 3. Config system (config.py + YAML)

  **What to do**:
  - 创建 `benchmark/config.py`，实现配置加载逻辑
  - 创建 `benchmark/configs/default.yaml`，定义默认配置

  **Must NOT do**:
  - 不要实现 schedule.yaml（Stage 2）
  - 不要硬编码 API key

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 11 (LLM Adapter)
  - **Blocked By**: None

  **References**:
  - `.sisyphus/specs/llm-benchmark-stage1.md` — "10. 配置文件" 章节

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 default.yaml**

  创建文件 `benchmark/configs/default.yaml`：

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
      auto_weight: 0.8
      judge_weight: 0.2

    backend-dev:
      adapter: "bigcodebench"
      auto_weight: 0.8
      judge_weight: 0.2

  # 难度权重
  difficulty_weights:
    easy: 1.0
    medium: 1.5
    hard: 2.0
  ```

  - [ ] **Step 2: 创建 config.py**

  创建文件 `benchmark/config.py`：

  ```python
  """配置管理：从 YAML 文件加载配置。"""

  from __future__ import annotations

  from pathlib import Path
  from typing import Any

  import yaml


  _CONFIG_DIR = Path(__file__).parent / "configs"


  def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
      """加载默认配置，可被指定路径覆盖。

      Args:
          config_path: 配置文件路径。为 None 时使用 configs/default.yaml。

      Returns:
          配置字典。
      """
      path = Path(config_path) if config_path else _CONFIG_DIR / "default.yaml"
      if not path.exists():
          raise FileNotFoundError(f"Config file not found: {path}")
      with open(path) as f:
          return yaml.safe_load(f)


  def load_models_config(models_path: str | Path | None = None) -> dict[str, Any]:
      """加载模型 API 配置。

      Args:
          models_path: 模型配置路径。为 None 时使用 configs/models.yaml。

      Returns:
          模型配置字典，包含 models 键。
      """
      path = Path(models_path) if models_path else _CONFIG_DIR / "models.yaml"
      if not path.exists():
          raise FileNotFoundError(
              f"Models config not found: {path}. "
              "Copy configs/models.yaml.example to configs/models.yaml and fill in your API keys."
          )
      with open(path) as f:
          return yaml.safe_load(f)


  def get_model_config(model_name: str) -> dict[str, Any]:
      """获取指定模型的配置。

      Args:
          model_name: 模型名称（如 glm-4.7）。

      Returns:
          该模型的配置字典。

      Raises:
          ValueError: 模型未在配置中定义。
      """
      models_cfg = load_models_config()
      models = models_cfg.get("models", {})
      if model_name not in models:
          available = ", ".join(models.keys()) if models else "none"
          raise ValueError(f"Model '{model_name}' not found. Available: {available}")
      return models[model_name]
  ```

  - [ ] **Step 3: 验证配置加载**

  ```bash
  python -c "
  from benchmark.config import load_config
  cfg = load_config()
  assert cfg['model'] == 'glm-4.7'
  assert cfg['temperature'] == 0.0
  assert 'reasoning' in cfg['dimensions']
  assert 'backend-dev' in cfg['dimensions']
  print('Config loaded OK')
  print('Dimensions:', list(cfg['dimensions'].keys()))
  "
  ```

  Expected: `Config loaded OK` + 维度列表

  - [ ] **Step 4: 验证 models.yaml 加载（使用 example 文件）**

  ```bash
  python -c "
  from benchmark.config import load_models_config
  cfg = load_models_config('benchmark/configs/models.yaml.example')
  assert 'models' in cfg
  assert 'glm-4.7' in cfg['models']
  print('Models config loaded OK')
  print('Available models:', list(cfg['models'].keys()))
  "
  ```

  Expected: `Models config loaded OK` + 模型列表

  - [ ] **Step 5: Commit**

  ```bash
  git add benchmark/config.py benchmark/configs/default.yaml benchmark/configs/models.yaml.example
  git commit -m "feat(benchmark): config system with YAML loading"
  ```

  **QA Scenarios**:
  ```
  Scenario: 默认配置加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.config import load_config; c = load_config(); print(c['model'], c['temperature'])"
    Expected: glm-4.7 0.0
    Evidence: .sisyphus/evidence/task-3-default-config.txt

  Scenario: 模型配置缺失报错
    Tool: Bash
    Steps:
      1. python -c "from benchmark.config import get_model_config; get_model_config('nonexistent')" 2>&1
    Expected: ValueError，提示模型不存在
    Evidence: .sisyphus/evidence/task-3-model-not-found.txt

  Scenario: models.yaml 不存在时报错
    Tool: Bash
    Steps:
      1. mv benchmark/configs/models.yaml benchmark/configs/models.yaml.bak 2>/dev/null; true
      2. python -c "from benchmark.config import load_models_config; load_models_config()" 2>&1
      3. mv benchmark/configs/models.yaml.bak benchmark/configs/models.yaml 2>/dev/null; true
    Expected: FileNotFoundError，提示复制 example 文件
    Evidence: .sisyphus/evidence/task-3-models-missing.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): config system with YAML loading`

---

- [ ] 4. Dataset adapter base class

  **What to do**:
  - 创建 `benchmark/adapters/base.py`，定义 `DatasetAdapter` 抽象基类

  **Must NOT do**:
  - 不要实现任何具体数据集适配器

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: Task 2

  **References**:
  - `benchmark/models/schemas.py:TaskDefinition` — 题目定义模型

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/adapters/base.py**

  ```python
  """数据集适配器基类。

  所有数据集适配器必须继承此基类并实现 load/validate/get_dimension 方法。
  """

  from __future__ import annotations

  from abc import ABC, abstractmethod
  from typing import List

  from benchmark.models.schemas import TaskDefinition


  class DatasetAdapter(ABC):
      """数据集适配器抽象基类。

      子类必须实现：
      - load(): 从数据源加载任务列表
      - validate(): 验证单个任务格式是否合法
      - get_dimension(): 返回适配器对应的评测维度名称
      """

      @abstractmethod
      def load(self, path: str = "") -> List[TaskDefinition]:
          """加载数据集，返回任务定义列表。

          Args:
              path: 数据集路径。为空时使用默认缓存路径。

          Returns:
              TaskDefinition 列表。
          """

      @abstractmethod
      def validate(self, task: TaskDefinition) -> bool:
          """验证任务格式是否合法。

          Args:
              task: 待验证的任务定义。

          Returns:
              True 表示格式合法。
          """

      @abstractmethod
      def get_dimension(self) -> str:
          """返回此适配器对应的评测维度名称。"""
  ```

  - [ ] **Step 2: 验证抽象类不可实例化**

  ```bash
  python -c "
  from benchmark.adapters.base import DatasetAdapter
  try:
      DatasetAdapter()
      print('ERROR: should have raised TypeError')
  except TypeError as e:
      print(f'OK: {e}')
  "
  ```

  Expected: `OK: Can't instantiate abstract class...`

  - [ ] **Step 3: 验证最小子类可行**

  ```bash
  python -c "
  from benchmark.adapters.base import DatasetAdapter
  from benchmark.models.schemas import TaskDefinition

  class DummyAdapter(DatasetAdapter):
      def load(self, path=''): return []
      def validate(self, task): return True
      def get_dimension(self): return 'test'

  d = DummyAdapter()
  assert d.load() == []
  assert d.get_dimension() == 'test'
  print('Subclass works OK')
  "
  ```

  Expected: `Subclass works OK`

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/adapters/base.py
  git commit -m "feat(benchmark): dataset adapter abstract base class"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): dataset adapter abstract base class`

---

- [ ] 5. Scorer base class + ScoreResult

  **What to do**:
  - 创建 `benchmark/scorers/base.py`，定义 `BaseScorer` 抽象基类和 `ScoreResult`（从 schemas 导入）

  **Must NOT do**:
  - 不要实现具体评分器

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: Tasks 8, 10
  - **Blocked By**: Task 2

  **References**:
  - `benchmark/models/schemas.py:ScoreResult` — 评分结果模型

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/scorers/base.py**

  ```python
  """评分器基类。

  所有评分器必须继承 BaseScorer 并实现 score/get_metric_name 方法。
  评分结果统一使用 benchmark.models.schemas.ScoreResult。
  """

  from __future__ import annotations

  from abc import ABC, abstractmethod

  from benchmark.models.schemas import ScoreResult, TaskDefinition


  class BaseScorer(ABC):
      """评分器抽象基类。

      子类必须实现：
      - score(): 对模型输出进行评分
      - get_metric_name(): 返回指标名称
      """

      @abstractmethod
      def score(
          self,
          model_output: str,
          expected: str,
          task: TaskDefinition,
      ) -> ScoreResult:
          """对模型输出进行评分。

          Args:
              model_output: 模型生成的文本/代码。
              expected: 期望输出（答案或空字符串）。
              task: 原始任务定义。

          Returns:
              ScoreResult 包含分数、是否通过、详情、理由。
          """

      @abstractmethod
      def get_metric_name(self) -> str:
          """返回此评分器的指标名称（如 exact_match, execution）。"""
  ```

  - [ ] **Step 2: 验证抽象类不可实例化**

  ```bash
  python -c "
  from benchmark.scorers.base import BaseScorer
  try:
      BaseScorer()
      print('ERROR: should have raised TypeError')
  except TypeError as e:
      print(f'OK: {e}')
  "
  ```

  Expected: `OK: Can't instantiate abstract class...`

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/scorers/base.py
  git commit -m "feat(benchmark): scorer abstract base class"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): scorer abstract base class`

---

- [ ] 6. SQLite database

  **What to do**:
  - 创建 `benchmark/models/database.py`，实现 SQLite 初始化、表创建、CRUD 操作
  - 包含：创建运行记录、保存单题结果、查询结果

  **Must NOT do**:
  - 不要使用 ORM（保持轻量，只用 sqlite3 标准库）
  - 不要实现复杂查询（趋势图查询在 Stage 2）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Tasks 12, 13
  - **Blocked By**: Task 2

  **References**:
  - `benchmark/models/schemas.py:EvalRun, EvalResult` — 数据模型
  - `.sisyphus/specs/llm-benchmark-stage1.md` — "Database" 章节

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/models/database.py**

  ```python
  """SQLite 数据库操作。

  负责评测结果的持久化存储。使用 sqlite3 标准库，不依赖 ORM。
  """

  from __future__ import annotations

  import json
  import sqlite3
  from datetime import datetime
  from pathlib import Path
  from typing import Optional

  from benchmark.models.schemas import EvalResult, EvalRun


  class Database:
      """SQLite 数据库操作类。"""

      def __init__(self, db_path: str | Path = "benchmark/data/results.db") -> None:
          self.db_path = Path(db_path)
          self.db_path.parent.mkdir(parents=True, exist_ok=True)
          self._init_db()

      def _get_conn(self) -> sqlite3.Connection:
          return sqlite3.connect(str(self.db_path))

      def _init_db(self) -> None:
          """初始化数据库表（如不存在则创建）。"""
          conn = self._get_conn()
          cursor = conn.cursor()

          cursor.execute("""
              CREATE TABLE IF NOT EXISTS eval_runs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT UNIQUE NOT NULL,
                  model TEXT NOT NULL,
                  dimension TEXT NOT NULL,
                  dataset TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  status TEXT NOT NULL DEFAULT 'running',
                  config_snapshot TEXT
              )
          """)

          cursor.execute("""
              CREATE TABLE IF NOT EXISTS eval_results (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  result_id TEXT UNIQUE NOT NULL,
                  run_id TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  task_content TEXT,
                  model_output TEXT,
                  functional_score REAL NOT NULL DEFAULT 0,
                  quality_score REAL NOT NULL DEFAULT 0,
                  final_score REAL NOT NULL DEFAULT 0,
                  passed INTEGER NOT NULL DEFAULT 0,
                  details TEXT,
                  execution_time REAL NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
              )
          """)

          conn.commit()
          conn.close()

      def create_run(self, run: EvalRun) -> str:
          """创建评测运行记录。

          Args:
              run: EvalRun 实例。

          Returns:
              run_id 字符串。
          """
          conn = self._get_conn()
          conn.execute(
              """INSERT INTO eval_runs
                 (run_id, model, dimension, dataset, started_at, status, config_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (
                  run.run_id,
                  run.model,
                  run.dimension,
                  run.dataset,
                  run.started_at.isoformat(),
                  run.status,
                  getattr(run, "config_snapshot", None),
              ),
          )
          conn.commit()
          conn.close()
          return run.run_id

      def finish_run(self, run_id: str, status: str = "completed") -> None:
          """标记运行记录为已完成。

          Args:
              run_id: 运行记录 ID。
              status: 最终状态（completed / failed）。
          """
          conn = self._get_conn()
          conn.execute(
              "UPDATE eval_runs SET finished_at = ?, status = ? WHERE run_id = ?",
              (datetime.now().isoformat(), status, run_id),
          )
          conn.commit()
          conn.close()

      def save_result(self, result: EvalResult) -> str:
          """保存单题评测结果。

          Args:
              result: EvalResult 实例。

          Returns:
              result_id 字符串。
          """
          conn = self._get_conn()
          conn.execute(
              """INSERT INTO eval_results
                 (result_id, run_id, task_id, task_content, model_output,
                  functional_score, quality_score, final_score, passed,
                  details, execution_time, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (
                  result.result_id,
                  result.run_id,
                  result.task_id,
                  result.task_content,
                  result.model_output,
                  result.functional_score,
                  result.quality_score,
                  result.final_score,
                  int(result.passed),
                  json.dumps(result.details, ensure_ascii=False),
                  result.execution_time,
                  result.created_at.isoformat(),
              ),
          )
          conn.commit()
          conn.close()
          return result.result_id

      def get_results(
          self,
          model: Optional[str] = None,
          dimension: Optional[str] = None,
      ) -> list[dict]:
          """查询评测结果。

          Args:
              model: 按模型名过滤（可选）。
              dimension: 按维度过滤（可选）。

          Returns:
              字典列表，每个字典代表一行结果。
          """
          conn = self._get_conn()
          query = """
              SELECT r.result_id, e.model, e.dimension,
                     r.task_id, r.final_score, r.passed,
                     r.execution_time, r.created_at
              FROM eval_results r
              JOIN eval_runs e ON r.run_id = e.run_id
              WHERE 1=1
          """
          params: list = []
          if model:
              query += " AND e.model = ?"
              params.append(model)
          if dimension:
              query += " AND e.dimension = ?"
              params.append(dimension)
          query += " ORDER BY r.created_at DESC"

          cursor = conn.execute(query, params)
          columns = [desc[0] for desc in cursor.description]
          rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
          conn.close()
          return rows

      def get_result_detail(self, result_id: str) -> Optional[dict]:
          """获取单条结果的完整详情。

          Args:
              result_id: 结果 ID。

          Returns:
              结果字典，不存在时返回 None。
          """
          conn = self._get_conn()
          cursor = conn.execute(
              """SELECT r.*, e.model, e.dimension
                 FROM eval_results r
                 JOIN eval_runs e ON r.run_id = e.run_id
                 WHERE r.result_id = ?""",
              (result_id,),
          )
          columns = [desc[0] for desc in cursor.description]
          row = cursor.fetchone()
          conn.close()
          if row is None:
              return None
          return dict(zip(columns, row))
  ```

  - [ ] **Step 2: 验证数据库初始化**

  ```bash
  python -c "
  from benchmark.models.database import Database
  import os

  db = Database('benchmark/data/test_results.db')
  assert os.path.exists('benchmark/data/test_results.db'), 'DB file not created'

  import sqlite3
  conn = sqlite3.connect('benchmark/data/test_results.db')
  tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
  assert 'eval_runs' in tables, 'eval_runs table missing'
  assert 'eval_results' in tables, 'eval_results table missing'
  conn.close()

  os.remove('benchmark/data/test_results.db')
  print('Database init OK, tables created')
  "
  ```

  Expected: `Database init OK, tables created`

  - [ ] **Step 3: 验证 CRUD 操作**

  ```bash
  python -c "
  from benchmark.models.database import Database
  from benchmark.models.schemas import EvalRun, EvalResult
  from datetime import datetime
  import os

  db = Database('benchmark/data/test_results.db')

  run = EvalRun(run_id='test-run-1', model='glm-4.7', dimension='reasoning', dataset='gsm8k', started_at=datetime.now(), status='running')
  db.create_run(run)

  result = EvalResult(result_id='test-res-1', run_id='test-run-1', task_id='task-1', task_content='2+3=?', model_output='5', functional_score=100, final_score=100, passed=True, execution_time=1.2, created_at=datetime.now())
  db.save_result(result)

  db.finish_run('test-run-1', 'completed')

  results = db.get_results()
  assert len(results) == 1, f'Expected 1 result, got {len(results)}'
  assert results[0]['final_score'] == 100
  assert results[0]['model'] == 'glm-4.7'

  detail = db.get_result_detail('test-res-1')
  assert detail is not None
  assert detail['task_content'] == '2+3=?'

  os.remove('benchmark/data/test_results.db')
  print('CRUD operations OK')
  "
  ```

  Expected: `CRUD operations OK`

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/models/database.py
  git commit -m "feat(benchmark): SQLite database with CRUD operations"
  ```

  **QA Scenarios**:
  ```
  Scenario: 数据库表创建
    Tool: Bash
    Steps:
      1. python -c "from benchmark.models.database import Database; Database('benchmark/data/qa_test.db')"
      2. sqlite3 benchmark/data/qa_test.db ".tables"
      3. rm benchmark/data/qa_test.db
    Expected: 输出包含 eval_runs 和 eval_results
    Evidence: .sisyphus/evidence/task-6-tables.txt

  Scenario: 写入和读取
    Tool: Bash
    Steps:
      1. python -c "
  from benchmark.models.database import Database
  from benchmark.models.schemas import EvalRun, EvalResult
  from datetime import datetime
  db = Database('benchmark/data/qa_test.db')
  db.create_run(EvalRun(run_id='qa-1', model='test-model', dimension='reasoning', dataset='gsm8k', started_at=datetime.now(), status='running'))
  db.save_result(EvalResult(result_id='qr-1', run_id='qa-1', task_id='t1', task_content='test', model_output='42', functional_score=0, final_score=0, passed=False, execution_time=0.5, created_at=datetime.now()))
  db.finish_run('qa-1')
  rows = db.get_results(model='test-model')
  print(f'Results: {len(rows)}')
  assert len(rows) == 1
  print('PASS')
  "
      2. rm -f benchmark/data/qa_test.db
    Expected: Results: 1, PASS
    Evidence: .sisyphus/evidence/task-6-crud.txt

  Scenario: 空数据库查询
    Tool: Bash
    Steps:
      1. python -c "from benchmark.models.database import Database; db = Database('benchmark/data/qa_empty.db'); print(db.get_results())"
      2. rm -f benchmark/data/qa_empty.db
    Expected: 输出空列表 []
    Evidence: .sisyphus/evidence/task-6-empty-query.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): SQLite database with CRUD operations`

---

- [ ] 7. GSM8K adapter

  **What to do**:
  - 创建 `benchmark/adapters/gsm8k_adapter.py`
  - 从 HuggingFace 加载 GSM8K 数据集
  - 按解答步骤数降序排序，选前 5 题（最难的5道）
  - 解析答案中的 `#### 数字` 格式，提取最终数值

  **Must NOT do**:
  - 不要下载完整 GSM8K 数据集（只取前5题）
  - 不要实现评测逻辑（由 CLI 调用）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8-11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - `benchmark/models/schemas.py:TaskDefinition` — 数据模型
  - GSM8K HuggingFace: https://huggingface.co/datasets/openai/gsm8k

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/adapters/gsm8k_adapter.py**

  ```python
  """GSM8K 数据集适配器。加载最难的5道数学推理题。步骤数最多的题目被认为最难."""

  from __future__ import annotations

  import os
  import re
  from typing import List

  from datasets import load_dataset
  from benchmark.adapters.base import DatasetAdapter
  from benchmark.models.schemas import TaskDefinition


  class GSM8KAdapter(DatasetAdapter):
      """GSM8K 适配器，选择解答步骤最多的 5 道题."""

      def load(self, path: str = "") -> List[TaskDefinition]:
          """加载 GSM8K 最难的5题（按解答步骤数排序）."""
          cache_dir = path or os.path.join("benchmark", "datasets", "gsm8k")

          # 如果本地已有缓存，直接加载
          if os.path.exists(os.path.join(cache_dir, "dataset_info.json")):
              dataset = load_dataset(
                  "openai/gsm8k", "main", split="test",
                  cache_dir=cache_dir, download_mode="reuse_dataset_if_exists",
              )
          else:
              dataset = load_dataset(
                  "openai/gsm8k", "main", split="test",
                  cache_dir=cache_dir,
              )

          # 按解答步骤数排序（步骤数 = 换行数，越多越难）
          items_with_steps = []
          for item in dataset:
              # solution 字段包含思维链，步骤数约等于换行数
              steps = len(item["answer"].split("\n"))
              items_with_steps.append((item, steps))

          items_with_steps.sort(key=lambda x: x[1], reverse=True)
          hardest_5 = [item for item, _ in items_with_steps[:5]]

          tasks = []
          for idx, item in enumerate(hardest_5):
              # 从 answer 字段提取最终答案（格式：#### 42）
              answer_text = item["answer"]
              match = re.search(r"####\s*(-?\d+\.?\d*)", answer_text)
              expected = match.group(1).strip() if match else ""

              task = TaskDefinition(
                  task_id=f"gsm8k_hardest_{idx + 1}",
                  dimension="reasoning",
                  dataset="gsm8k",
                  prompt=item["question"],
                  expected_output=expected,
                  metadata={"difficulty": "hard", "source": "openai/gsm8k"},
              )
              tasks.append(task)

          return tasks

      def validate(self, task: TaskDefinition) -> bool:
          """验证任务格式。必须有 task_id, prompt, expected_output."""
          return bool(task.task_id and task.prompt and task.expected_output)

      def get_dimension(self) -> str:
          return "reasoning"
  ```

  - [ ] **Step 2: 验证 GSM8K 加载（需要网络，首次会下载数据集）**

  ```bash
  python -c "
  from benchmark.adapters.gsm8k_adapter import GSM8KAdapter
  a = GSM8KAdapter()
  tasks = a.load()
  print(f'Loaded {len(tasks)} tasks')
  for t in tasks:
      print(f'  {t.task_id}: prompt={t.prompt[:50]}... answer={t.expected_output}')
      assert a.validate(t), f'Validation failed for {t.task_id}'
  print('All tasks valid')
  "
  ```

  Expected: `Loaded 5 tasks` + 每题的 task_id 和答案

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/adapters/gsm8k_adapter.py
  git commit -m "feat(benchmark): GSM8K adapter — loads hardest 5 tasks by step count"
  ```

  **QA Scenarios**:
  ```
  Scenario: GSM8K 数据加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.gsm8k_adapter import GSM8KAdapter; a = GSM8KAdapter(); tasks = a.load(); print(len(tasks)); assert len(tasks) == 5; print('OK')"
    Expected: 5\nOK
    Evidence: .sisyphus/evidence/task-7-gsm8k-load.txt

  Scenario: 维证所有题目
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.gsm8k_adapter import GSM8KAdapter; a = GSM8KAdapter(); tasks = a.load(); assert all(a.validate(t) for t in tasks); print('All valid')"
    Expected: All valid
    Evidence: .sisyphus/evidence/task-7-gsm8k-validate.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): GSM8K adapter — loads hardest 5 tasks by step count`

---

- [ ] 8. ExactMatchScorer

  **What to do**:
  - 创建 `benchmark/scorers/exact_match_scorer.py`
  - 从模型输出中提取最后一个数字，与 expected_output 精确匹配
  - 匹配成功 score=100，失败 score=0

  **Must NOT do**:
  - 不要实现模糊匹配（Stage 1 只做精确匹配）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 9-11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:
  - `benchmark/scorers/base.py:BaseScorer` — 基类接口
  - `benchmark/models/schemas.py:ScoreResult` — 评分结果模型

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/scorers/exact_match_scorer.py**

  ```python
  """精确匹配评分器。从模型输出中提取数字，与期望答案精确比较."""

  from __future__ import annotations

  import re

  from benchmark.models.schemas import ScoreResult, TaskDefinition
  from benchmark.scorers.base import BaseScorer


  class ExactMatchScorer(BaseScorer):
      """精确匹配评分器，用于 reasoning 维度（GSM8K）。

      从模型输出中提取最后一个数字作为预测答案，
      与 expected_output 中的期望答案进行字符串精确比较。
      匹配成功 score=100，失败 score=0。
      """

      def score(
          self,
          model_output: str,
          expected: str,
          task: TaskDefinition,  # noqa: ARG002 — 基类接口要求
      ) -> ScoreResult:
          # 提取模型输出中的所有数字
          numbers = re.findall(r"-?\d+\.?\d*", model_output)

          if not numbers:
              return ScoreResult(
                  score=0,
                  passed=False,
                  details={"error": "No number found in output", "raw_output": model_output[:200]},
                  reasoning="Model output contains no numeric answer",
              )

          # 取最后一个数字作为预测答案
          predicted = numbers[-1]

          # 精确匹配（字符串比较）
          passed = predicted == expected.strip()
          score = 100.0 if passed else 0.0

          return ScoreResult(
              score=score,
              passed=passed,
              details={"predicted": predicted, "expected": expected.strip()},
              reasoning=(
                  f"Correct: predicted={predicted}"
                  if passed
                  else f"Incorrect: predicted={predicted}, expected={expected.strip()}"
              ),
          )

      def get_metric_name(self) -> str:
          return "exact_match"
  ```

  - [ ] **Step 2: 验证评分器**

  ```bash
  python -c "
  from benchmark.scorers.exact_match_scorer import ExactMatchScorer
  from benchmark.models.schemas import TaskDefinition

  s = ExactMatchScorer()
  task = TaskDefinition(task_id='t1', dimension='reasoning', dataset='gsm8k', prompt='2+3=?', expected_output='5')

  # 正确答案
  r1 = s.score('The answer is 5', '5', task)
  assert r1.passed and r1.score == 100, f'Expected pass, got {r1}'
  print(f'Correct: passed={r1.passed}, score={r1.score}')

  # 错误答案
  r2 = s.score('The answer is 6', '5', task)
  assert not r2.passed and r2.score == 0
  print(f'Wrong: passed={r2.passed}, score={r2.score}')

  # 无数字
  r3 = s.score('No numbers here!', '5', task)
  assert not r3.passed and r3.score == 0
  print(f'No number: passed={r3.passed}, score={r3.score}')

  print('All scorer tests passed')
  "
  ```

  Expected: `All scorer tests passed`

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/scorers/exact_match_scorer.py
  git commit -m "feat(benchmark): exact match scorer for reasoning dimension"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): exact match scorer for reasoning dimension`

---

- [ ] 9. BigCodeBench adapter

  **What to do**:
  - 创建 `benchmark/adapters/bigcodebench_adapter.py`
  - 从 HuggingFace 加载 BigCodeBench-Hard 子集，随机选 5 题

  **Must NOT do**:
  - 不要加载完整 BigCodeBench（只用 Hard 子集）
  - 不要实现代码执行逻辑（由 ExecutionScorer 负责）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7-8, 10-11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - BigCodeBench-Hard: https://huggingface.co/datasets/bigcode/bigcodebench-hard

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/adapters/bigcodebench_adapter.py**

  ```python
  """BigCodeBench 数据集适配器.加载官方 Hard 子集，随机选 5 题."""

  from __future__ import annotations

  import os
  import random
  from typing import List

  from datasets import load_dataset
  from benchmark.adapters.base import DatasetAdapter
  from benchmark.models.schemas import TaskDefinition


  class BigCodeBenchAdapter(DatasetAdapter):
      """BigCodeBench 适配器，从 Hard 子集随机选 5 题."""

      def load(self, path: str = "") -> List[TaskDefinition]:
          """加载 BigCodeBench-Hard 子集，随机选 5 题."""
          cache_dir = path or os.path.join("benchmark", "datasets", "bigcodebench")

          dataset = load_dataset(
              "bigcode/bigcodebench-hard",
              split="test",
              cache_dir=cache_dir,
              download_mode="reuse_dataset_if_exists",
              trust_remote_code=True,
          )

          # 随机选 5 题（固定随机种子保证可复现）
          rng = random.Random(42)
          indices = rng.sample(range(len(dataset)), min(5, len(dataset)))
          selected = [dataset[i] for i in indices]

          tasks = []
          for idx, item in enumerate(selected):
              task_id = f"bigcodebench_hard_{idx + 1}"

              # BigCodeBench 字段：task_id, instruct, test, entry_point
              task = TaskDefinition(
                  task_id=task_id,
                  dimension="backend-dev",
                  dataset="bigcodebench",
                  prompt=item.get("instruct", item.get("prompt", "")),
                  expected_output="",
                  metadata={
                      "difficulty": "hard",
                      "source": "bigcode/bigcodebench-hard",
                      "test": item.get("test", ""),
                      "entry_point": item.get("entry_point", ""),
                  },
              )
              tasks.append(task)

          return tasks

      def validate(self, task: TaskDefinition) -> bool:
          """验证任务格式.必须有 task_id, prompt, test 元数据."""
          has_test = "test" in task.metadata and task.metadata["test"]
          return bool(task.task_id and task.prompt and has_test)

      def get_dimension(self) -> str:
          return "backend-dev"
  ```

  - [ ] **Step 2: 验证 BigCodeBench 加载**

  ```bash
  python -c "
  from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
  a = BigCodeBenchAdapter()
  tasks = a.load()
  print(f'Loaded {len(tasks)} tasks')
  for t in tasks:
      print(f'  {t.task_id}: has_test={bool(t.metadata.get(\"test\"))}, prompt_len={len(t.prompt)}')
      assert a.validate(t), f'Validation failed for {t.task_id}'
  print('All tasks valid')
  "
  ```

  Expected: `Loaded 5 tasks` + 所有题目有效

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/adapters/bigcodebench_adapter.py
  git commit -m "feat(benchmark): BigCodeBench adapter — loads 5 tasks from Hard subset"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): BigCodeBench adapter — loads 5 tasks from Hard subset`

---

- [ ] 10. ExecutionScorer

  **What to do**:
  - 创建 `benchmark/scorers/execution_scorer.py`
  - 使用 subprocess 在临时文件中执行模型生成的代码 + 测试用例
  - 设置 30 秒超时，捕获 stdout/stderr
  - 通过（退出码 0）→ score=100，失败 → score=0

  **Must NOT do**:
  - 不要使用 Docker（用 subprocess 沙箱）
  - 不要允许网络请求（不需要，v1 简化）
  - 不要实现 pass@k（v1 只做单次执行）

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7-9, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:
  - `benchmark/scorers/base.py:BaseScorer` — 基类接口
  - `benchmark/models/schemas.py:ScoreResult` — 评分结果模型

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/scorers/execution_scorer.py**

  ```python
  """执行验证评分器.在 subprocess 沙箱中运行模型生成的代码并检查测试用例."""

  from __future__ import annotations

  import os
  import subprocess
  import tempfile

  from benchmark.models.schemas import ScoreResult, TaskDefinition
  from benchmark.scorers.base import BaseScorer


  class ExecutionScorer(BaseScorer):
      """执行验证评分器，用于 backend-dev 维度（BigCodeBench）。

      将模型生成的代码写入临时文件，附加测试用例，
      在 subprocess 中执行，30 秒超时。
      退出码 0 → score=100，非 0 → score=0。
      """

      def __init__(self, timeout: int = 30) -> None:
          self.timeout = timeout

      def score(
          self,
          model_output: str,
          expected: str,  # noqa: ARG002 — 基类接口要求
          task: TaskDefinition,
      ) -> ScoreResult:
          test_code = task.metadata.get("test", "")
          entry_point = task.metadata.get("entry_point", "")

          # 构造完整的可执行脚本
          full_code = self._build_executable(model_output, test_code, entry_point)

          # 写入临时文件执行
          fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_exec_")
          try:
              with os.fdopen(fd, "w") as f:
                  f.write(full_code)

              return self._run_and_score(temp_path, task.task_id)
          finally:
              if os.path.exists(temp_path):
                  os.unlink(temp_path)

      def _build_executable(self, model_code: str, test_code: str, entry_point: str) -> str:
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
                  ["python", script_path],
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

  - [ ] **Step 2: 验证正确代码通过**

  ```bash
  python -c "
  from benchmark.scorers.execution_scorer import ExecutionScorer
  from benchmark.models.schemas import TaskDefinition

  s = ExecutionScorer(timeout=10)
  task = TaskDefinition(
      task_id='test-add',
      dimension='backend-dev',
      dataset='test',
      prompt='implement add',
      expected_output='',
      metadata={'test': 'assert add(1, 2) == 3\nprint(\"PASS\")'},
  )

  # 正确代码
  r = s.score('def add(a, b): return a + b', '', task)
  assert r.passed and r.score == 100, f'Expected pass, got {r}'
  print(f'Correct code: passed={r.passed}, score={r.score}')
  "
  ```

  Expected: `Correct code: passed=True, score=100.0`

  - [ ] **Step 3: 验证错误代码失败**

  ```bash
  python -c "
  from benchmark.scorers.execution_scorer import ExecutionScorer
  from benchmark.models.schemas import TaskDefinition

  s = ExecutionScorer(timeout=10)
  task = TaskDefinition(
      task_id='test-bad',
      dimension='backend-dev',
      dataset='test',
      prompt='implement add',
      expected_output='',
      metadata={'test': 'assert add(1, 2) == 4'},
  )

  # 错误代码
  r = s.score('def add(a, b): return a - b', '', task)
  assert not r.passed and r.score == 0
  print(f'Wrong code: passed={r.passed}, score={r.score}')
  "
  ```

  Expected: `Wrong code: passed=False, score=0.0`

  - [ ] **Step 4: 验证超时**

  ```bash
  python -c "
  from benchmark.scorers.execution_scorer import ExecutionScorer
  from benchmark.models.schemas import TaskDefinition

  s = ExecutionScorer(timeout=2)
  task = TaskDefinition(
      task_id='test-timeout',
      dimension='backend-dev',
      dataset='test',
      prompt='infinite loop',
      expected_output='',
      metadata={'test': ''},
  )

  r = s.score('import time; time.sleep(100)', '', task)
  assert not r.passed and r.score == 0
  assert 'Timeout' in r.details.get('error', '')
  print(f'Timeout: passed={r.passed}, details={r.details}')
  "
  ```

  Expected: `Timeout: passed=False` + details 包含 Timeout

  - [ ] **Step 5: Commit**

  ```bash
  git add benchmark/scorers/execution_scorer.py
  git commit -m "feat(benchmark): execution scorer — subprocess sandbox with timeout"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): execution scorer — subprocess sandbox with timeout`

---

- [ ] 11. LLM Adapter

  **What to do**:
  - 创建 `benchmark/core/llm_adapter.py`
  - 从 `models.yaml` 加载模型配置
  - 实现 `generate()` 方法，调用 OpenAI 兼容 API
  - 支持重试（最多 3 次，指数退避）

  **Must NOT do**:
  - 不要实现 `generate_with_tools`（Stage 3）
  - 不要实现 `batch_generate`（v1 顺序执行）
  - 不要使用 `openai` SDK（直接用 requests 调用，减少依赖）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7-10)
  - **Blocks**: Task 12
  - **Blocked By**: Task 3

  **References**:
  - `benchmark/config.py:get_model_config` — 获取模型配置
  - `benchmark/configs/models.yaml.example` — 配置格式

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/core/llm_adapter.py**

  ```python
  """LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

  from __future__ import annotations

  import time
  from typing import Any

  import requests

  from benchmark.config import get_model_config


  class LLMEvalAdapter:
      """LLM 调用适配器.

      从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
      支持重试（最多 max_retries 次，指数退避）。
      """

      def __init__(self, max_retries: int = 3, timeout: int = 300) -> None:
          self.max_retries = max_retries
          self.timeout = timeout

      def generate(
          self,
          prompt: str,
          model: str,
          temperature: float = 0.0,
          max_tokens: int = 4096,
      ) -> str:
          """调用 LLM 生成文本.

          Args:
              prompt: 输入提示.
              model: 模型名称（需在 models.yaml 中配置）.
              temperature: 温度参数（评测时固定为 0）.
              max_tokens: 最大输出 token 数.

          Returns:
              模型生成的文本.

          Raises:
              ValueError: 模型未配置.
              ConnectionError: 重试耗尽后仍失败.
          """
          cfg = get_model_config(model)
          api_key = cfg["api_key"]
          api_base = cfg["api_base"].rstrip("/")
          model_max_tokens = cfg.get("max_tokens", max_tokens)

          url = f"{api_base}/chat/completions"
          headers = {
              "Authorization": f"Bearer {api_key}",
              "Content-Type": "application/json",
          }
          payload: dict[str, Any] = {
              "model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature,
              "max_tokens": min(max_tokens, model_max_tokens),
          }

          last_error: Exception | None = None
          for attempt in range(self.max_retries):
              try:
                  resp = requests.post(
                      url, headers=headers, json=payload, timeout=self.timeout,
                  )
                  resp.raise_for_status()
                  data = resp.json()
                  return data["choices"][0]["message"]["content"]

              except requests.exceptions.RequestException as exc:
                  last_error = exc
                  if attempt < self.max_retries - 1:
                      wait = 2 ** attempt  # 1s, 2s, 4s
                      time.sleep(wait)

          raise ConnectionError(
              f"Failed after {self.max_retries} retries for model '{model}': {last_error}"
          ) from last_error
  ```

  - [ ] **Step 2: 验证模型配置不存在时报错**

  ```bash
  python -c "
  from benchmark.core.llm_adapter import LLMEvalAdapter
  a = LLMEvalAdapter()
  try:
      a.generate('test', 'nonexistent-model')
      print('ERROR: should have raised ValueError')
  except ValueError as e:
      print(f'OK: {e}')
  "
  ```

  Expected: `OK: Model 'nonexistent-model' not found...`

  - [ ] **Step 3: 验证真实 API 调用（需要有效 models.yaml）**

  > 此步骤需要用户先配置 `benchmark/configs/models.yaml`（复制 example 并填入真实 key）。
  > 如果没有配置，跳过此步骤，在 Task 12 集成时验证。

  ```bash
  python -c "
  from benchmark.core.llm_adapter import LLMEvalAdapter
  a = LLMEvalAdapter()
  result = a.generate('What is 2+3? Answer with just the number.', 'glm-4.7')
  print(f'Response: {result[:100]}')
  assert '5' in result, f'Expected 5 in response, got: {result}'
  print('LLM call OK')
  "
  ```

  Expected: 包含 "5" 的回复（如果 API key 有效）

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/core/llm_adapter.py
  git commit -m "feat(benchmark): LLM adapter with retry and OpenAI-compatible API"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): LLM adapter with retry and OpenAI-compatible API`

---

- [ ] 12. CLI commands (evaluate + list-datasets + export)

  **What to do**:
  - 创建 `benchmark/cli.py`，实现 3 个命令： `evaluate`, `list-datasets`, `export`
  - evaluate 命令调用 LLM → 评分 → 写入 SQLite 的完整流程
  - list-datasets 列出可用数据集
  - export 从 SQLite 导出 JSON/CSV

  **Must NOT do**:
  - 不要实现 `scheduler` 命令（Stage 2）
  - 不要实现 `report` 命令（Stage 3）
  - 不要并行执行（v1 顺序）

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 13, Final Verification
  - **Blocked By**: Tasks 7-11

  **References**:
  - `benchmark/core/llm_adapter.py:LLMEvalAdapter` — LLM 调用
  - `benchmark/adapters/gsm8k_adapter.py:GSM8KAdapter` — GSM8K 适配器
  - `benchmark/adapters/bigcodebench_adapter.py:BigCodeBenchAdapter` — BigCodeBench 适配器
  - `benchmark/scorers/exact_match_scorer.py:ExactMatchScorer` — 稡式匹配评分
  - `benchmark/scorers/execution_scorer.py:ExecutionScorer` — 执行验证评分
  - `benchmark/models/database.py:Database` — SQLite 操作
  - `benchmark/models/schemas.py` — 数据模型

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/cli.py**

  ```python
  """CLI 命令入口。支持 evaluate / list-datasets / export 命令."""

  from __future__ import annotations

  import json
  import csv
  import uuid
  from datetime import datetime
  from pathlib import Path

  import click
  from rich.console import Console
  from rich.progress import Progress

  from benchmark.adapters.gsm8k_adapter import GSM8KAdapter
  from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
  from benchmark.scorers.exact_match_scorer import ExactMatchScorer
  from benchmark.scorers.execution_scorer import ExecutionScorer
  from benchmark.models.schemas import EvalRun, EvalResult
  from benchmark.models.database import Database
  from benchmark.core.llm_adapter import LLMEvalAdapter

  console = Console()

  # 维度 -> (适配器, 评分器) 映射
  DIMENSION_REGISTRY: dict[str, tuple] = {
      "reasoning": (GSM8KAdapter, ExactMatchScorer),
      "backend-dev": (BigCodeBenchAdapter, ExecutionScorer),
  }


  @click.group()
  def cli():
      """LLM Benchmark 评测工具."""
      pass


  @cli.command()
  @click.option("--model", required=True, help="模型名称（需在 configs/models.yaml 中配置）")
  @click.option("--dimension", required=True, type=click.Choice(["reasoning", "backend-dev"]), help="评测维度")
  @click.option("--samples", default=5, help="评测题目数量")
  def evaluate(model: str, dimension: str, samples: int) -> None:
      """运行评测。调用 LLM 生成答案，评分并保存结果."""
      if dimension not in DIMENSION_REGISTRY:
          console.print(f"[red]Unknown dimension: {dimension}[/red]")
          raise System.Exit(1)

      adapter_cls, scorer_cls = DIMENSION_REGISTRY[dimension]
      adapter = adapter_cls()
      scorer = scorer_cls()
      llm = LLMEvalAdapter()
      db = Database()

      # 加载数据集
      tasks = adapter.load()[:samples]
      if not tasks:
          console.print("[red]No tasks loaded.[/red]")
          raise System.Exit(1)

      console.print(f"[bold green]Starting evaluation:[/bold green] {dimension} with {len(tasks)} tasks, model={model}")

      # 创建运行记录
      run_id = str(uuid.uuid4())[:8]
      run = EvalRun(
          run_id=run_id,
          model=model,
          dimension=dimension,
          dataset=adapter.get_dimension(),
          started_at=datetime.now(),
          status="running",
      )
      db.create_run(run)

      # 评测每道题
      total_score = 0.0
      passed_count = 0
      with Progress() as progress:
          task_progress = progress.add_task("Evaluating", total=len(tasks))
          for i, task in enumerate(tasks, 1):
              # 调用 LLM
              start_time = datetime.now()
              model_output = llm.generate(task.prompt, model)
              execution_time = (datetime.now() - start_time).total_seconds()

              # 评分
              score_result = scorer.score(model_output, task.expected_output, task)

              # 保存结果
              result = EvalResult(
                  result_id=str(uuid.uuid4())[:8],
                  run_id=run_id,
                  task_id=task.task_id,
                  task_content=task.prompt,
                  model_output=model_output,
                  functional_score=score_result.score,
                  final_score=score_result.score,
                  passed=score_result.passed,
                  details=score_result.details,
                  execution_time=execution_time,
                  created_at=datetime.now(),
              )
              db.save_result(result)

              total_score += score_result.score
              if score_result.passed:
                  passed_count += 1

              status_icon = "[green]PASS[/green]" if score_result.passed else "[red]FAIL[/red]"
              console.print(
                  f"  [{i}/{len(tasks)}] {task.task_id} | "
                  f"Score: {score_result.score:.0f} | {status_icon} | "
                  f"Time: {execution_time:.1f}s"
              )
              task_progress.advance()

      # 完成运行
      db.finish_run(run_id, "completed")

      avg_score = total_score / len(tasks) if tasks else 0
      console.print(
          f"\n[bold]Evaluation complete:[/bold] run_id={run_id}\n"
          f"  Average Score: [bold]{avg_score:.1f}[/bold]\n"
          f"  Passed: {passed_count}/{len(tasks)}"
      )


  @cli.command("list-datasets")
  def list_datasets() -> None:
      """列出可用数据集."""
      console.print("[bold]Available datasets:[/bold]")
      console.print("  [cyan]reasoning:[/cyan]     GSM8K (hardest 5 tasks by step count)")
      console.print("  [cyan]backend-dev:[/cyan]  BigCodeBench-Hard (5 tasks)")


  @cli.command()
  @click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json", help="导出格式")
  @click.option("--output", default="results.json", help="输出文件路径")
  @click.option("--model", default=None, help="按模型过滤")
  @click.option("--dimension", default=None, help="按维度过滤")
  def export(fmt: str, output: str, model: str | None, dimension: str | None) -> None:
      """导出评测结果."""
      db = Database()
      results = db.get_results(model=model, dimension=dimension)

      if not results:
          console.print("[yellow]No results found.[/yellow]")
          return

      output_path = Path(output)

      if fmt == "json":
          data = [dict(row) for row in results]
          with open(output_path, "w", encoding="utf-8") as f:
              json.dump(data, f, indent=2, ensure_ascii=False, default=str)
          console.print(f"[green]Exported {len(data)} results to {output_path}[/green]")

      elif fmt == "csv":
          if results:
              keys = results[0].keys()
              with open(output_path, "w", newline="", encoding="utf-8") as f:
                  writer = csv.DictWriter(f, fieldnames=keys)
                  writer.writeheader()
                  writer.writerows(results)
              console.print(f"[green]Exported {len(results)} results to {output_path}[/green]")
  ```

  - [ ] **Step 2: 验证 CLI help**

  ```bash
  python -m benchmark --help
  ```

  Expected: 显示 evaluate, list-datasets, export 命令

  - [ ] **Step 3: 验证 list-datasets**

  ```bash
  python -m benchmark list-datasets
  ```

  Expected: 列出 reasoning 和 backend-dev 数据集

  - [ ] **Step 4: 验证 evaluate --help**

  ```bash
  python -m benchmark evaluate --help
  ```

  Expected: 显示 --model, --dimension, --samples 参数

  - [ ] **Step 5: Commit**

  ```bash
  git add benchmark/cli.py benchmark/__main__.py
  git commit -m "feat(benchmark): CLI commands — evaluate, list-datasets, export"
  ```

  **QA Scenarios**:
  ```
  Scenario: evaluate reasoning 维度
    Tool: Bash
    Steps:
      1. cp benchmark/configs/models.yaml.example benchmark/configs/models.yaml
      2. sed -i 's/YOUR_GLM_API_KEY_HERE/YOUR_REAL_KEY/' benchmark/configs/models.yaml
      3. python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 5
    Expected: 输出进度和每题分数，最终显示平均分
    Evidence: .sisyphus/evidence/task-12-eval-reasoning.txt

  Scenario: 无效维度报错
    Tool: Bash
    Steps:
      1. python -m benchmark evaluate --model glm-4.7 --dimension invalid 2>&1 || true
    Expected: 退出码非零
    Evidence: .sisyphus/evidence/task-12-invalid-dim.txt

  Scenario: list-datasets 命令
    Tool: Bash
    Steps:
      1. python -m benchmark list-datasets
    Expected: 列出 reasoning 和 backend-dev
    Evidence: .sisyphus/evidence/task-12-list-datasets.txt

  Scenario: JSON 导出
    Tool: Bash
    Steps:
      1. python -m benchmark export --format json --output /tmp/benchmark_test.json
      2. python -c "import json; d = json.load(open('/tmp/benchmark_test.json')); print(type(d), len(d) if isinstance(d, list) else 'dict')"
    Expected: 合法 JSON 文件
    Evidence: .sisyphus/evidence/task-12-export-json.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): CLI commands — evaluate, list-datasets, export`

---

- [ ] 13. Streamlit basic interface

  **What to do**:
  - 创建 `benchmark/visualization/app.py`
  - 结果列表展示（表格）
  - 按 model/dimension/date 过滤
  - 单题详情查看（题目、输出、分数）

  **Must NOT do**:
  - 不要实现趋势图（Stage 2）
  - 不要实现统计检验（Stage 3）
  - 不要实现实时更新

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 12)
  - **Blocks**: Final Verification
  - **Blocked By**: Task 6

  **References**:
  - `benchmark/models/database.py:Database` — SQLite 操作
  - `benchmark/models/schemas.py` — 数据模型

  **Step-by-step instructions**:

  - [ ] **Step 1: 创建 benchmark/visualization/app.py**

  ```python
  """Streamlit 可视化界面。展示评测结果列表和单题详情."""

  import sqlite3
  from datetime import datetime

  import pandas as pd
  import streamlit as st

  DB_PATH = "benchmark/data/results.db"


  @st.cache_resource
  def get_connection() -> sqlite3.Connection:
      """获取 SQLite 连接（缓存）."""
      conn = sqlite3.connect(DB_PATH)
      conn.row_factory = sqlite3.Row
      return conn


  def get_models(conn: sqlite3.Connection) -> list[str]:
      """获取所有已评测的模型名称."""
      cursor = conn.execute("SELECT DISTINCT model FROM eval_runs")
      return [row["model"] for row in cursor.fetchall()]


  def get_dimensions(conn: sqlite3.Connection) -> list[str]:
      """获取所有已评测的维度名称."""
      cursor = conn.execute("SELECT DISTINCT dimension FROM eval_runs")
      return [row["dimension"] for row in cursor.fetchall()]


  def get_results_df(conn: sqlite3.Connection, model: str | None, dimension: str | None) -> pd.DataFrame:
      """查询结果并返回 DataFrame."""
      query = """
          SELECT
              r.result_id,
              e.model,
              e.dimension,
              r.task_id,
              r.final_score,
              r.passed,
              r.execution_time,
              r.created_at
          FROM eval_results r
          JOIN eval_runs e ON r.run_id = e.run_id
          WHERE 1=1
      """
      params: list[str] = []

      if model and model != "All":
          query += " AND e.model = ?"
          params.append(model)
      if dimension and dimension != "All":
          query += " AND e.dimension = ?"
          params.append(dimension)

      query += " ORDER BY r.created_at DESC"
      return pd.read_sql_query(query, conn, params=params)


  def get_result_detail(conn: sqlite3.Connection, result_id: str) -> dict | None:
      """查询单条结果的详情."""
      cursor = conn.execute(
          "SELECT * FROM eval_results WHERE result_id = ?",
          (result_id,),
      )
      row = cursor.fetchone()
      if not row:
          return None
      columns = [desc[0] for desc in cursor.description]
      return dict(zip(columns, row))


  def main() -> None:
      st.set_page_config(page_title="LLM Benchmark", page_icon="📊", layout="wide")
      st.title("LLM Benchmark Results")

      conn = get_connection()

      # 检查是否有数据
      results_check = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()
      if results_check[0] == 0:
          st.info("No evaluation results yet. Run an evaluation to get started.")
          st.code("python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 5")
          return

      # 侧边栏过滤器
      st.sidebar.header("Filters")

      models = get_models(conn)
      model_options = ["All"] + models
      selected_model = st.sidebar.selectbox("Model", model_options)

      dimensions = get_dimensions(conn)
      dim_options = ["All"] + dimensions
      selected_dimension = st.sidebar.selectbox("Dimension", dim_options)

      # 结果列表
      st.subheader("Evaluation Results")
      df = get_results_df(conn, selected_model, selected_dimension)

      if df.empty:
          st.warning("No results match the selected filters.")
          return

      # 格式化显示
      display_df = df.copy()
      display_df["passed"] = display_df["passed"].map(lambda x: "✅" if x else "❌")
      display_df["execution_time"] = display_df["execution_time"].round(2).astype(str) + "s"
      display_df.columns = ["ID", "Model", "Dimension", "Task", "Score", "Passed", "Time", "Date"]

      st.dataframe(display_df, use_container_width=True, hide_index=True)

      # 单题详情
      st.subheader("Result Detail")
      result_ids = df["result_id"].tolist()
      selected_result = st.selectbox("Select a result to view details", result_ids)

      if selected_result:
          detail = get_result_detail(conn, selected_result)
          if detail:
              col1, col2 = st.columns(2)
              with col1:
                  st.metric_label("Score")
                  st.write(f"{detail['final_score']:.1f}")
                  st.metric_label("Passed")
                  st.write("Yes" if detail["passed"] else "No")
                  st.metric_label("Execution Time")
                  st.write(f"{detail['execution_time']:.2f}s")

              with col2:
                  st.metric_label("Task Content")
                  st.text_area("Prompt", value=detail.get("task_content", ""), height=200, disabled=True)
                  st.metric_label("Model Output")
                  st.text_area("Output", value=detail.get("model_output", ""), height=300, disabled=True)

              if detail.get("details"):
                  with st.expander("Score Details"):
                      st.json(detail["details"])


  if __name__ == "__main__":
      main()
  ```

  - [ ] **Step 2: 验证 Streamlit 启动**

  ```bash
  streamlit run benchmark/visualization/app.py --server.headless true --server.port 8501 &
  sleep 5
  curl -s http://localhost:8501 | grep -o "LLM Benchmark" || echo "Found"
  curl -s http://localhost:8501 | grep -o "No evaluation results yet" && echo "Found empty state"
  kill %2  # 停止 Streamlit
  ```

  Expected: 页面加载成功，显示空状态提示

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/visualization/app.py
  git commit -m "feat(benchmark): Streamlit basic interface — results list and detail view"
  ```

  **QA Scenarios**:
  ```
  Scenario: 空状态显示
    Tool: Bash
    Steps:
      1. streamlit run benchmark/visualization/app.py --server.headless true --server.port 8502 &
      2. sleep 5
      3. curl -s http://localhost:8502 | grep -o "No evaluation results yet"
      4. kill %2
    Expected: 页面包含空状态提示
    Evidence: .sisyphus/evidence/task-13-empty-state.txt

  Scenario: 有数据时展示
    Tool: Bash
    Steps:
      1. python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 2
      2. streamlit run benchmark/visualization/app.py --server.headless true --server.port 8503 &
      3. sleep 5
      4. curl -s http://localhost:8503 | grep -o "LLM Benchmark"
      5. kill %2
    Expected: 页面包含结果数据
    Evidence: .sisyphus/evidence/task-13-with-data.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): Streamlit basic interface — results list and detail view`

---

## Final Verification Wave