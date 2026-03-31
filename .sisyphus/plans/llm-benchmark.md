# LLM Benchmark 评测系统

## TL;DR

> **Quick Summary**: 在现有 chat 项目下构建独立的 benchmark 子项目，支持 5 个维度（frontend-dev、backend-dev、system-architecture、tool-use-and-agentic、reasoning）的 LLM 评测能力。集成 6 个公开 benchmark 数据集，采用两阶段混合评分策略（functional_score 自动化验证 + quality_score LLM-as-judge，带短路规则），维度差异化权重（auto/judge），难度加权（easy=1.0, medium=1.5, hard=2.0）。结果持久化到 SQLite，通过 Streamlit 提供 Web 可视化界面。
> 
> **Deliverables**:
> - `benchmark/` 独立子项目（CLI 评测引擎 + 评分器 + 数据集适配器）
> - 5 个维度的评分引擎（执行验证、LLM-as-judge、exact match）
> - 6 个数据集适配器（SWE-bench、BigCodeBench、AgentBench、GSM8K/MATH、MMLU、FrontCode）
> - SQLite 结果存储 + 评估恢复机制
> - Streamlit Web 可视化界面（结果展示、过滤、导出）
> - 自定义题目扩展机制（JSON 格式）
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: 数据模型 → 数据集适配器 → 评分引擎 → CLI → Streamlit

---

## Context

### Project Structure
**项目结构说明**：
- 本项目是一个独立的 benchmark 项目，项目根目录直接包含 `benchmark/` 目录
- 项目根目录结构：
  ```
  llm-api-daily-benchmark/
  ├── benchmark/
  │   ├── __init__.py
  │   ├── __main__.py
  │   ├── adapters/        # 数据集适配器
  │   ├── scorers/          # 评分引擎
  │   ├── datasets/         # 数据集存储
  │   ├── models/           # 数据模型
  │   ├── core/             # 核心逻辑
  │   ├── visualization/    # Streamlit 可视化
  │   ├── configs/          # 配置文件
  │   └── docs/             # 文档
  ├── pyproject.toml
  └── README.md
  ```
- **依赖管理**：使用 `uv` 或 `pip` 管理依赖，`pyproject.toml` 定义项目元数据和依赖
- **数据集存储**：数据集缓存在 `benchmark/datasets/` 目录下，避免重复下载

### Original Request
构建多维度 LLM benchmark 系统，覆盖 frontend-dev、backend-dev、system-architecture、tool use and agentic、reasoning 五个维度。核心问题：如何评估 LLM 的输出质量。

### Interview Summary
**Key Discussions**:
- 数据集来源：使用现有公开 benchmark 数据集
- 评估方式：混合方案，不同维度用不同策略
- 输出形式：Web 可视化界面
- 技术栈：Streamlit（Python 原生）、SQLite
- 多模型对比：不需要实时选择，结果持久化后后续展示对比
- 自定义扩展：支持未来添加自定义题目
- 项目结构：独立子项目（chat/benchmark/）
- 前端数据集：FrontCode / 自建前端题
- 测试策略：NO unit tests，Agent-Executed QA only

**Research Findings**:
- SWE-bench: 2,294 个真实 GitHub issue/PR，执行验证，% Resolved
- BigCodeBench: 1,140 个编程任务，pass@k
- AgentBench: 8 个环境，任务完成率
- GSM8K/MATH: 数学推理，exact match
- MMLU: 57 学科多选题，准确率
- 评分引擎：执行验证、AST 匹配、LLM-as-judge、语义相似度
- 非确定性处理：temperature=0、多次采样、bootstrap CI

### Metis Review
**Identified Gaps** (addressed):
- 代码执行沙箱：v1 用 subprocess + timeout/memory 限制，不做 Docker
- 前端评估：v1 静态分析 + LLM-as-judge，不做浏览器渲染
- 评估恢复：检查 SQLite 已有结果，跳过已完成
- 数据量：v1 每个维度取子集（10-50 题）
- LLM-as-judge：temperature=0 确保一致性
- 并发：v1 顺序执行

---

## Dimension Mapping

> **维度到数据集/评分器的映射关系**

| 维度 | 数据集 | 评分器 | auto_weight | judge_weight | 说明 |
|------|--------|--------|--------------|--------------|------|
| frontend-dev | FrontCode (自建) | LLM-as-judge | 0.8 | 0.2 | 前端代码质量评估，不做浏览器渲染 |
| backend-dev | SWE-bench / BigCodeBench | 执行验证 (test pass) | 0.8 | 0.2 | 代码执行测试，通过率 |
| system-architecture | MMLU (5个子集) | Exact Match + LLM Judge | 0.2 | 0.8 | 多选题精确匹配 + 推理评估 |
| tool-use-agentic | AgentBench (DB/OS) | Agent Loop + 工具验证 | 0.5 | 0.5 | 多轮工具调用，任务完成率 |
| reasoning | GSM8K / MATH | Exact Match | 0.8 | 0.2 | 数学答案精确匹配 |

**数据集来源**：
- SWE-bench: HuggingFace `princeton-nlp/SWE-bench_Lite`
- BigCodeBench: HuggingFace `bigcode/bigcodebench`
- AgentBench: GitHub `THUDM/AgentBench`
- GSM8K: HuggingFace `openai/gsm8k`
- MATH: HuggingFace `hendrycks/competition_math`
- MMLU: HuggingFace `cais/mmlu` (philosophy, CS, math, physics, engineering 子集)
- FrontCode: 自建数据集 `benchmark/datasets/frontcode/tasks.jsonl`

---

## Work Objectives

### Core Objective
构建一个多维度 LLM 评测系统，集成公开 benchmark 数据集，支持混合评分策略，结果持久化并通过 Web 界面可视化。

### Concrete Deliverables
- `benchmark/` 子项目：CLI 评测引擎 + 评分器 + 数据集适配器
- 5 个维度 × 6 个数据集的完整评测流水线
- 两阶段评分引擎：Stage 1 functional_score（自动化验证）+ Stage 2 quality_score（LLM-as-judge）
- 短路规则：functional_score=0 直接 0 分，不送 LLM Judge
- 维度差异化权重：auto_weight / judge_weight 按维度不同
- 难度加权：easy=1.0, medium=1.5, hard=2.0
- Agent Loop 实现：多轮工具调用循环（tool-use 维度）
- SQLite 结果存储 + 评估恢复
- Streamlit Web 可视化
- 自定义题目扩展（YAML 格式，借鉴 design doc schema）

### Definition of Done
- [ ] `python -m benchmark evaluate --dimension reasoning --model glm-4.7 --samples 5` 成功运行并写入 SQLite
- [ ] `streamlit run benchmark/app.py` 启动 Web 界面，展示评测结果
- [ ] 5 个维度均可独立评测，结果正确持久化
- [ ] 自定义题目可通过 YAML 文件添加并参与评测
- [ ] 两阶段评分正确工作：functional_score=0 时跳过 LLM Judge
- [ ] 维度权重和难度加权正确计算最终分数

### Must Have
- 5 个维度独立评测能力
- 两阶段评分策略：
  - Stage 1: functional_score (0-100) — 自动化验证（test_pass_rate, build_success, output_correctness）
  - Stage 2: quality_score (0-100) — LLM-as-judge（code_quality, architecture_design, reasoning_depth）
  - 短路规则：functional_score=0 直接 0 分，不送 LLM Judge
- 维度差异化权重：
  - Frontend: auto=0.8, judge=0.2
  - Backend: auto=0.8, judge=0.2
  - Architecture: auto=0.2, judge=0.8
  - Agentic: auto=0.5, judge=0.5
  - Reasoning: auto=0.8, judge=0.2
- 难度加权：easy=1.0, medium=1.5, hard=2.0
- SQLite 持久化 + 评估恢复
- Streamlit Web 可视化
- CLI 接口 + 进度报告
- 自定义题目扩展（YAML 格式）
- 错误处理（API 失败、超时、格式错误）
- Agent Loop 实现（tool-use 维度，多轮工具调用循环）

### Must NOT Have (Guardrails)
- NO Docker 容器化执行（v1 用 subprocess 沙箱）
- NO 浏览器渲染验证前端代码（v1 用静态分析 + LLM-as-judge）
- NO 实时多模型对比 UI（结果持久化后后续展示）
- NO 认证/授权（本地单用户工具）
- NO 分布式评估（v1 顺序执行）
- NO 复杂插件架构（简单 YAML 自定义题目）
- NO 高级 Streamlit 特性（实时推送、复杂过滤）
- NO Node.js sidecar（纯 Python 实现）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest via pyproject.toml)
- **Automated tests**: NO (user specified)
- **Framework**: none
- **Agent-Executed QA**: ALL tasks include QA scenarios

### QA Policy
每个任务包含 Agent-Executed QA 场景：
- CLI 命令执行 → 验证输出和 SQLite 写入
- Streamlit 界面 → curl 验证页面加载
- 错误处理 → 触发异常场景 → 验证优雅降级
- 证据保存到 `.sisyphus/evidence/task-{N}-{scenario}.{ext}`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — 可并行 7 任务):
├── Task 1: 项目脚手架 + 依赖配置 [quick]
├── Task 2: 数据模型 + SQLite Schema [quick]
├── Task 3: LLMClient 评测适配器 [quick]
├── Task 4: 数据集适配器基类 [quick]
├── Task 5: 评分引擎基类 + 两阶段评分管道 [quick]
├── Task 6: 配置系统 + CLI 框架 [quick]
└── Task 7: 评分聚合（维度权重 + 难度加权） [quick]

Wave 2 (Datasets + Scorers — 可并行 7 任务):
├── Task 8: SWE-bench / BigCodeBench 适配器 [deep]
├── Task 9: AgentBench 适配器 [deep]
├── Task 10: GSM8K / MATH 适配器 [quick]
├── Task 11: MMLU 适配器 [quick]
├── Task 12: FrontCode 数据集 + 适配器 [quick]
├── Task 13: 自动化验证评分器（functional_score） [deep]
└── Task 14: LLM-as-judge 评分器（quality_score） [deep]

Wave 3 (Integration — 可并行 5 任务):
├── Task 15: CLI 评测命令完整实现 [deep]
├── Task 16: Agent Loop 实现（tool-use 维度） [deep]
├── Task 17: 评估恢复机制 [quick]
├── Task 18: 自定义题目系统（YAML 格式） [quick]
└── Task 19: 结果导出（JSON/CSV） [quick]

Wave 4 (Visualization + Error Handling — 可并行 2 任务):
├── Task 20: Streamlit 主界面 [visual-engineering]
└── Task 21: 错误处理 + 日志系统 [quick]

Wave FINAL (Verification — 4 并行审查):
├── Task F1: 计划合规审计 [oracle]
├── Task F2: 代码质量审查 [unspecified-high]
├── Task F3: 手动 QA 验证 [unspecified-high]
└── Task F4: 范围保真检查 [deep]
```

### Dependency Matrix
- **1-7**: — — 8-14, 15-19
- **4**: 2 — 8-12
- **5**: 2 — 13-14
- **7**: 2, 5 — 15
- **8-12**: 4 — 15
- **13-14**: 5 — 15
- **15**: 7-14 — 16-20
- **16**: 15 — 20
- **17-19**: 15 — 20
- **20**: 15-19 — F1-F4

### Agent Dispatch Summary
- **Wave 1**: 7 任务 — T1-T7 → `quick`
- **Wave 2**: 7 任务 — T8-T9 → `deep`, T10-T12 → `quick`, T13-T14 → `deep`
- **Wave 3**: 5 任务 — T15-T16 → `deep`, T17-T19 → `quick`
- **Wave 4**: 2 任务 — T20 → `visual-engineering`, T21 → `quick`
- **FINAL**: 4 任务 — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. 项目脚手架 + 依赖配置

  **What to do**:
  - 在 `chat/` 下创建 `benchmark/` 目录结构
  - 创建 `benchmark/__init__.py`、`benchmark/__main__.py`
  - 更新 `pyproject.toml` 添加依赖：`pydantic`, `rich`（CLI 进度）, `streamlit`, `datasets`（HuggingFace 数据集）, `pyyaml`（配置解析）
  - 创建目录结构：`adapters/`, `scorers/`, `datasets/`, `models/`, `visualization/`, `configs/`, `docs/`
  - 创建 `benchmark/docs/ENV_SETUP.md` 说明环境配置：
    - API key 配置（GLM_API_KEY 或其他 LLM provider 的 key）
    - HuggingFace token 配置（可选，用于私有数据集）
    - `.env` 文件模板
  - 创建 `benchmark/README.md` 说明文档

  **Must NOT do**:
  - 不要创建任何实现代码，只创建目录结构和依赖配置
  - 不要修改现有 `chat/` 项目的核心文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯脚手架工作，无复杂逻辑
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `py-preference`: Python 项目，但脚手架不涉及编码风格决策

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-6)
  - **Blocks**: All subsequent tasks
  - **Blocked By**: None

  **References**:
  - `pyproject.toml` — 现有依赖配置，需追加新依赖
  - `benchmark/` — 新建目录，与 `llm_client.py` 同级

**Acceptance Criteria**:
  - [ ] `pyproject.toml` 包含 pydantic、rich、streamlit、datasets、pyyaml 依赖
  - [ ] `uv sync` 成功安装所有依赖
  - [ ] 环境变量说明文档存在：`benchmark/docs/ENV_SETUP.md` 说明 API key 配置

  **QA Scenarios**:
  ```
  Scenario: 目录结构验证
    Tool: Bash
    Steps:
      1. ls -la benchmark/
      2. 检查 __init__.py 和 __main__.py 存在
      3. 检查 docs/ENV_SETUP.md 存在
    Expected: 目录结构完整，文件存在
    Evidence: .sisyphus/evidence/task-1-dir-structure.txt

  Scenario: 依赖安装验证
    Tool: Bash
    Steps:
      1. uv sync
      2. python -c "import pydantic; import rich; import streamlit; import datasets; import yaml"
    Expected: 无 ImportError
    Evidence: .sisyphus/evidence/task-1-deps-install.txt

  Scenario: 环境配置文档验证
    Tool: Bash
    Steps:
      1. cat benchmark/docs/ENV_SETUP.md
    Expected: 文档包含 API key 配置说明和 .env 模板
    Evidence: .sisyphus/evidence/task-1-env-doc.txt
  ```

  **Commit**: YES (groups with 2-6)
  - Message: `feat(benchmark): project scaffolding and dependencies`
  - Pre-commit: `uv sync`

---

- [ ] 2. 数据模型 + SQLite Schema

  **What to do**:
  - 定义 Pydantic 数据模型：
    - `EvalRun`: 评测运行记录（run_id, model, dimension, dataset, timestamp, status, config_hash）
    - `EvalResult`: 单题结果（result_id, run_id, task_id, task_content, model_output, score, score_details, status, execution_time）
    - `TaskDefinition`: 题目定义（task_id, dimension, dataset, prompt, expected_output, metadata）
  - 设计 SQLite Schema：
    - `eval_runs` 表：id, run_id, model, dimension, dataset, started_at, finished_at, status, config_hash
    - `eval_results` 表：id, result_id, run_id, task_id, task_content, model_output, score, score_details, status, execution_time, created_at
    - `custom_tasks` 表：id, task_id, dimension, prompt, expected_output, metadata, created_at
  - 创建 `benchmark/models/schemas.py` 和 `benchmark/models/database.py`
  - 实现 SQLite 连接管理（使用 `sqlite3` 标准库）

  **Must NOT do**:
  - 不要使用 SQLAlchemy 等 ORM（保持轻量）
  - 不要实现任何业务逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 数据模型定义，结构清晰
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-6)
  - **Blocks**: Tasks 4-5, 7-11
  - **Blocked By**: None

  **References**:
  - `benchmark/models/` — 新建目录
  - Pydantic docs: https://docs.pydantic.dev/latest/ — 数据验证模型

  **Acceptance Criteria**:
  - [ ] `schemas.py` 包含 EvalRun、EvalResult、TaskDefinition 模型
  - [ ] `database.py` 包含 SQLite 初始化、表创建、CRUD 基础方法
  - [ ] 可独立运行 `python -m benchmark.models.database` 创建数据库

  **QA Scenarios**:
  ```
  Scenario: 数据库初始化
    Tool: Bash
    Steps:
      1. python -m benchmark.models.database
      2. sqlite3 benchmark/data/results.db ".tables"
    Expected: 输出 eval_runs, eval_results, custom_tasks 三个表
    Evidence: .sisyphus/evidence/task-2-db-init.txt

  Scenario: 模型验证
    Tool: Bash
    Steps:
      1. python -c "from benchmark.models.schemas import EvalRun; r = EvalRun(run_id='test', model='glm-4.7', dimension='reasoning', dataset='gsm8k', status='running'); print(r.model_dump())"
    Expected: 输出合法 JSON，字段完整
    Evidence: .sisyphus/evidence/task-2-models.txt
  ```

  **Commit**: YES (groups with 1, 3-6)
  - Message: `feat(benchmark): data models and SQLite schema`

---

- [ ] 3. LLMClient 评测适配器

  **What to do**:
  - 创建 `benchmark/core/llm_adapter.py`
  - 实现评测专用 LLM 接口（不依赖现有的 `llm_client.py`，从零实现）：
    - `generate(prompt: str, model: str, temperature: float = 0, max_tokens: int = 4096) -> str`
    - `generate_with_tools(prompt: str, tools: list, model: str) -> dict`（用于 agentic 评测）
    - `batch_generate(prompts: list[str], model: str) -> list[str]`
  - 支持多种 LLM provider（通过环境变量配置）：
    - `GLM_API_KEY` / `GLM_API_BASE` — 智谱 GLM
    - `OPENAI_API_KEY` / `OPENAI_API_BASE` — OpenAI 兼容接口
  - 支持 temperature 控制（评测时 temperature=0）
  - 支持超时和重试（API 失败时最多重试 3 次）
  - 记录每次调用的 token 消耗和耗时

  **Must NOT do**:
  - 不要修改原有 `llm_client.py`
  - 不要实现流式响应（评测不需要）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 封装现有客户端，接口简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-6)
  - **Blocks**: Tasks 12-14
  - **Blocked By**: None

  **References**:
  - `llm_client.py` — 现有 LLMClient 实现，需封装
  - `message_adapter.py` — 消息格式适配参考

  **Acceptance Criteria**:
  - [ ] `llm_adapter.py` 提供 generate、generate_with_tools、batch_generate 方法
  - [ ] 可调用 `python -c "from benchmark.core.llm_adapter import LLMEvalAdapter; a = LLMEvalAdapter(); print(a.generate('1+1=?', 'glm-4.7'))"`

  **QA Scenarios**:
  ```
  Scenario: 基本生成调用
    Tool: Bash
    Steps:
      1. python -c "from benchmark.core.llm_adapter import LLMEvalAdapter; a = LLMEvalAdapter(); result = a.generate('计算 2+3', 'glm-4.7'); print(result)"
    Expected: 输出包含 "5" 的文本
    Evidence: .sisyphus/evidence/task-3-generate.txt

  Scenario: API 失败重试
    Tool: Bash
    Steps:
      1. 模拟无效 API key 或无效模型名
      2. 验证重试逻辑（最多 3 次）后抛出明确异常
    Expected: 重试 3 次后抛出 ConnectionError 或 AuthenticationError
    Evidence: .sisyphus/evidence/task-3-retry-error.txt
  ```

  **Commit**: YES (groups with 1-2, 4-6)
  - Message: `feat(benchmark): LLM evaluation adapter`

---

- [ ] 4. 数据集适配器基类

  **What to do**:
  - 创建 `benchmark/adapters/base.py`
  - 定义抽象基类 `DatasetAdapter`：
    - `load(path: str) -> list[TaskDefinition]` — 加载数据集
    - `validate(task: TaskDefinition) -> bool` — 验证题目格式
    - `get_dimension() -> str` — 返回所属维度
  - 创建 `benchmark/adapters/__init__.py` 注册表
  - 统一 TaskDefinition 格式：task_id, prompt, expected_output, metadata, test_cases（可选）

  **Must NOT do**:
  - 不要实现任何具体数据集适配器
  - 不要硬编码任何数据集路径

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 抽象基类定义，接口清晰
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5-6)
  - **Blocks**: Tasks 7-11
  - **Blocked By**: Task 2

  **References**:
  - `benchmark/models/schemas.py:TaskDefinition` — 题目定义模型
  - OpenCompass dataset pattern — YAML 配置驱动的数据集加载

  **Acceptance Criteria**:
  - [ ] `DatasetAdapter` 抽象基类包含 load、validate、get_dimension 方法
  - [ ] 尝试实例化抽象类抛出 TypeError
  - [ ] 具体子类必须实现所有抽象方法

  **QA Scenarios**:
  ```
  Scenario: 抽象基类验证
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.base import DatasetAdapter; DatasetAdapter()"
    Expected: 抛出 TypeError（抽象类不可实例化）
    Evidence: .sisyphus/evidence/task-4-abstract.txt

  Scenario: 子类实现验证
    Tool: Bash
    Steps:
      1. 创建一个最小实现子类，实现所有方法返回 mock 数据
      2. 调用 load() 验证返回 TaskDefinition 列表
    Expected: 返回合法的 TaskDefinition 列表
    Evidence: .sisyphus/evidence/task-4-subclass.txt
  ```

  **Commit**: YES (groups with 1-3, 5-6)
  - Message: `feat(benchmark): dataset adapter base class`

---

- [ ] 5. 评分引擎基类 + 两阶段评分管道

  **What to do**:
  - 创建 `benchmark/scorers/base.py`
  - 定义抽象基类 `BaseScorer`：
    - `score(model_output: str, expected: str, task: TaskDefinition) -> ScoreResult` — 评分
    - `get_metric_name() -> str` — 返回指标名称
  - 定义 `ScoreResult` 数据模型：score (0-100), passed (bool), details (dict), reasoning (str)
  - 定义两阶段评分管道 `TwoStageScoringPipeline`：
    - Stage 1: `compute_functional_score(model_output, expected, task) -> float (0-100)`
    - 短路规则：functional_score=0 → 直接返回 0，不调用 LLM Judge
    - Stage 2: `compute_quality_score(model_output, task, rubric) -> float (0-100)`（仅当 functional_score > 0）
    - 聚合：`final_score = functional_score * auto_weight + quality_score * judge_weight`
  - 定义维度默认权重配置：
    ```python
    DIMENSION_WEIGHTS = {
        "frontend-dev": {"auto": 0.8, "judge": 0.2},
        "backend-dev": {"auto": 0.8, "judge": 0.2},
        "system-architecture": {"auto": 0.2, "judge": 0.8},
        "tool-use-agentic": {"auto": 0.5, "judge": 0.5},
        "reasoning": {"auto": 0.8, "judge": 0.2},
    }
    ```
  - 创建 `benchmark/scorers/__init__.py` 注册表

  **Must NOT do**:
  - 不要实现任何具体评分器（functional 和 quality 的具体实现由 Task 13/14 负责）
  - 不要硬编码任何评分逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 抽象基类 + 管道定义
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4, 6-7)
  - **Blocks**: Tasks 13-14
  - **Blocked By**: Task 2

  **References**:
  - `benchmark/models/schemas.py:EvalResult` — 评分结果存储模型
  - `docs/superpowers/specs/2026-03-31-llm-benchmark-design.md` — 两阶段评分设计参考

  **Acceptance Criteria**:
  - [ ] `BaseScorer` 抽象基类包含 score、get_metric_name 方法
  - [ ] `ScoreResult` 数据模型包含 score (0-100), passed, details, reasoning 字段
  - [ ] `TwoStageScoringPipeline` 实现短路规则（functional_score=0 跳过 Judge）
  - [ ] 维度权重配置正确定义
  - [ ] 抽象类不可直接实例化

  **QA Scenarios**:
  ```
  Scenario: 短路规则验证
    Tool: Bash
    Steps:
      1. python -c "
from benchmark.scorers.base import TwoStageScoringPipeline
p = TwoStageScoringPipeline(auto_weight=0.8, judge_weight=0.2)
result = p.compute(0, None)  # functional_score=0
assert result == 0, f'Expected 0, got {result}'
print('Short-circuit rule works')
"
    Expected: 输出 "Short-circuit rule works"
    Evidence: .sisyphus/evidence/task-5-short-circuit.txt

  Scenario: 评分结果模型验证
    Tool: Bash
    Steps:
      1. python -c "from benchmark.scorers.base import ScoreResult; r = ScoreResult(score=85, passed=True, details={'metric': 'accuracy'}, reasoning='Correct'); print(r.model_dump())"
    Expected: 输出合法 JSON
    Evidence: .sisyphus/evidence/task-5-scoreresult.txt
  ```

  **Commit**: YES (groups with 1-4, 6-7)
  - Message: `feat(benchmark): scorer engine base and two-stage pipeline`

---

- [ ] 6. 配置系统 + CLI 框架

  **What to do**:
  - 创建 `benchmark/config.py` — 配置管理
    - 支持从 `benchmark/configs/` 读取 YAML 配置
    - 默认配置：model、temperature、max_tokens、max_retries、timeout
  - 创建 `benchmark/configs/default.yaml` — 默认配置文件
    ```yaml
    model: "glm-4.7"
    temperature: 0.0
    max_tokens: 4096
    max_retries: 3
    timeout: 300
    dataset_root: "benchmark/datasets"
    
    dimensions:
      frontend-dev:
        adapter: "frontcode"
        auto_weight: 0.8
        judge_weight: 0.2
      backend-dev:
        adapter: "swebench"
        auto_weight: 0.8
        judge_weight: 0.2
      system-architecture:
        adapter: "mmlu"
        auto_weight: 0.2
        judge_weight: 0.8
      tool-use-agentic:
        adapter: "agentbench"
        auto_weight: 0.5
        judge_weight: 0.5
      reasoning:
        adapter: "gsm8k"
        auto_weight: 0.8
        judge_weight: 0.2
    
    difficulty_weights:
      easy: 1.0
      medium: 1.5
      hard: 2.0
    ```
  - 创建 `benchmark/cli.py` — CLI 入口（使用 `click` 或 `argparse`）
  - 实现 `__main__.py` 入口：`python -m benchmark`
  - CLI 命令框架：
    - `benchmark evaluate --dimension <dim> --model <model> --samples <n>`
    - `benchmark list-datasets`
    - `benchmark list-models`
    - `benchmark results --run-id <id>`
    - `benchmark export --format <fmt> --output <path>`

  **Must NOT do**:
  - 不要实现 evaluate 命令的实际逻辑
  - 不要添加过多子命令（只保留核心 5 个）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: CLI 框架搭建，使用 argparse/click
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-5)
  - **Blocks**: Tasks 15-19
  - **Blocked By**: None

  **References**:
  - `pyproject.toml` — 需在 [project.scripts] 中注册 CLI 入口

  **Acceptance Criteria**:
  - [ ] `python -m benchmark --help` 显示所有可用命令
  - [ ] `python -m benchmark evaluate --help` 显示 evaluate 命令参数
  - [ ] 配置文件可从 YAML 加载默认值

  **QA Scenarios**:
  ```
  Scenario: CLI help 验证
    Tool: Bash
    Steps:
      1. python -m benchmark --help
    Expected: 输出包含 evaluate, list-datasets, list-models, results, export 命令
    Evidence: .sisyphus/evidence/task-6-cli-help.txt

  Scenario: 配置加载验证
    Tool: Bash
    Steps:
      1. 创建默认 configs/default.yaml
      2. python -c "from benchmark.config import load_config; c = load_config(); print(c)"
    Expected: 输出包含 model、temperature、max_tokens 等默认配置
    Evidence: .sisyphus/evidence/task-6-config.txt
  ```

  **Commit**: YES (groups with 1-5, 7)
  - Message: `feat(benchmark): config system and CLI framework`

---

- [ ] 7. 评分聚合（维度权重 + 难度加权）

  **What to do**:
  - 创建 `benchmark/scorers/aggregator.py`
  - 实现难度加权：
    ```
    dimension_score = sum(task_score * difficulty_weight) / sum(difficulty_weight)
    difficulty: easy=1.0, medium=1.5, hard=2.0
    ```
  - 实现维度总分计算：
    ```
    overall_score = avg(frontend, backend, architecture, agentic, reasoning)
    ```
  - 支持 CLI 参数自定义维度权重：`--dimension-weights "frontend=0.3,backend=0.3,..."`
  - 任务难度从 metadata 读取（默认 medium=1.5）
  - 返回 `AggregationResult`：dimension_scores (dict), overall_score (float), task_count (int)

  **Must NOT do**:
  - 不要实现任何评分逻辑（只负责聚合已有分数）
  - 不要实现数据库查询（由调用方传入分数列表）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯数学计算，逻辑简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-6)
  - **Blocks**: Task 15
  - **Blocked By**: Task 2, 5

  **References**:
  - `benchmark/scorers/base.py` — 评分基类
  - `docs/superpowers/specs/2026-03-31-llm-benchmark-design.md` — 难度加权设计参考

  **Acceptance Criteria**:
  - [ ] 难度加权正确计算（easy=1.0, medium=1.5, hard=2.0）
  - [ ] 维度分数按难度加权平均
  - [ ] 总体分数为各维度等权平均
  - [ ] CLI 参数可自定义维度权重

  **QA Scenarios**:
  ```
  Scenario: 难度加权计算
    Tool: Bash
    Steps:
      1. python -c "
from benchmark.scorers.aggregator import compute_dimension_score
scores = [
    {'score': 80, 'difficulty': 'easy'},
    {'score': 60, 'difficulty': 'medium'},
    {'score': 90, 'difficulty': 'hard'},
]
result = compute_dimension_score(scores)
print(f'Dimension score: {result}')
# Expected: (80*1.0 + 60*1.5 + 90*2.0) / (1.0+1.5+2.0) = 350/4.5 = 77.78
"
    Expected: 输出约 77.78
    Evidence: .sisyphus/evidence/task-7-weighting.txt

  Scenario: 总体分数计算
    Tool: Bash
    Steps:
      1. python -c "
from benchmark.scorers.aggregator import compute_overall_score
dim_scores = {'frontend': 80, 'backend': 70, 'architecture': 90, 'agentic': 60, 'reasoning': 85}
result = compute_overall_score(dim_scores)
print(f'Overall score: {result}')
# Expected: (80+70+90+60+85)/5 = 77.0
"
    Expected: 输出 77.0
    Evidence: .sisyphus/evidence/task-7-overall.txt
  ```

  **Commit**: YES (groups with 1-6)
  - Message: `feat(benchmark): scoring aggregation with difficulty weighting`

---

- [ ] 8. SWE-bench / BigCodeBench 适配器

  **What to do**:
  - 创建 `benchmark/adapters/swebench_adapter.py`
  - SWE-bench 数据加载：
    - 如果本地存在 `benchmark/datasets/swebench/`，从本地 JSONL 加载
    - 如果本地不存在，使用 `datasets` 库从 HuggingFace 下载并缓存到本地：
      ```python
      from datasets import load_dataset
      dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
      dataset.save_to_disk("benchmark/datasets/swebench/")
      ```
    - 提取字段：task_id, problem_statement (prompt), repo, base_commit, test_patch
    - v1 使用 lite/verified 子集（约 300 题）
  - 创建 `benchmark/adapters/bigcodebench_adapter.py`
  - BigCodeBench 数据加载：
    - 从 JSONL 或 HuggingFace 加载
    - 提取字段：task_id, prompt (docstring/instruct), test, entry_point
  - 两个适配器都继承 DatasetAdapter 基类
  - 数据缓存到 `benchmark/datasets/swebench/` 和 `benchmark/datasets/bigcodebench/`

  **Must NOT do**:
  - 不要实现代码执行逻辑（由评分器负责）
  - 不要下载完整数据集（使用 lite 子集）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 涉及外部数据集下载、格式解析、字段映射
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8-13)
  - **Blocks**: Task 14
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - SWE-bench: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
  - BigCodeBench: https://huggingface.co/datasets/bigcode/bigcodebench

  **Acceptance Criteria**:
  - [ ] SWE-bench 适配器可加载 lite 子集，返回 TaskDefinition 列表
  - [ ] BigCodeBench 适配器可加载数据，返回含 test_cases 的 TaskDefinition
  - [ ] 每个适配器通过 validate() 验证所有题目

  **QA Scenarios**:
  ```
  Scenario: SWE-bench 数据加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.swebench_adapter import SWEbenchAdapter; a = SWEbenchAdapter(); tasks = a.load('benchmark/datasets/swebench/'); print(f'Loaded {len(tasks)} tasks'); print(tasks[0].task_id)"
    Expected: 输出加载的题目数量和首个 task_id
    Evidence: .sisyphus/evidence/task-7-swebench-load.txt

  Scenario: BigCodeBench 数据验证
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter; a = BigCodeBenchAdapter(); tasks = a.load('benchmark/datasets/bigcodebench/'); assert all(a.validate(t) for t in tasks); print('All tasks valid')"
    Expected: 输出 "All tasks valid"
    Evidence: .sisyphus/evidence/task-7-bigcodebench-validate.txt
  ```

  **Commit**: YES (groups with 8-11)
  - Message: `feat(benchmark): SWE-bench and BigCodeBench dataset adapters`

---

- [ ] 9. AgentBench 适配器

  **What to do**:
  - 创建 `benchmark/adapters/agentbench_adapter.py`
  - AgentBench 数据加载：
    - 从 GitHub 或 HuggingFace 加载
    - v1 仅支持 DB（数据库）和 OS（操作系统）子环境
    - 提取字段：task_id, instruction, expected_result, environment_config
  - 支持多轮交互格式（agent 需要多次 tool call）
  - 数据缓存到 `benchmark/datasets/agentbench/`

  **Must NOT do**:
  - 不要实现 Docker 容器环境（v1 用 subprocess 模拟）
  - 不要支持全部 8 个环境（v1 只支持 DB + OS）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 多轮交互格式、环境配置解析
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 9-13)
  - **Blocks**: Task 14
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - AgentBench: https://github.com/THUDM/AgentBench

  **Acceptance Criteria**:
  - [ ] AgentBench 适配器可加载 DB/OS 子集
  - [ ] TaskDefinition 包含多轮交互所需的 environment_config
  - [ ] 每个题目包含 task_id、instruction、expected_result

  **QA Scenarios**:
  ```
  Scenario: AgentBench 数据加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.agentbench_adapter import AgentBenchAdapter; a = AgentBenchAdapter(); tasks = a.load('benchmark/datasets/agentbench/'); print(f'Loaded {len(tasks)} tasks, env: {tasks[0].metadata.get(\"environment\")}')"
    Expected: 输出题目数量和环境类型
    Evidence: .sisyphus/evidence/task-8-agentbench-load.txt
  ```

  **Commit**: YES (groups with 7, 9-11)
  - Message: `feat(benchmark): AgentBench dataset adapter`

---

- [ ] 10. GSM8K / MATH 适配器

  **What to do**:
  - 创建 `benchmark/adapters/gsm8k_adapter.py`
  - GSM8K 数据加载：
    - 从 HuggingFace 加载 `openai/gsm8k` 数据集
    - 提取字段：question (prompt), answer (expected)
    - 解析 answer 中的最终数值（格式：`#### 42`）
  - 创建 `benchmark/adapters/math_adapter.py`
  - MATH 数据加载：
    - 从 HuggingFace 加载 `hendrycks/competition_math` 数据集
    - 提取字段：problem (prompt), solution (expected)
  - 数据缓存到 `benchmark/datasets/gsm8k/` 和 `benchmark/datasets/math/`

  **Must NOT do**:
  - 不要实现 LaTeX 渲染（v1 纯文本处理）
  - 不要下载完整 MATH 数据集（使用 test 子集）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单 JSON/JSONL 格式解析
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7-8, 10-13)
  - **Blocks**: Task 14
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - GSM8K: https://huggingface.co/datasets/openai/gsm8k

  **Acceptance Criteria**:
  - [ ] GSM8K 适配器可加载数据，正确解析 #### 后的答案
  - [ ] MATH 适配器可加载数据，提取最终答案
  - [ ] 两个适配器都通过 validate() 验证

  **QA Scenarios**:
  ```
  Scenario: GSM8K 答案解析
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.gsm8k_adapter import GSM8KAdapter; a = GSM8KAdapter(); tasks = a.load('benchmark/datasets/gsm8k/'); print(tasks[0].expected_output); assert '####' not in tasks[0].expected_output"
    Expected: 输出纯数值答案，不含 #### 标记
    Evidence: .sisyphus/evidence/task-9-gsm8k-parse.txt
  ```

  **Commit**: YES (groups with 7-8, 10-11)
  - Message: `feat(benchmark): GSM8K and MATH dataset adapters`

---

- [ ] 11. MMLU 适配器

  **What to do**:
  - 创建 `benchmark/adapters/mmlu_adapter.py`
  - MMLU 数据加载：
    - 从 HuggingFace 加载 `cais/mmlu` 数据集
    - v1 只加载 5 个子集：philosophy, computer_science, mathematics, physics, engineering
    - 提取字段：question, choices (A/B/C/D), answer (letter)
    - 转换为 few-shot prompt 格式
  - 数据缓存到 `benchmark/datasets/mmlu/`

  **Must NOT do**:
  - 不要加载全部 57 个子集
  - 不要实现复杂的 prompt 模板引擎

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 标准多选题格式，解析简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7-9, 11-13)
  - **Blocks**: Task 14
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - MMLU: https://huggingface.co/datasets/cais/mmlu

  **Acceptance Criteria**:
  - [ ] MMLU 适配器可加载指定子集
  - [ ] 题目格式为 "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:"
  - [ ] expected_output 为单个字母 (A/B/C/D)

  **QA Scenarios**:
  ```
  Scenario: MMLU 多选题加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.mmlu_adapter import MMLUAdapter; a = MMLUAdapter(); tasks = a.load('benchmark/datasets/mmlu/', subsets=['philosophy']); print(tasks[0].prompt[:200]); print('Answer:', tasks[0].expected_output)"
    Expected: 输出包含 A/B/C/D 选项和单个字母答案
    Evidence: .sisyphus/evidence/task-10-mmlu-load.txt
  ```

  **Commit**: YES (groups with 7-9, 11)
  - Message: `feat(benchmark): MMLU dataset adapter`

---

- [ ] 12. FrontCode 数据集 + 适配器

  **What to do**:
  - 创建 `benchmark/datasets/frontcode/` 目录
  - 创建 `benchmark/datasets/frontcode/tasks.jsonl` — 自建前端题目
  - 至少包含 20 道前端题目，覆盖：
    - HTML 结构生成（5 题）
    - CSS 样式实现（5 题）
    - JavaScript 交互逻辑（5 题）
    - React 组件生成（5 题）
  - 每道题目格式：task_id, dimension, prompt, expected_output (参考代码), rubric_criteria
  - 创建 `benchmark/adapters/frontcode_adapter.py`
  - 实现 DatasetAdapter 基类

  **Must NOT do**:
  - 不要创建复杂的前端项目结构
  - 不要包含浏览器测试

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: JSONL 文件创建 + 简单适配器
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7-10, 12-13)
  - **Blocks**: Task 14
  - **Blocked By**: Task 4

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - `benchmark/datasets/frontcode/` — 新建数据集目录

  **Acceptance Criteria**:
  - [ ] `tasks.jsonl` 包含至少 20 道题目
  - [ ] 覆盖 HTML/CSS/JS/React 四个类别
  - [ ] 适配器可加载并验证所有题目

  **QA Scenarios**:
  ```
  Scenario: FrontCode 数据集加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.adapters.frontcode_adapter import FrontCodeAdapter; a = FrontCodeAdapter(); tasks = a.load('benchmark/datasets/frontcode/'); print(f'Loaded {len(tasks)} tasks'); categories = set(t.metadata.get('category') for t in tasks); print(f'Categories: {categories}')"
    Expected: 输出 >= 20 题目，包含 html/css/js/react 类别
    Evidence: .sisyphus/evidence/task-11-frontcode-load.txt
  ```

  **Commit**: YES (groups with 7-10)
  - Message: `feat(benchmark): FrontCode dataset and adapter`

---

- [ ] 13. 执行验证评分器（functional_score）

  **What to do**:
  - 创建 `benchmark/scorers/execution_scorer.py`
  - 用于 backend-dev 维度（SWE-bench / BigCodeBench）
  - 评分逻辑：
    - 将 LLM 生成的代码与题目测试用例结合
    - 使用 subprocess 在隔离环境中执行测试
    - 设置 timeout（默认 30s）和内存限制
    - 捕获 stdout/stderr 判断 pass/fail
  - 返回 ScoreResult：score (0 or 1), passed (bool), details (execution_time, error_message)
  - 支持 pass@k（生成 k 个样本，统计通过率）

  **Must NOT do**:
  - 不要使用 Docker（v1 用 subprocess 沙箱）
  - 不要执行网络请求相关的代码
  - 不要允许文件系统写操作（限制在临时目录）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 代码执行沙箱、超时控制、安全限制
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7-11, 13)
  - **Blocks**: Task 14
  - **Blocked By**: Task 5

  **References**:
  - `benchmark/scorers/base.py:BaseScorer` — 基类接口
  - `benchmark/adapters/bigcodebench_adapter.py` — 提供 test_cases 字段
  - HumanEval execution pattern: https://github.com/openai/human-eval

  **Acceptance Criteria**:
  - [ ] 正确代码执行后返回 score=1, passed=True
  - [ ] 错误代码执行后返回 score=0, passed=False
  - [ ] 超时代码在 timeout 后终止，返回 passed=False
  - [ ] 无限循环代码被正确终止

  **QA Scenarios**:
  ```
  Scenario: 正确代码通过测试
    Tool: Bash
    Steps:
      1. python -c "
from benchmark.scorers.execution_scorer import ExecutionScorer
s = ExecutionScorer(timeout=10)
code = 'def add(a, b): return a + b'
result = s.score(code, expected='', test_cases=['assert add(1, 2) == 3'])
print(result.model_dump())
"
    Expected: score=1, passed=True
    Evidence: .sisyphus/evidence/task-12-pass.txt

  Scenario: 超时处理
    Tool: Bash
    Steps:
      1. python -c "
from benchmark.scorers.execution_scorer import ExecutionScorer
s = ExecutionScorer(timeout=2)
code = 'import time; time.sleep(100)'
result = s.score(code, expected='', test_cases=[])
print(result.model_dump())
"
    Expected: passed=False, details 包含 timeout 信息
    Evidence: .sisyphus/evidence/task-12-timeout.txt

  Scenario: 错误代码处理
    Tool: Bash
    Steps:
      1. 提供语法错误的代码
      2. 验证返回 passed=False 和错误信息
    Expected: passed=False, details 包含 SyntaxError
    Evidence: .sisyphus/evidence/task-12-error.txt
  ```

  **Commit**: YES (groups with 13)
  - Message: `feat(benchmark): execution-based scorer`

---

- [ ] 14. LLM-as-judge 评分器（quality_score）

  **What to do**:
  - 创建 `benchmark/scorers/llm_judge_scorer.py`
  - 用于 frontend-dev、system-architecture 维度
  - 评分逻辑：
    - 构建 judge prompt：包含题目、模型输出、评分准则
    - 调用 LLM（temperature=0）进行评分
    - 解析 LLM 返回的分数（1-10 或 1-5）
    - 支持多维度 rubric 评分
  - 创建 `benchmark/scorers/rubrics/` 目录
  - 创建 rubric 配置文件：
    - `frontend_code_quality.yaml` — 前端代码质量评分准则
    - `architecture_design.yaml` — 架构设计评分准则
    - `reasoning_quality.yaml` — 推理质量评分准则
  - 返回 ScoreResult：score (1-10), passed (score >= threshold), details (rubric_scores), reasoning (judge 的理由)

  **Must NOT do**:
  - 不要实现复杂的 prompt 优化逻辑
  - 不要使用多个 judge 模型（v1 用单一模型）
  - 不要实现 judge 一致性校验

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: prompt 设计、rubric 配置、结果解析
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7-12)
  - **Blocks**: Task 14
  - **Blocked By**: Task 5

  **References**:
  - `benchmark/scorers/base.py:BaseScorer` — 基类接口
  - `benchmark/core/llm_adapter.py` — 调用 LLM
  - Prometheus judge pattern: https://github.com/PrometheusEvaluates/Prometheus

  **Acceptance Criteria**:
  - [ ] 可从 YAML 加载 rubric 配置
  - [ ] judge prompt 包含题目、输出、评分准则
  - [ ] 可解析 LLM 返回的分数
  - [ ] 解析失败时有 fallback 机制

  **QA Scenarios**:
  ```
  Scenario: 基本 LLM-as-judge 评分
    Tool: Bash
    Steps:
      1. python -c "
from benchmark.scorers.llm_judge_scorer import LLMJudgeScorer
s = LLMJudgeScorer(judge_model='glm-4.7')
result = s.score('print(\"Hello World\")', expected='输出 Hello World', rubric='correctness')
print(result.model_dump())
"
    Expected: score 为 1-10 的整数，包含 reasoning
    Evidence: .sisyphus/evidence/task-13-judge-basic.txt

  Scenario: Rubric 配置加载
    Tool: Bash
    Steps:
      1. python -c "from benchmark.scorers.llm_judge_scorer import load_rubric; r = load_rubric('frontend_code_quality'); print(r)"
    Expected: 输出 rubric 配置，包含 criteria 和 scoring scale
    Evidence: .sisyphus/evidence/task-13-rubric-load.txt

  Scenario: 解析失败 fallback
    Tool: Bash
    Steps:
      1. 模拟 LLM 返回非数字分数
      2. 验证 fallback 机制（返回默认分数或重试）
    Expected: 不抛出异常，返回合理分数
    Evidence: .sisyphus/evidence/task-13-judge-fallback.txt
  ```

  **Commit**: YES (groups with 12)
  - Message: `feat(benchmark): LLM-as-judge scorer`

---

- [ ] 15. CLI 评测命令完整实现

  **What to do**:
  - 完善 `benchmark/cli.py` 中的 `evaluate` 命令
  - 实现维度到数据集/评分器的映射配置：
    ```python
    DIMENSION_MAPPING = {
        "frontend-dev": {
            "adapter": ["frontcode"],  # FrontCode 自建题库
            "scorers": ["llm_judge"],  # LLM-as-judge（代码质量评估）
            "auto_weight": 0.8,
            "judge_weight": 0.2,
        },
        "backend-dev": {
            "adapter": ["swebench", "bigcodebench"],  # SWE-bench 或 BigCodeBench
            "scorers": ["execution"],  # 执行验证（test pass）
            "auto_weight": 0.8,
            "judge_weight": 0.2,
        },
        "system-architecture": {
            "adapter": ["mmlu"],  # MMLU 子集（philosophy, CS, math, physics, engineering）
            "scorers": ["exact_match", "llm_judge"],  # exact match for choice, LLM judge for reasoning
            "auto_weight": 0.2,
            "judge_weight": 0.8,
        },
        "tool-use-agentic": {
            "adapter": ["agentbench"],  # AgentBench DB/OS 环境
            "scorers": ["agent_loop"],  # Agent Loop + 工具调用验证
            "auto_weight": 0.5,
            "judge_weight": 0.5,
        },
        "reasoning": {
            "adapter": ["gsm8k", "math"],  # GSM8K 或 MATH
            "scorers": ["exact_match"],  # 答案精确匹配
            "auto_weight": 0.8,
            "judge_weight": 0.2,
        },
    }
    ```
  - 实现完整评测流程：
    1. 加载配置（model、dimension、samples）
    2. 根据 dimension 选择对应数据集适配器（从 DIMENSION_MAPPING）
    3. 根据 dimension 选择对应评分器（从 DIMENSION_MAPPING）
    4. 创建 EvalRun 记录
    5. 遍历题目：调用 LLM → Stage 1 functional_score → 短路检查 → Stage 2 quality_score（如需要）→ 保存 EvalResult
    6. 输出进度（使用 rich.progress）
    7. 汇总统计（维度分数、难度加权、总体分数）
  - 实现 `list-datasets`、`list-models`、`results` 命令

  **Must NOT do**:
  - 不要实现并行执行（v1 顺序）
  - 不要实现断点续传（由 Task 17 负责）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 核心编排逻辑，整合所有组件
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential after Wave 2)
  - **Blocks**: Tasks 16-20
  - **Blocked By**: Tasks 7-14

  **References**:
  - `benchmark/cli.py` — CLI 框架
  - `benchmark/adapters/` — 所有数据集适配器
  - `benchmark/scorers/` — 所有评分器
  - `benchmark/scorers/base.py:TwoStageScoringPipeline` — 两阶段评分
  - `benchmark/scorers/aggregator.py` — 难度加权
  - `benchmark/core/llm_adapter.py` — LLM 调用
  - `benchmark/models/database.py` — 结果存储

  **Acceptance Criteria**:
  - [ ] `python -m benchmark evaluate --dimension reasoning --model glm-4.7 --samples 5` 成功运行
  - [ ] 输出进度条和每题结果
  - [ ] 两阶段评分正确工作（functional_score=0 时跳过 Judge）
  - [ ] 结果写入 SQLite
  - [ ] 汇总统计正确计算（含难度加权）

  **QA Scenarios**:
  ```
  Scenario: 完整评测流程（reasoning 维度）
    Tool: Bash
    Steps:
      1. python -m benchmark evaluate --dimension reasoning --model glm-4.7 --samples 3
    Expected: 
      - 输出 "Loading GSM8K dataset..."
      - 显示进度条 3/3
      - 输出汇总统计（平均分、通过率）
      - SQLite 中有 3 条结果
    Evidence: .sisyphus/evidence/task-14-eval-reasoning.txt

  Scenario: 短路规则验证
    Tool: Bash
    Steps:
      1. 提供会触发 functional_score=0 的题目
      2. 验证 LLM Judge 未被调用
    Expected: 日志不包含 "LLM Judge" 相关输出
    Evidence: .sisyphus/evidence/task-14-short-circuit.txt

  Scenario: 完整评测流程（backend-dev 维度）
    Tool: Bash
    Steps:
      1. python -m benchmark evaluate --dimension backend-dev --model glm-4.7 --samples 2
    Expected:
      - 输出执行验证过程
      - 显示 pass/fail 结果
      - SQLite 中有 2 条结果
    Evidence: .sisyphus/evidence/task-14-eval-backend.txt

  Scenario: 无效维度处理
    Tool: Bash
    Steps:
      1. python -m benchmark evaluate --dimension invalid --model glm-4.7
    Expected: 输出错误信息 "Unknown dimension: invalid"，退出码非零
    Evidence: .sisyphus/evidence/task-14-invalid-dimension.txt

  Scenario: list-datasets 命令
    Tool: Bash
    Steps:
      1. python -m benchmark list-datasets
    Expected: 列出所有可用数据集（swebench, bigcodebench, agentbench, gsm8k, math, mmlu, frontcode）
    Evidence: .sisyphus/evidence/task-14-list-datasets.txt
  ```

  **Commit**: YES
  - Message: `feat(benchmark): complete evaluation CLI with two-stage scoring`

---

- [ ] 16. Agent Loop 实现（tool-use 维度）

  **What to do**:
  - 创建 `benchmark/core/agent_loop.py`
  - 实现多轮工具调用循环：
    ```python
    async def run_agent_loop(
        task: Task,
        model: ModelProvider,
        tools: list[dict],
        max_iterations: int = 10,
    ) -> ExecutionResult:
        messages = [{"role": "user", "content": task.prompt}]
        all_tool_calls = []
        for i in range(max_iterations):
            response = await model.generate(messages, tools=tools)
            if not response.tool_calls:
                break  # 模型返回纯文本，视为最终答案
            # 执行工具调用，收集结果
            # 将结果添加到 messages
        return ExecutionResult(...)
    ```
  - 定义 ToolCall / ToolResult 数据模型
  - 实现轻量工具模拟器 `ToolSimulator`：
    - `read_file`, `write_file`, `list_directory`, `run_command`, `search_code`
    - 每个工具在 subprocess 沙箱中执行
  - 终止条件：模型不发起工具调用 或 达到 max_iterations
  - 工具调用失败不自动重试（观察模型是否自行恢复是评估指标之一）
  - 记录完整工具调用链（iteration, tool_name, args, result_success）

  **Must NOT do**:
  - 不要实现 Docker 容器（v1 用 subprocess）
  - 不要实现复杂的 agent 框架
  - 不要自动重试失败的工具调用

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 多轮循环、工具模拟、状态管理
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 15, 17-19)
  - **Blocks**: Task 20
  - **Blocked By**: Task 14

  **References**:
  - `benchmark/core/llm_adapter.py` — LLM 调用
  - `benchmark/models/schemas.py` — 数据模型
  - `docs/superpowers/specs/2026-03-31-llm-benchmark-design.md` — Agent Loop 定义

  **Acceptance Criteria**:
  - [ ] Agent loop 在模型不发起工具调用时终止
  - [ ] Agent loop 在达到 max_iterations 时终止
  - [ ] 工具调用结果正确反馈给模型
  - [ ] 工具调用失败不自动重试
  - [ ] 完整工具调用链被记录

  **QA Scenarios**:
  ```
  Scenario: 基本 Agent Loop
    Tool: Bash
    Steps:
      1. 创建一个简单任务（需要 read_file + write_file）
      2. 运行 agent_loop
    Expected: 工具调用被记录，循环在模型返回纯文本时终止
    Evidence: .sisyphus/evidence/task-16-basic-loop.txt

  Scenario: 最大迭代限制
    Tool: Bash
    Steps:
      1. 设置 max_iterations=3
      2. 模拟模型持续发起工具调用
    Expected: 循环在第 3 次迭代后终止
    Evidence: .sisyphus/evidence/task-16-max-iterations.txt
  ```

  **Commit**: YES (groups with 15, 17-19)
  - Message: `feat(benchmark): agent loop for tool-use dimension`

---

- [ ] 17. 评估恢复机制

  **What to do**:
  - 创建 `benchmark/core/resume.py`
  - 实现评估恢复逻辑：
    - 在开始评测前，检查 SQLite 中是否已有相同 run_id 的结果
    - 已有结果跳过，只评测未完成的题目
    - 支持 `--resume` 参数自动恢复上次未完成的评测
    - 支持 `--force` 参数强制重新评测
  - 生成唯一 run_id（基于 model + dimension + dataset + timestamp hash）
  - 在 EvalRun 中记录 completed_tasks 列表

  **Must NOT do**:
  - 不要实现复杂的检查点序列化
  - 不要修改已有结果（只追加新结果）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 逻辑简单，查询 SQLite 去重
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 16-17)
  - **Blocks**: Task 18
  - **Blocked By**: Task 14

  **References**:
  - `benchmark/models/database.py` — SQLite 操作
  - `benchmark/cli.py` — CLI 入口

  **Acceptance Criteria**:
  - [ ] 重复运行相同评测时，跳过已完成题目
  - [ ] `--resume` 参数可恢复中断的评测
  - [ ] `--force` 参数可强制重新评测
  - [ ] 不会产生重复结果

  **QA Scenarios**:
  ```
  Scenario: 自动跳过已完成题目
    Tool: Bash
    Steps:
      1. 运行评测 3 题
      2. 再次运行相同评测
    Expected: 第二次运行输出 "3 tasks already completed, skipping"
    Evidence: .sisyphus/evidence/task-15-skip-completed.txt

  Scenario: 评估恢复
    Tool: Bash
    Steps:
      1. 运行评测 10 题，中途 Ctrl+C 中断（模拟：只运行 5 题）
      2. 使用 --resume 参数恢复
    Expected: 恢复后只评测剩余 5 题，总计 10 条结果
    Evidence: .sisyphus/evidence/task-15-resume.txt
  ```

  **Commit**: YES (groups with 16-17)
  - Message: `feat(benchmark): evaluation resume mechanism`

---

- [ ] 18. 自定义题目系统（YAML 格式）

  **What to do**:
  - 创建 `benchmark/datasets/custom/` 目录
  - 定义自定义题目 YAML Schema（借鉴 design doc）：
    ```yaml
    id: "custom-001"
    dimension: "reasoning"
    difficulty: "easy"
    prompt_file: "prompt.md"
    verify_script: "verify.py"
    scoring:
      auto_weight: 0.8
      judge_weight: 0.2
    source: "original"
    ```
  - 创建 `benchmark/adapters/custom_adapter.py`
  - 实现从 `benchmark/datasets/custom/` 加载所有 `.yaml` 文件
  - 每个题目目录包含：task.yaml, prompt.md, verify.py（可选）, judge-criteria.md（可选）
  - 在 `list-datasets` 命令中显示自定义题目数量
  - 支持 `--custom` 参数仅评测自定义题目

  **Must NOT do**:
  - 不要创建管理 UI（手动编辑 YAML 文件）
  - 不要实现题目验证/预览功能

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: YAML 文件加载 + 简单适配器
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 16-17, 19)
  - **Blocks**: Task 20
  - **Blocked By**: Task 14

  **References**:
  - `benchmark/adapters/base.py:DatasetAdapter` — 基类接口
  - `benchmark/models/schemas.py:TaskDefinition` — 题目模型
  - `docs/superpowers/specs/2026-03-31-llm-benchmark-design.md` — Task YAML Schema

  **Acceptance Criteria**:
  - [ ] 可在 `custom/` 目录下放置题目目录（含 task.yaml）
  - [ ] 适配器自动加载所有 YAML 文件
  - [ ] `--custom` 参数仅评测自定义题目
  - [ ] 题目格式错误时有明确提示

  **QA Scenarios**:
  ```
  Scenario: 自定义题目加载
    Tool: Bash
    Steps:
      1. 创建 benchmark/datasets/custom/test-001/ 包含 task.yaml 和 prompt.md
      2. python -m benchmark evaluate --custom --model glm-4.7
    Expected: 加载自定义题目并完成评测
    Evidence: .sisyphus/evidence/task-18-custom-eval.txt

  Scenario: 格式错误处理
    Tool: Bash
    Steps:
      1. 创建格式错误的 YAML（缺少 required 字段）
      2. 运行加载命令
    Expected: 输出警告信息，跳过无效题目
    Evidence: .sisyphus/evidence/task-18-invalid-format.txt
  ```

  **Commit**: YES (groups with 16-17, 19)
  - Message: `feat(benchmark): custom question system (YAML format)`

---

- [ ] 19. 结果导出（JSON/CSV）

  **What to do**:
  - 在 `benchmark/cli.py` 中完善 `export` 命令
  - 支持导出格式：
    - `--format json` — 导出为 JSON 文件
    - `--format csv` — 导出为 CSV 文件
  - 支持过滤条件：
    - `--dimension <dim>` — 按维度过滤
    - `--model <model>` — 按模型过滤
    - `--run-id <id>` — 按评测批次过滤
  - 导出内容包含：run_id, model, dimension, task_id, score, passed, execution_time, timestamp
  - 创建 `benchmark/core/exporter.py`

  **Must NOT do**:
  - 不要实现 PDF 导出
  - 不要实现复杂的报告模板

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 标准数据导出，逻辑简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 15-16)
  - **Blocks**: Task 18
  - **Blocked By**: Task 14

  **References**:
  - `benchmark/models/database.py` — SQLite 查询
  - `benchmark/cli.py` — CLI 命令

  **Acceptance Criteria**:
  - [ ] `python -m benchmark export --format json --output results.json` 生成合法 JSON
  - [ ] `python -m benchmark export --format csv --output results.csv` 生成合法 CSV
  - [ ] 过滤条件正确工作

  **QA Scenarios**:
  ```
  Scenario: JSON 导出
    Tool: Bash
    Steps:
      1. python -m benchmark export --format json --output /tmp/test_results.json
      2. python -c "import json; data = json.load(open('/tmp/test_results.json')); print(f'Exported {len(data)} results')"
    Expected: 合法 JSON，包含所有评测结果
    Evidence: .sisyphus/evidence/task-17-export-json.txt

  Scenario: CSV 导出 + 过滤
    Tool: Bash
    Steps:
      1. python -m benchmark export --format csv --dimension reasoning --output /tmp/reasoning.csv
      2. head -5 /tmp/reasoning.csv
    Expected: CSV 表头 + reasoning 维度数据
    Evidence: .sisyphus/evidence/task-17-export-csv.txt
  ```

  **Commit**: YES (groups with 15-16)
  - Message: `feat(benchmark): result export (JSON/CSV)`

---

- [ ] 20. Streamlit 主界面

  **What to do**:
  - 创建 `benchmark/visualization/app.py` — Streamlit 主应用
  - 页面布局：
    - 侧边栏：过滤器（model、dimension、date range）
    - 主区域：
      - 总览卡片：评测批次数量、各维度平均分
      - 结果表格：可排序、可搜索
      - 简单图表：各维度分数柱状图（使用 st.bar_chart）
  - 功能：
    - 从 SQLite 加载数据
    - 按 model/dimension/date 过滤
    - 显示单题详情（题目、输出、分数、推理过程）
    - 导出按钮（调用 exporter）
  - 创建 `benchmark/visualization/__init__.py`
  - 空状态处理：无数据时显示友好提示

  **Must NOT do**:
  - 不要实现实时更新（手动刷新）
  - 不要实现复杂图表（只用 st.bar_chart / st.line_chart）
  - 不要实现认证/多用户

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Web UI 开发，需要良好的布局和交互
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (with Task 21)
  - **Blocks**: Final Verification
  - **Blocked By**: Tasks 14-19

  **References**:
  - `benchmark/models/database.py` — SQLite 数据查询
  - `benchmark/core/exporter.py` — 导出功能
  - Streamlit docs: https://docs.streamlit.io/ — 组件和布局

  **Acceptance Criteria**:
  - [ ] `streamlit run benchmark/visualization/app.py` 成功启动
  - [ ] 页面显示评测结果列表
  - [ ] 侧边栏过滤器工作正常
  - [ ] 空状态显示友好提示
  - [ ] 导出按钮可下载 JSON/CSV

  **QA Scenarios**:
  ```
  Scenario: 页面加载
    Tool: Bash
    Steps:
      1. streamlit run benchmark/visualization/app.py --server.headless true --server.port 8501 &
      2. sleep 5
      3. curl -s http://localhost:8501 | grep -o "LLM Benchmark"
    Expected: 输出 "LLM Benchmark"
    Evidence: .sisyphus/evidence/task-18-page-load.txt

  Scenario: 数据展示
    Tool: Bash
    Steps:
      1. 先运行一次评测写入数据
      2. 启动 Streamlit
      3. curl 页面检查是否显示结果
    Expected: 页面包含评测结果数据
    Evidence: .sisyphus/evidence/task-18-data-display.txt

  Scenario: 空状态处理
    Tool: Bash
    Steps:
      1. 清空 SQLite 或使用空数据库
      2. 启动 Streamlit
    Expected: 显示 "No evaluation results yet. Run an evaluation to get started."
    Evidence: .sisyphus/evidence/task-18-empty-state.txt
  ```

  **Commit**: YES (groups with 19)
  - Message: `feat(benchmark): Streamlit visualization interface`

---

- [ ] 21. 错误处理 + 日志系统

  **What to do**:
  - 创建 `benchmark/core/logging.py` — 日志配置
  - 日志配置：
    - 文件日志：`benchmark/logs/benchmark.log`
    - 控制台日志：rich 格式化输出
    - 日志级别：INFO（默认）、DEBUG（--verbose）
  - 全局异常处理：
    - API 调用失败 → 重试 3 次 → 记录错误 → 标记题目为 failed
    - 代码执行超时 → 记录超时信息 → 标记为 failed
    - 数据集加载失败 → 明确错误信息
    - SQLite 写入失败 → 重试 + 错误日志
  - 创建 `benchmark/logs/` 目录（.gitignore 排除日志文件）
  - 在 `.gitignore` 中添加 `benchmark/logs/` 和 `benchmark/data/*.db`

  **Must NOT do**:
  - 不要使用复杂的日志框架（用 logging 标准库）
  - 不要实现告警/通知机制

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 标准日志配置 + 异常处理
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 20)
  - **Blocks**: Final Verification
  - **Blocked By**: Task 14

  **References**:
  - `benchmark/cli.py` — CLI 入口，需添加全局异常处理
  - `benchmark/core/llm_adapter.py` — API 调用异常处理

  **Acceptance Criteria**:
  - [ ] 日志文件在 `benchmark/logs/benchmark.log`
  - [ ] API 失败时自动重试 3 次
  - [ ] 所有异常被捕获并记录日志
  - [ ] `--verbose` 参数启用 DEBUG 日志

  **QA Scenarios**:
  ```
  Scenario: API 失败重试
    Tool: Bash
    Steps:
      1. 使用无效 API key 运行评测
      2. 检查日志文件
    Expected: 日志包含 3 次重试记录和最终错误信息
    Evidence: .sisyphus/evidence/task-19-api-retry-log.txt

  Scenario: 无效维度优雅降级
    Tool: Bash
    Steps:
      1. python -m benchmark evaluate --dimension invalid --model glm-4.7
    Expected: 输出明确错误信息，不抛出未捕获异常
    Evidence: .sisyphus/evidence/task-19-invalid-dimension.txt

  Scenario: 日志文件验证
    Tool: Bash
    Steps:
      1. 运行一次正常评测
      2. cat benchmark/logs/benchmark.log
    Expected: 日志包含开始/结束时间、每题结果、汇总统计
    Evidence: .sisyphus/evidence/task-19-log-content.txt
  ```

  **Commit**: YES (groups with 18)
  - Message: `feat(benchmark): error handling and logging`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run CLI command, check SQLite). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile` on all .py files, check for: bare except, unused imports, hardcoded paths, missing type hints, console.log-style prints. Check AI slop: excessive comments, over-abstraction, generic names. Verify pyproject.toml dependencies match imports.
  Output: `Syntax [PASS/FAIL] | Imports [N clean/N issues] | AI Slop [N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute key QA scenarios: (1) evaluate reasoning dimension, (2) evaluate backend-dev dimension, (3) launch Streamlit, (4) export results, (5) custom question eval, (6) resume eval. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1 (Tasks 1-7)**: `feat(benchmark): foundation — scaffolding, models, adapters, scorers, CLI, aggregation`
- **Wave 2 (Tasks 8-14)**: `feat(benchmark): dataset adapters and scorers`
  - 8: `feat(benchmark): SWE-bench and BigCodeBench adapters`
  - 9: `feat(benchmark): AgentBench adapter`
  - 10: `feat(benchmark): GSM8K and MATH adapters`
  - 11: `feat(benchmark): MMLU adapter`
  - 12: `feat(benchmark): FrontCode dataset and adapter`
  - 13: `feat(benchmark): execution-based scorer (functional_score)`
  - 14: `feat(benchmark): LLM-as-judge scorer (quality_score)`
- **Wave 3 (Tasks 15-19)**: `feat(benchmark): evaluation engine and utilities`
  - 15: `feat(benchmark): complete evaluation CLI with two-stage scoring`
  - 16: `feat(benchmark): agent loop for tool-use dimension`
  - 17: `feat(benchmark): evaluation resume mechanism`
  - 18: `feat(benchmark): custom question system (YAML format)`
  - 19: `feat(benchmark): result export (JSON/CSV)`
- **Wave 4 (Tasks 20-21)**: `feat(benchmark): visualization and error handling`
  - 20: `feat(benchmark): Streamlit visualization interface`
  - 21: `feat(benchmark): error handling and logging`

---

## Success Criteria

### Verification Commands
```bash
python -m benchmark evaluate --dimension reasoning --model glm-4.7 --samples 3  # Expected: 3 题评测完成，SQLite 有 3 条结果
python -m benchmark evaluate --dimension backend-dev --model glm-4.7 --samples 2  # Expected: 2 题执行验证完成
streamlit run benchmark/visualization/app.py  # Expected: Web 界面启动，端口 8501
python -m benchmark export --format json --output results.json  # Expected: 合法 JSON 文件
python -m py_compile benchmark/*.py benchmark/**/*.py  # Expected: 无语法错误
```

### Final Checklist
- [ ] 5 个维度均可独立评测
- [ ] 6 个数据集适配器工作正常
- [ ] 两阶段评分正确工作（functional_score + quality_score）
- [ ] 短路规则正确工作（functional_score=0 跳过 LLM Judge）
- [ ] 维度差异化权重正确应用
- [ ] 难度加权正确计算（easy=1.0, medium=1.5, hard=2.0）
- [ ] Agent Loop 正确工作（tool-use 维度，多轮工具调用）
- [ ] SQLite 持久化 + 评估恢复
- [ ] Streamlit Web 界面展示结果
- [ ] 自定义题目可通过 YAML 添加并评测
- [ ] 结果可导出为 JSON/CSV
- [ ] 错误处理完善（API 失败、超时、格式错误）
- [ ] 无 Docker 依赖
- [ ] 无浏览器渲染验证
- [ ] 无认证/授权
- [ ] 无 Node.js sidecar
- [ ] 所有 Must NOT Have 规则遵守
