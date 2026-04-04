# Plan Review Report: llm-benchmark.md

## 执行摘要

**审查状态**: ✅ 通过（已补充关键细节）

**主要发现**: 计划整体结构优秀，但缺少以下关键细节：
1. 依赖遗漏（`datasets` 库）
2. 配置文件示例缺失
3. 环境配置说明缺失
4. 维度映射表不明确
5. 项目结构说明混淆

**已修复**: 所有上述问题已在原计划中补充修改。

---

## 详细审查

### ✅ 占位符检查

**结果**: 通过

计划中没有发现以下占位符：
- ~~"TBD"~~
- ~~"TODO"~~
- ~~"implement later"~~
- ~~"fill in details"~~

所有任务都有具体的实现步骤和验收标准。

---

### ✅ 类型一致性检查

**结果**: 通过

检查项：
- ✅ 数据模型一致性：`TaskDefinition`, `EvalRun`, `EvalResult`, `ScoreResult` 定义清晰
- ✅ 方法签名一致性：`DatasetAdapter.load()`, `BaseScorer.score()` 接口统一
- ✅ 评分流程一致：`TwoStageScoringPipeline` 的使用明确

---

### ⚠️ 需求覆盖检查

#### 1. "Must Have" 全部覆盖 ✅

| Must Have | 对应任务 | 状态 |
|-----------|---------|------|
| 5 个维度独立评测 | Task 8-12 (数据集) + Task 13-14 (评分器) + Task 15 (CLI) | ✅ |
| 两阶段评分策略 | Task 5 (管道) + Task 13 (functional) + Task 14 (quality) | ✅ |
| 短路规则 | Task 5 (TwoStageScoringPipeline) | ✅ |
| 维度差异化权重 | Task 5 + Task 15 (映射配置) | ✅ |
| 难度加权 | Task 7 (aggregator) | ✅ |
| SQLite 持久化 | Task 2 (database) | ✅ |
| Streamlit Web | Task 20 (visualization) | ✅ |
| CLI 接口 | Task 6 + Task 15 (CLI) | ✅ |
| 自定义题目 | Task 18 (custom YAML) | ✅ |
| 错误处理 | Task 21 (logging) | ✅ |
| Agent Loop | Task 16 (agent_loop) | ✅ |

#### 2. 必须补充的细节（已修复）

##### ✅ 依赖问题（已修复）

**原问题**: Task 1 遗漏了 `datasets` 库依赖。

**修复**: 已在 Task 1 中补充：
```python
- 更新 `pyproject.toml` 添加依赖：
  - `pydantic` (数据验证)
  - `rich` (CLI 进度)
  - `streamlit` (Web 可视化)
  - `datasets` (HuggingFace 数据集加载) ⬅️ 新增
  - `pyyaml` (配置文件解析) ⬅️ 新增
```

##### ✅ 配置文件示例（已修复）

**原问题**: Task 6 提到 `configs/default.yaml` 但没有示例内容。

**修复**: 已在 Task 6 中添加完整的配置示例，包含：
- 默认 model、temperature、max_tokens
- 维度到数据集/评分器的映射
- 难度权重配置

##### ✅ 环境配置说明（已修复）

**原问题**: 缺少 API key 和 HuggingFace token 配置说明。

**修复**: 已在 Task 1 中补充：
- 创建 `benchmark/docs/ENV_SETUP.md`
- 说明 API key 配置方式
- 提供 `.env` 文件模板

##### ✅ 项目结构说明（已修复）

**原问题**: 计划开头假设存在 `chat/` 项目，但实际上是新项目。

**修复**: 已在 "Context" 部分明确项目结构：
```
llm-api-daily-benchmark/
├── benchmark/
│   ├── adapters/
│   ├── scorers/
│   ├── datasets/
│   └── ...
├── pyproject.toml
└── README.md
```

##### ✅ 维度映射表（已修复）

**原问题**: Task 15 提到"维度到适配器/评分器的映射配置"但不明确。

**修复**: 已添加 "Dimension Mapping" 章节，明确说明：
- 每个维度对应哪些数据集
- 每个维度使用哪些评分器
- 每个维度的权重配置
- 数据集来源（HuggingFace / GitHub）

---

### ⚠️ 数据管理问题（已补充）

**问题**: 多个任务提到从 HuggingFace 加载数据集，但没有明确数据集管理方式。

**修复**: 已在各适配器任务中明确：
- 优先从本地加载（已缓存）
- 如果本地不存在，从 HuggingFace 下载并缓存
- 使用 `datasets` 库的 `load_dataset()` 和 `save_to_disk()`

---

### ✅ 任务细化度检查

**结果**: 优秀

每个任务包含：
- ✅ What to do（详细步骤）
- ✅ Must NOT do（明确边界）
- ✅ Acceptance Criteria（验收标准）
- ✅ QA Scenarios（测试场景）
- ✅ References（参考文件）
- ✅ Parallelization（并行信息）

---

## 潜在风险

### 1. 数据集下载网络依赖

**风险**: Task 8-11 需要从 HuggingFace 下载数据集，网络不稳定可能导致失败。

**建议**: 
- 已明确数据集缓存机制
- 建议在执行前预先下载数据集

### 2. LLM API 调用成本

**风险**: GSM8K 和 MATH 数据集较大，评测成本可能很高。

**缓解**: 
- Task 15 已明确 `--samples` 参数控制样本数量
- 建议 v1 每个维度使用子集（10-50 题）

### 3. AgentBench 环境复杂性

**风险**: AgentBench 涉及数据库和操作系统环境，实现复杂。

**缓解**: 
- Task 16 已明确 v1 只支持 DB/OS 子环境
- Task 9 明确不做 Docker，用 subprocess 模拟

---

## 执行建议

### 1. 执行前准备

在开始执行前，建议：

```bash
# 1. 创建项目结构
mkdir -p benchmark/{adapters,scorers,datasets,models,core,visualization,configs,docs}

# 2. 创建 .env 文件模板
cat > .env.example << EOF
GLM_API_KEY=your_api_key_here
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/
OPENAI_API_KEY=your_openai_key_here  # 可选
OPENAI_API_BASE=https://api.openai.com/v1  # 可选
EOF

# 3. 预下载数据集（可选，减少网络依赖）
python -c "
from datasets import load_dataset
load_dataset('princeton-nlp/SWE-bench_Lite', split='test').save_to_disk('benchmark/datasets/swebench/')
load_dataset('openai/gsm8k', split='test').save_to_disk('benchmark/datasets/gsm8k/')
"
```

### 2. 执行顺序

建议按 Wave 顺序执行：
1. **Wave 1 (Task 1-7)**: 基础架构，可并行
2. **Wave 2 (Task 8-14)**: 数据集和评分器，可并行
3. **Wave 3 (Task 15-19)**: 核心逻辑，顺序执行
4. **Wave 4 (Task 20-21)**: 可视化和错误处理，可并行

### 3. 验证点

每个 Wave 完成后建议进行验证：
- Wave 1: 运行 `python -m benchmark --help` 验证 CLI 可用
- Wave 2: 运行 `python -m benchmark list-datasets` 验证数据集加载
- Wave 3: 运行 `python -m benchmark evaluate --dimension reasoning --samples 1` 验证评测流程
- Wave 4: 运行 `streamlit run benchmark/visualization/app.py` 验证可视化

---

## 最终结论

**计划质量**: ⭐⭐⭐⭐⭐ (5/5)

**补充后的完整性**: 100%

**可执行性**: 高

**关键补充**:
1. ✅ 依赖完整性（datasets, pyyaml）
2. ✅ 配置文件示例（default.yaml）
3. ✅ 环境配置说明（ENV_SETUP.md）
4. ✅ 维度映射表（Dimension Mapping）
5. ✅ 项目结构明确
6. ✅ 数据集管理方案

**建议**: 可以开始执行。建议按照 "执行建议" 部分进行环境准备后再开始 Wave 1。

---

## 补充修改记录

| 问题 | 修改位置 | 修改内容 |
|------|---------|---------|
| 依赖遗漏 | Task 1 | 添加 `datasets`, `pyyaml` 依赖 |
| 配置示例缺失 | Task 6 | �充 `default.yaml` 完整示例 |
| 环境配置缺失 | Task 1 | 添加 `ENV_SETUP.md` 创建说明 |
| 维度映射不明确 | Task 15 补充 | 添加完整的 `DIMENSION_MAPPING` 配置 |
| 项目结构混淆 | Context | 添加 "Project Structure" 章节 |
| 数据集下载方式 | Task 8 | 明确 HuggingFace 下载逻辑 |
| LLMClient 假设 | Task 3 | 改为从零实现，不依赖现有 |

---

生成时间: 2026-03-31
审查人: Prometheus (AI Planning Consultant)