# LLM Benchmark 评测系统 - 总览规格

**文档版本**: 1.0  
**创建日期**: 2026-03-31  
**状态**: 已确认  
**实施策略**: 分 3 个阶段交付，每个阶段独立可用

---

## 概述

### 项目目标

构建一个 LLM 模型性能评测系统，用于：
- **定时自动化评测**：每日评测指定模型，追踪性能变化趋势
- **单机评测工具**：支持本地手动运行评测
- **跨模型对比**：提供多个模型之间的性能对比分析

### 核心策略

**难度优先**：每个评测维度只选择最难的 5 道题目

**优势**：
1. ✅ 评测速度快（25题总计，约30分钟）
2. ✅ 成本可控（API调用少）
3. ✅ 区分度高（最难问题最暴露差异）
4. ✅ 快速迭代（快速验证系统）

---

## 评测维度与数据集

### 5 个评测维度

| 维度 | 数据集 | 最难问题定义 | 题目数 |
|------|--------|-------------|--------|
| **reasoning** | GSM8K | 步骤数最多（7-8步） | 5题 |
| **backend-dev** | BigCodeBench-Hard | 官方Hard子集 | 5题 |
| **tool-use-agentic** | AgentBench | WebShop/Mind2Web环境 | 5题 |
| **system-architecture** | MMLU | 法律/道德领域 | 5题 |
| **frontend-dev** | FrontCode | 自建设计 | 5题 |

**总计**：25道题目（5维度 × 5题）

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI (benchmark/cli.py)                    │
│  evaluate | scheduler | list-datasets | export | report    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│   Scheduler  │  │   Scorer    │  │  Reporter   │
│  (定时调度)   │  │  (评分引擎) │  │  (报告生成) │
└───────┬──────┘  └──────┬──────┘  └──────┬──────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│    Adapters  │  │   Storage   │  │ Visualization│
│  (数据集适配) │  │  (SQLite)   │  │  (Streamlit) │
└──────────────┘  └─────────────┘  └──────────────┘
```

---

## 分阶段实施

### Stage 1: MVP 核心评测能力（15-21小时）

**数据集**：reasoning + backend-dev，共10题

**功能**：
- 手动评测单个模型
- 执行验证 + 精确匹配评分
- SQLite 存储
- Streamlit 基础界面

**详细规格**：见 `llm-benchmark-stage1.md`

---

### Stage 2: 自动化 + 增强（17-23小时）

**数据集**：+ system-architecture + frontend-dev，共20题

**功能**：
- 定时调度器
- 趋势图
- 基础统计
- LLM Judge评分器（仅frontend-dev）

**详细规格**：见 `llm-benchmark-stage2.md`

---

### Stage 3: 高级 + 完善（17-26小时）

**数据集**：+ tool-use-agentic，共25题

**功能**：
- Agent Loop实现
- 高级统计（Bootstrap + t-test）
- 报告生成（PDF/HTML）

**详细规格**：见 `llm-benchmark-stage3.md`

---

## 使用场景

### Scene 1：手动评测单个模型

```bash
# Stage 1 可用
python -m benchmark evaluate --model glm-4.7 --dimension reasoning
streamlit run benchmark/visualization/app.py
```

### Scene 2：定时自动化评测

```bash
# Stage 2 可用
python -m benchmark scheduler start
# 每天2点自动评测
streamlit run benchmark/visualization/app.py
# 查看趋势图
```

### Scene 3：深度对比分析

```bash
# Stage 3 可用
python -m benchmark report \
  --models glm-4.7,gpt-4 \
  --date-range 2024-01-01,2024-01-31 \
  --output report.pdf
# 生成PDF报告
```

---

## 技术栈

### 核心依赖

- `pydantic`: 数据验证
- `rich`: CLI进度显示
- `streamlit`: Web可视化
- `datasets`: HuggingFace数据集加载
- `pyyaml`: 配置文件解析
- `sqlite3`: 结果存储

### 新增依赖（Stage 2/3）

- `apscheduler`: 定时任务调度
- `matplotlib`: 趋势图绑制
- `jinja2`: HTML报告模板
- `scipy`: 统计检验
- `weasyprint`: PDF生成（可选）

---

## 成功标准

### Stage 1验收标准

- ✅ `python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 5` 成功运行
- ✅ SQLite中有10条评测结果
- ✅ Streamlit显示结果列表和详情

### Stage 2 验收标准

- ✅ `python -m benchmark scheduler start` 启动定时任务
- ✅ Streamlit显示趋势图
- ✅ 可以看到均值、标准差、置信区间

### Stage 3 验收标准

- ✅ `python -m benchmark evaluate --model glm-4.7 --dimension tool-use-agentic` 成功运行
- ✅ `python -m benchmark report --models glm-4.7,gpt-4 --output report.pdf` 生成PDF
- ✅ Streamlit显示显著性检验结果

---

## 项目目录结构

```
llm-api-daily-benchmark/
├── benchmark/
│   ├── __init__.py
│   ├── __main__.py              # CLI入口
│   ├── cli.py                   # CLI命令实现
│   ├── config.py                # 配置管理
│   │
│   ├── core/                    # 核心逻辑
│   │   ├── llm_adapter.py       # LLM API调用适配
│   │   ├── scheduler.py         # 定时调度器
│   │   ├── reporter.py          # 报告生成
│   │   └── scanner.py           # 结果扫描
│   │
│   ├── adapters/               # 数据集适配器
│   │   ├── base.py              # 基类
│   │   ├── gsm8k_adapter.py     # GSM8K
│   │   ├── bigcodebench_adapter.py  # BigCodeBench
│   │   ├── agentbench_adapter.py    # AgentBench
│   │   ├── mmlu_adapter.py      # MMLU
│   │   └── frontcode_adapter.py # 自建前端题
│   │
│   ├── scorers/                # 评分引擎
│   │   ├── base.py             # 基类
│   │   ├── execution_scorer.py  # 执行验证
│   │   ├── exact_match_scorer.py # 精确匹配
│   │   ├── llm_judge_scorer.py # LLM-as-judge
│   │   └── agent_loop.py        # Agent Loop
│   │
│   ├── models/                 # 数据模型
│   │   ├── schemas.py           # Pydantic模型
│   │   └── database.py          # SQLite操作
│   │
│   ├── visualization/           # Web界面
│   │   ├── app.py
│   │   └── components/          # Streamlit组件
│   │
│   ├── datasets/               # 数据集存储
│   │   ├── gsm8k/
│   │   ├── bigcodebench/
│   │   ├── agentbench/
│   │   ├── mmlu/
│   │   └── frontcode/
│   │
│   └── configs/                # 配置文件
│       ├── default.yaml
│       ├── models.yaml
│       └── schedule.yaml
│
├── pyproject.toml               # 项目依赖
└── README.md
```

---

## 下一步

1. ✅ 总览规格已确认
2. ✅ Stage 1-3 规格已文档化
3. 🔄 开始创建 Stage 1 实施计划

---

## 文档链接

- **Stage 1 详细规格**：`llm-benchmark-stage1.md`
- **Stage 2 概要规格**：`llm-benchmark-stage2.md`
- **Stage 3 概要规格**：`llm-benchmark-stage3.md`