# LLM Benchmark 评测系统设计规格

**文档版本**: 1.0  
**创建日期**: 2026-03-31  
**状态**: 已确认

---

## 概述

### 目标

构建一个 LLM 模型性能评测系统，主要用途：
- **定时自动化评测**：每日评测指定模型，追踪性能变化趋势
- **单机评测工具**：支持本地手动运行评测
- **跨模型对比**：提供多个模型之间的性能对比分析

### 核心功能

1. **多维度评测**：覆盖 5 个能力维度（frontend-dev、backend-dev、system-architecture、tool-use-agentic、reasoning）
2. **定时调度**：内置调度器，支持 cron 表达式配置定时任务
3. **趋势追踪**：可视化模型性能随时间的变化
4. **统计检验**：提供置信区间和显著性检验
5. **报告生成**：生成 PDF/HTML 格式的对比分析报告

### 使用场景

- **主要场景**：定时自动化评测，追踪大模型每日性能变化
- **辅助场景**：本地手动运行评测，快速验证某个模型
- **对比需求**：跨模型对比，生成趋势图和统计报告

---

## 架构设计

### 系统架构

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

### 核心组件

| 组件 | 职责 | 关键文件 |
|------|------|---------|
| **CLI** | 命令行入口，调度各组件 | `benchmark/cli.py` |
| **Scheduler** | 内置定时任务调度 | `benchmark/core/scheduler.py` |
| **Adapters** | 数据集加载和适配 | `benchmark/adapters/*.py` |
| **Scorers** | 两阶段评分引擎 | `benchmark/scorers/*.py` |
| **Storage** | 结果持久化 | `benchmark/models/database.py` |
| **Visualization** | Web 界面展示 | `benchmark/visualization/app.py` |
| **Reporter** | 报告生成 | `benchmark/core/reporter.py` |

---

## 关键需求（来自用户澄清）

### 1. 定时调度器

**需求**：内置调度器，配置文件驱动

**配置文件**：`benchmark/configs/schedule.yaml`
```yaml
schedules:
  - name: "daily_glm_evaluation"
    cron: "0 2 * * *"  # 每天凌晨2点
    models: ["glm-4.7", "glm-4-flash"]
    dimensions: ["all"]
    
  - name: "weekly_competition"
    cron: "0 4 * * 0"  # 每周日凌晨4点
    models: ["glm-4.7", "gpt-4", "claude-3-sonnet"]
    dimensions: ["reasoning", "backend-dev"]
```

**CLI 命令**：
```bash
python -m benchmark scheduler start  # 启动调度器
python -m benchmark scheduler status  # 查看状态
python -m benchmark scheduler stop    # 停止调度器
```

### 2. 跨模型对比能力

需要支持（用户选择了 ABCD 全部）：

**A. 基础对比表** ✅ 已在 plan 中（Streamlit 表格）

**B. 趋势图** ❌ **Plan 缺失**
- X 轴：时间（日期）
- Y 轴：分数
- 支持单模型多维度、多模型单维度对比
- 需要：时间序列数据存储 + Streamlit 图表

**C. 统计检验** ❌ **Plan 缺失**
- Bootstrap 置信区间估计
- t-test 显著性检验
- 需要：统计学计算模块

**D. 报告生成** ❌ **Plan 缺失**
- PDF/HTML 报告
- 包含趋势图、统计表格、显著性检验
- 需要：报告生成模块（Jinja2 + WeasyPrint）

### 3. 模型 API 配置

**需求**：配置文件明文存储（个人项目，安全简化）

**配置文件**：`benchmark/configs/models.yaml`
```yaml
models:
  glm-4.7:
    provider: "glm"
    api_key: "your_glm_api_key_here"  # 明文存储（个人项目）
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
```

**安全措施**：
- 配置文件加入 `.gitignore`
- README 提醒用户不要提交密钥

### 4. 数据集管理

**需求**：固定数据集，历史结果可比

**缓存策略**：
- 首次使用从 HuggingFace 下载
- 缓存到 `benchmark/datasets/`
- 后续使用直接从本地加载

### 5. 自定义题目

**需求**：补充测试官方数据集的不足

**存储**：`benchmark/datasets/custom/`
**格式**：YAML

---

## 新增依赖

**Plan 中已有的依赖**：
- `pydantic`, `rich`, `streamlit`, `datasets`, `pyyaml`

**Spec 新增依赖**（需要补充到 Plan）：
- `apscheduler` - 定时任务调度器
- `matplotlib` - 趋势图绘制（Streamlit 内嵌）
- `jinja2` - HTML 报告模板
- `weasyprint` - PDF 报告生成（可选）
- `scipy` - 统计检验（t-test, bootstrap）

---

## Plan 缺失检查

### ✅ 已在 Plan 中

1. ✅ CLI 框架（Task 6）
2. ✅ 数据集适配器（Task 8-12）
3. ✅ 评分引擎（Task 5, 13, 14）
4. ✅ SQLite 存储（Task 2）
5. ✅ Streamlit 基础界面（Task 20）
6. ✅ 评估恢复机制（Task 17）
7. ✅ 自定义题目系统（Task 18）

### ❌ Plan 缺失（需要补充）

1. ❌ **定时调度器**（`benchmark/core/scheduler.py`）
   - 读取 `schedule.yaml`
   - APScheduler cron 调度
   - PID 文件管理
   - 调度日志

2. ❌ **趋势图模块**（在 Task 20 中补充）
   - 时间序列查询
   - matplotlib 绘图
   - Streamlit 图表组件

3. ❌ **统计检验模块**（新增 Task）
   - Bootstrap 置信区间
   - t-test 显著性检验
   - 统计结果展示

4. ❌ **报告生成模块**（新增 Task）
   - Jinja2 HTML 模板
   - WeasyPrint PDF 生成
   - 报告导出 CLI

5. ❌ **models.yaml 配置**（在 Task 3 或 Task 6 中补充）
   - 多模型配置管理
   - API key 明文存储

6. ❌ **schedule.yaml 配置**（在 Task 6 中补充）
   - 定时任务定义
   - cron 表达式

---

## 数据流程图

```
定时触发（Scheduler）
    │
    ├─> 读取 schedule.yaml
    │
    ├─> 读取 models.yaml
    │
    ├─> 调用 LLM API (LLMEvalAdapter)
    │
    ├─> 评分（两阶段）
    │       ├─> ExecutionScorer (backend-dev)
    │       ├─> ExactMatchScorer (reasoning)
    │       ├─> LLMJudgeScorer (frontend, architecture)
    │       └─> AgentLoopScorer (agentic)
    │
    ├─> 保存结果（SQLite）
    │       ├─> eval_runs (运行记录)
    │       └─> eval_results (题目结果)
    │
    └─> 可视化（Streamlit）
            ├─> 基础对比表
            ├─> 趋势图（matplotlib）
            └─> 统计检验（scipy）
```

---

## 不在范围内

- ❌ Docker 容器化执行
- ❌ 浏览器渲染验证前端
- ❌ 实时多模型对比 UI
- ❌ 认证/授权
- ❌ 分布式评测
- ❌ 复杂插件架构
- ❌ 高级 Streamlit 特性
- ❌ Node.js sidecar
- ❌ 数据集自动更新
- ❌ 加密存储 API key