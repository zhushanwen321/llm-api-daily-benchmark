# LLM API 稳定性监控设计文档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 检测 LLM 供应商是否在不同时间偷偷更换模型、使用量化版本、或提供不一致的服务质量。

**Architecture:** 在现有 benchmark 流程中叠加质量信号采集层和稳定性分析层。Phase 1 在每日 benchmark 中嵌入信号采集 + 自动分析；Phase 2 增加高频 Probe 系统和模型指纹库。

**Tech Stack:** Python 3.13, SQLite, asyncio, httpx（与现有项目一致）

---

## 背景

用户使用多家供应商的 Coding Plan（如 zai/glm-4.7），观察到以下不一致现象：

- **上午**：性能和质量最高
- **下午**：token 速度降至 20 t/s 以下
- **晚上**：思考不严谨、遗漏、重复输出浪费 token、输出乱码、不按格式输出

核心诉求：通过 benchmark + 统计检验，判断不同时间大模型的表现是否一致，甚至是否为同一模型，或是否为量化模型。

---

## Phase 1：质量信号采集 + 稳定性分析

### 1. QualitySignalCollector

**位置**：`benchmark/analysis/quality_signals.py`

**调用时机**：在 `_evaluate_task` 的评分阶段之后、DB 写入之前调用。

**13 个质量信号**：

#### 零成本信号（从已有输出提取）

| # | 信号 | 字段名 | 类型 | 检测逻辑 |
|---|---|---|---|---|
| 1 | 格式遵从 | `format_compliance` | float 0-1 | 按维度检测：reasoning 检测 `\boxed{}`，backend-dev/system-architecture 检测 JSON `{"answer":...}` 或 `{"code":...}`，frontend-dev 检测 JSON `{"code":...}`。有 = 1.0，无 = 0.0 |
| 2 | 重复率 | `repetition_ratio` | float 0-1 | 提取所有 trigram，`重复 trigram 数 / 总 trigram 数`。> 0.3 为严重重复 |
| 3 | 乱码率 | `garbled_text_ratio` | float 0-1 | 统计不可打印 ASCII 字符（排除换行/制表符）和 Unicode 私用区字符占比 |
| 4 | 拒绝检测 | `refusal_detected` | int 0/1 | 正则匹配："作为.*AI"、"I cannot"、"I'm unable"、"抱歉.*无法"、"Sorry.*can't" |
| 5 | 语言一致性 | `language_consistency` | float 0-1 | 检测输出中是否混入其他语言的片段（如中文输出中突然出现大段英文或日文）。用 Unicode 范围分段统计，主语言占比 |
| 6 | 输出长度异常 | `output_length_zscore` | float | `len(raw_output) - hist_mean) / hist_std`。无历史时为 0.0 |
| 7 | 思考比例 | `thinking_ratio` | float 0-1 | `reasoning_tokens / completion_tokens`（从 gen_metrics 取）。无 reasoning 时为 0.0 |
| 8 | 空思考检测 | `empty_reasoning` | int 0/1 | 配置了 thinking 的模型但 reasoning_content 为空 |
| 9 | 截断率 | `truncated` | int 0/1 | finish_reason = "length" |
| 10 | Token 效率 | `token_efficiency_zscore` | float | 同一 prompt 的 prompt_tokens 与历史均值的 z-score。tokenizer 变化 = 模型更换 |

#### 低成本信号（从 API 指标时序分析）

| # | 信号 | 字段名 | 类型 | 检测逻辑 |
|---|---|---|---|---|
| 11 | TPS 异常 | `tps_zscore` | float | tokens_per_second 与历史基线的 z-score |
| 12 | TTFT 异常 | `ttft_zscore` | float | ttft_content 与历史基线的 z-score |
| 13 | 答案分布熵 | `answer_entropy` | float | 同批次不同题目的 passed 分布：`-p_pass*log(p_pass) - p_fail*log(p_fail)` |

### 2. StabilityAnalyzer

**位置**：`benchmark/analysis/stability_analyzer.py`

**调用时机**：一个完整 run 的所有 task 完成后调用。

#### 分析流程

```
StabilityAnalyzer.run(model, run_id)
  │
  ├─ 1. 加载当前 run 的 quality_signals + eval_results
  ├─ 2. 加载历史基线（最近 N 天，默认 7 天）
  ├─ 3. 逐信号计算 z-score 异常（|z| > 2 标记）
  ├─ 4. CUSUM 变化点检测（TPS, TTFT, score, thinking_ratio）
  ├─ 5. Welch's t-test（当前 run vs 历史，对 score + 每个信号）
  │     - Bonferroni 校正：α = 0.05 / 13 ≈ 0.004
  └─ 6. 判定 overall_status + 生成 StabilityReport
```

#### CUSUM 变化点检测

对每个关键信号（TPS, TTFT, score, thinking_ratio）：
1. 计算历史均值 μ 和标准差 σ
2. CUSUM 上界：`S_high = max(0, S_high_prev + (x_i - μ - k))` 其中 k = 0.5σ
3. CUSUM 下界：`S_low = min(0, S_low_prev + (x_i - μ + k))`
4. 当 S_high > h 或 S_low < -h（h = 5σ）时标记变化点
5. 至少需要 5 个历史数据点才启用 CUSUM

#### Overall Status 判定规则

| 状态 | 条件 |
|---|---|
| **stable** | 无异常信号、无变化点、t-test 无显著差异 |
| **suspicious** | TPS 或 TTFT 有变化点 OR thinking_ratio 显著变化，但分数未显著下降 |
| **degraded** | 分数显著下降（p < 0.004）OR 3+ 信号异常 OR format_compliance/repetition_ratio 显著恶化 |

#### 数据模型

```python
@dataclass
class AnomalyDetail:
    signal_name: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float

@dataclass
class ChangePoint:
    signal_name: str
    detected_at: datetime
    direction: "increase" | "decrease"
    magnitude: float  # 偏离基线的倍数

@dataclass
class StabilityReport:
    model: str
    run_id: str
    overall_status: str  # stable / degraded / suspicious
    anomalies: list[AnomalyDetail]
    change_points: list[ChangePoint]
    stat_tests: list[dict]  # {signal, p_value, effect_size, significant}
    summary: str
    created_at: datetime
```

### 3. 数据库变更

**新增表 `quality_signals`**：

```sql
CREATE TABLE IF NOT EXISTS quality_signals (
    signal_id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES eval_results(result_id),
    format_compliance REAL DEFAULT 0,
    repetition_ratio REAL DEFAULT 0,
    garbled_text_ratio REAL DEFAULT 0,
    refusal_detected INTEGER DEFAULT 0,
    language_consistency REAL DEFAULT 1.0,
    output_length_zscore REAL DEFAULT 0,
    thinking_ratio REAL DEFAULT 0,
    empty_reasoning INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    token_efficiency_zscore REAL DEFAULT 0,
    tps_zscore REAL DEFAULT 0,
    ttft_zscore REAL DEFAULT 0,
    answer_entropy REAL DEFAULT 0,
    raw_output_length INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_qs_result_id ON quality_signals(result_id);
CREATE INDEX IF NOT EXISTS idx_qs_created_at ON quality_signals(created_at);
```

**新增表 `stability_reports`**：

```sql
CREATE TABLE IF NOT EXISTS stability_reports (
    report_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    run_id TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    anomalies TEXT NOT NULL DEFAULT '[]',
    change_points TEXT NOT NULL DEFAULT '[]',
    stat_tests TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sr_model ON stability_reports(model);
CREATE INDEX IF NOT EXISTS idx_sr_run_id ON stability_reports(run_id);
```

### 4. 集成方式

**修改 `_evaluate_task`（`benchmark/cli.py`）**：

在 DB 写入阶段（`await db.asave_result(result)` 和 `await db.asave_metrics(...)` 之后）新增：

```python
# 质量信号采集（零成本，解析已有输出）
from benchmark.analysis.quality_signals import QualitySignalCollector
signal_collector = QualitySignalCollector(db=db, model=model)
await signal_collector.collect_and_save(
    result_id=result_id,
    raw_output=ctx.raw_output,
    reasoning_content=ctx.reasoning_content,
    gen_metrics=gm,
    finish_reason=ctx.raw_output,  # 从 GenerateResponse 取
    task=task,
    dimension=dimension,
)
```

**修改 `_run_evaluation`（`benchmark/cli.py`）**：

在 run 结束（`db.finish_run` 之前）新增：

```python
# 稳定性分析
from benchmark.analysis.stability_analyzer import StabilityAnalyzer
analyzer = StabilityAnalyzer(db=db)
report = analyzer.run(model=model, run_id=run_id, dimension=dimension)
console.print(f"  Stability: [{report.overall_status}] {report.summary}")
```

### 5. 新增文件结构

```
benchmark/
  analysis/
    __init__.py
    quality_signals.py       ← QualitySignalCollector
    stability_analyzer.py    ← StabilityAnalyzer
    models.py                ← StabilityReport, AnomalyDetail, ChangePoint
```

---

## Phase 2：高频 Probe 系统 + 模型指纹库

### 1. Probe 数据集

**位置**：`benchmark/datasets/probe/tasks.json`

20 题，3-5 分钟跑完，针对量化模型弱点：

| 题目类型 | 数量 | 设计意图 |
|---|---|---|
| 格式遵从题 | 5 | 要求严格 JSON 输出，量化模型格式出错率高 |
| 复杂推理题 | 5 | 多步数学推理，量化模型精度下降明显 |
| 长上下文一致性 | 3 | 上下文前后矛盾的检测 |
| 指令遵从题 | 4 | "只用英文回答"、"不超过 100 词" 等，量化模型容易忽视 |
| 已知答案题 | 3 | 固定答案的题目，用于 fingerprint_hash |

### 2. Probe 适配器

**位置**：`benchmark/adapters/probe_adapter.py`

实现 `DatasetAdapter` 接口，加载 `probe/tasks.json`。每题的 metadata 包含：
- `expected_format`: "json" / "boxed" / "text"
- `expected_answer`: 已知答案题的正确答案
- `instruction_constraints`: 指令遵从约束列表
- `consistency_group`: 用于一致性方差检测的分组 ID

### 3. 指纹库

每次 probe 运行后，将 20 题的答案向量哈希后存入：

```
fingerprint_db/
  {model}/
    baseline.json      ← 首次运行的基线指纹
    {timestamp}.json   ← 每次运行
```

指纹向量包含：
- 20 个题目的分数（0 or 100）
- 13 个聚合质量信号
- 总共 33 维向量

对比时计算余弦相似度，低于阈值（默认 0.85）标记为 "suspected model change"。

### 4. 高频调度

新增 CLI 命令：
- `benchmark-cli probe run --model glm/glm-4.7` — 单次 probe 运行
- `benchmark-cli probe schedule --interval 2h` — 自动定期运行

Probe 结果写入相同的 `quality_signals` 表，复用 StabilityAnalyzer 分析。

### 5. 新增文件结构

```
benchmark/
  adapters/
    probe_adapter.py          ← Probe 适配器
  datasets/
    probe/
      tasks.json              ← 20 题 Probe 数据集
  analysis/
    fingerprint.py            ← 指纹生成与对比
scripts/
  generate_probe_tasks.py     ← Probe 题目生成脚本（类似 frontcode）
```

---

## Phase 3：模型身份识别（长期）

积累足够指纹数据后：
1. 用聚类算法（DBSCAN）对不同时间点的指纹做自动聚类
2. 同一模型的不同聚类 = 确认发生了模型替换
3. 用分类模型（SVM / KNN）识别"这是哪个已知模型的输出"

此阶段需要 Phase 1 + Phase 2 运行数周积累数据后再设计，当前不做详细设计。

---

## 实施优先级

| 阶段 | 内容 | 预期收益 |
|---|---|---|
| Phase 1a | QualitySignalCollector + DB schema | 每日自动采集 13 个质量信号 |
| Phase 1b | StabilityAnalyzer + 报告输出 | 自动检测稳定性异常 |
| Phase 2a | Probe 数据集设计 + 适配器 | 高频监控能力 |
| Phase 2b | 指纹库 + 高频调度 | 模型身份检测 |
| Phase 3 | 聚类/分类 | 自动识别模型 |
