# LLM Benchmark - Stage 2 规格概要

**阶段**: 可视化增强 + 新维度
**版本**: 1.1
**创建日期**: 2026-03-31
**更新日期**: 2026-04-02
**状态**: 规划中
**前置条件**: Stage 1 完成并验收
**估算工作量**: 12-18 小时

---

## 概述

### 目标

扩展评测维度，增强可视化和统计分析能力。

### 交付价值

- ✅ 追踪性能变化趋势
- ✅ 跨模型对比（4个维度）
- ✅ 基础统计分析

### 数据集

| 维度 | 数据集 | 题目数 | 状态 |
|------|--------|--------|------|
| reasoning | GSM8K最难的5题 | 5题 | ✅ Stage 1 |
| backend-dev | BigCodeBench-Hard 5题 | 5题 | ✅ Stage 1 |
| **system-architecture** | **MMLU（法律/道德）** | **5题** | **🆕 Stage 2** |
| **frontend-dev** | **FrontCode自建** | **5题** | **🆕 Stage 2** |
| **总计** | | **20题** | |

---

## 新增功能

### 1. Trends 图

**文件**：`benchmark/visualization/components/trends.py`

**功能**：
- 时间序列数据查询（从SQLite）
- matplotlib 绘制趋势图
- Streamlit 组件封装

**展示内容**：
- 单模型多维度趋势（多条线）
- 多模型单维度对比（多条线）
- X轴：日期
- Y轴：分数

---

### 2. 基础统计

**文件**：`benchmark/core/statistics.py`

**功能**：
- 均值计算
- 标准差计算
- 95% 置信区间（单次计算，非Bootstrap）
- 统计结果格式化

**展示位置**：
- Streamlit结果表格中显示
- 导出JSON时包含统计信息

---

### 3. MMLU Adapter

**文件**：`benchmark/adapters/mmlu_adapter.py`

**数据集**：MMLU（5个学科）

**选择策略**：
- 选择法律、道德领域（最难的学科）
- 选择哲学、计算机科学、数学（补充分布）
- 每个学科1题，共5题

**评分器**：ExactMatchScorer（选择题匹配）

---

### 4. FrontCode Adapter

**文件**：`benchmark/adapters/frontcode_adapter.py`

**数据集**：自建设计

**题目设计**：
- HTML结构生成：1题
- CSS样式实现：1题
- JavaScript交互逻辑：1题
- React组件生成：1题
- 复杂前端任务：1题

**评分器**：LLMJudgeScorer（代码质量评估）

---

### 5. LLMJudgeScorer

**文件**：`benchmark/scorers/llm_judge_scorer.py`

**功能**：
- 调用LLM作为评判者
- 评估代码质量
- 返回0-10分（映射到0-100）

**为什么Stage 2才实现**：
- LLM Judge需要额外设计和调试
- 需要设计评分标准（rubric）
- 前端评测质量比功能更重要

---

## 验收标准

### 1. 趋势图

```bash
# 启动Streamlit
streamlit run benchmark/visualization/app.py

# 预期：
# - 可以看到"趋势"标签页
# - 可以选择模型、维度、时间范围
# - 显示分数随时间变化的折线图
```

### 2. 基础统计

```bash
# 查看Streamlit结果表格
# 预期：每列显示均值、标准差、置信区间
```

### 3. 新维度评测

```bash
# 评测system-architecture维度
python -m benchmark evaluate --model glm-4.7 --dimension system-architecture --samples 5
# 预期：MMLU评测成功，结果显示分数

# 评测frontend-dev维度
python -m benchmark evaluate --model glm-4.7 --dimension frontend-dev --samples 5
# 预期：FrontCode评测成功，LLMJudge打分
```

---

## 评分策略调整

### Stage 1 评分

- **reasoning**: functional_score（精确匹配）
- **backend-dev**: functional_score（执行验证）

### Stage 2 评分

- **reasoning**: functional_score（精确匹配）
- **backend-dev**: functional_score（执行验证）
- **system-architecture**: functional_score（选择题匹配）
- **frontend-dev**: quality_score（LLM Judge）⚠️ 这是唯一使用LLM Judge的维度

---

## 配置文件更新

### default.yaml（更新）

```yaml
# 新增维度配置
dimensions:
  reasoning:
    adapter: "gsm8k"
    auto_weight: 0.8
    judge_weight: 0.2

  backend-dev:
    adapter: "bigcodebench"
    auto_weight: 0.8
    judge_weight: 0.2

  system-architecture:  # 新增
    adapter: "mmlu"
    auto_weight: 0.8
    judge_weight: 0.2

  frontend-dev:  # 新增
    adapter: "frontcode"
    auto_weight: 0.2  # ⚠️ 注意：这里权重相反
    judge_weight: 0.8  # LLM Judge权重高
```

---

## 依赖更新

**新增依赖**：
- `matplotlib>=3.7` - 趋势图绑定
- `scipy>=1.11` - 统计计算

---

## 不在范围内

Stage 2 **不包含**：

- ❌ 定时调度器（Stage 4）
- ❌ Docker容器化（Stage 4）
- ❌ Agent Loop（Stage 3）
- ❌ Bootstrap置信区间（Stage 3）
- ❌ t-test显著性检验（Stage 3）
- ❌ PDF报告生成（Stage 3）
- ❌ tool-use-agentic维度（Stage 3）

---

## 状态追踪

| 组件 | 状态 | 说明 |
|------|------|------|
| Trends组件 | ⏳ 待实施 | 文件：`visualization/components/trends.py` |
| Statistics | ⏳ 待实施 | 文件：`core/statistics.py` |
| MMLUAdapter | ⏳ 待实施 | 文件：`adapters/mmlu_adapter.py` |
| FrontCodeAdapter | ⏳ 待实施 | 文件：`adapters/frontcode_adapter.py` |
| LLMJudgeScorer | ⏳ 待实施 | 文件：`scorers/llm_judge_scorer.py` |

---

## 下一步

⏳ 等待Stage 1完成并验收后开始实施

---

## 实施顺序建议

1. **Week 1**: 实现趋势图 + 基础统计
2. **Week 2**: 实现MMLU和FrontCode适配器 + LLM Judge

**总工作量**：12-18小时
