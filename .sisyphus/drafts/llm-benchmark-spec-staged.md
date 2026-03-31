# LLM Benchmark 分阶段实施规格

**文档版本**: 1.0  
**创建日期**: 2026-03-31  
**状态**: 已确认  
**关键策略**: 每个维度只选最优难度的 5 道题目，快速迭代

---

## 📊 数据集选择策略

### 难度优先原则

**核心原则**：每个评测维度只选择最难的 5 道题目，通过少量高难度问题来区分模型能力差异。

**优势**：
1. ✅ **评测速度快**：每个维度只需评测 5 题，大幅缩短运行时间
2. ✅ **成本可控**：API 调用次数少，降低评测成本
3. ✅ **区分度高**：最难问题最能暴露模型能力差异
4. ✅ **快速迭代**：可以快速验证系统是否工作

---

## 🎯 维度与数据集映射

### 5 个评测维度

| 维度 | 数据集 | 最难问题定义 | 选择标准 | 题目数 |
|------|--------|-------------|---------|--------|
| **reasoning** | GSM8K | 步骤数最多（7-8步） | 按解答长度排序，选前5 | 5 题 |
| **backend-dev** | BigCodeBench-Hard | 官方 Hard 子集 | 随机选 5 题 | 5 题 |
| **tool-use-agentic** | AgentBench | WebShop/Mind2Web 环境 | 选环境复杂度最高的 5 题 | 5 题 |
| **system-architecture** | MMLU-Polish | 法律/道德领域 | 准确率最低的学科，选5题 | 5 题 |
| **frontend-dev** | FrontCode | 自建设计 | 设计高难度前端任务 | 5 题 |

**总计**：25 道题目（5 维度 × 5 题）

---

## 🚀 Stage 1: MVP 核心评测能力

**目标**：能跑起来，算出分数，看到结果

### 交付价值

- ✅ 手动评测单个模型
- ✅ 多维度分数展示
- ✅ 基础结果查询
- ✅ 结果持久化存储

### 包含功能

#### 1. 数据集适配器（简化版）

**reasoning 维度**：
- ✅ GSM8K 数据集（5 题，最难的）
- ✅ ExactMatchScorer（答案精确匹配）
- ✅ 从 HuggingFace 下载并缓存到本地

**backend-dev 维度**：
- ✅ BigCodeBench-Hard 子集（5 题）
- ✅ ExecutionScorer（代码执行验证）
- ✅ subprocess 沙箱执行

**不含其他维度**：
- ❌ frontend-dev（v2）
- ❌ system-architecture（v2）
- ❌ tool-use-agentic（v3）

#### 2. LLM API 调用适配器

- ✅ 从 `models.yaml` 加载模型配置
- ✅ 支持 GLM、OpenAI 兼容接口
- ✅ temperature=0（确定性评测）
- ✅ 超时和重试机制

#### 3. 评分引擎（简化版）

**两阶段评分**：
- ✅ Stage 1: functional_score（执行验证或精确匹配）
- ❌ Stage 2: quality_score（v3 实现）

**短路规则**：
- ✅ functional_score=0 直接返回 0 分

**维度权重**：
- ✅ reasoning: auto=0.8, judge=0.2（暂时只算 auto）
- ✅ backend-dev: auto=0.8, judge=0.2（暂时只算 auto）

#### 4. SQLite 存储

**Schema**：
- ✅ eval_runs（运行记录）
- ✅ eval_results（题目结果）
- ✅ 时间戳、分数、执行时间

#### 5. CLI 基础命令

```bash
# 运行评测
python -m benchmark evaluate --model glm-4.7 --dimension reasoning

# 列出数据集
python -m benchmark list-datasets

# 导出结果
python -m benchmark export --format json --output results.json
```

#### 6. Streamlit 基础界面

- ✅ 结果列表展示（表格）
- ✅ 按 model/dimension/date 过滤
- ✅ 单题详情查看（题目、输出、分数）
- ✅ 结果导出（JSON）

### 不包含

- ❌ 定时调度器
- ❌ Agent Loop（tool-use 维度）
- ❌ LLM Judge（v3）
- ❌ 趋势图
- ❌ 统计检验
- ❌ 报告生成

### 验收标准

```bash
# 1. 评测 reason# LLM Benchmark 分阶段实施规格

**文档版本**: 1.0  
**创建日期**: 2026-03-31  
**状态**: 已确认  
**关键策略**: 每个维度只选最优难度的 5 道题目，快速迭代

---

## 📊 数据集选择策略

### 难度优先原则

**核心原则**：每个评测维度只选择最难的 5 道题目，通过少量高难度问题来区分模型能力差异。

**优势**：
1. ✅ **评测速度快**：每个维度只需评测 5 题，大幅缩短运行时间
2. ✅ **成本可控**：API 调用次数少，降低评测成本
3. ✅ **区分度高**：最难问题最能暴露模型能力差异
4. ✅ **快速迭代**：可以快速验证系统是否工作

---

## 🎯 维度与数据集映射

### 5 个评测维度

| 维度 | 数据集 | 最难问题定义 | 选择标准 | 题目数 |
|------|--------|-------------|---------|--------|
| **reasoning** | GSM8K | 步骤数最多（7-8步） | 按解答长度排序，选前5 | 5 题 |
| **backend-dev** | BigCodeBench-Hard | 官方 Hard 子集 | 随机选 5 题 | 5 题 |
| **tool-use-agentic** | AgentBench | WebShop/Mind2Web 环境 | 选环境复杂度最高的 5 题 | 5 题 |
| **system-architecture** | MMLU-Polish | 法律/道德领域 | 准确率最低的学科，选5题 | 5 题 |
| **frontend-dev** | FrontCode | 自建设计 | 设计高难度前端任务 | 5 题 |

**总计**：25 道题目（5 维度 × 5 题）

---

## 🚀 Stage 1: MVP 核心评测能力

**目标**：能跑起来，算出分数，看到结果

**数据集**：2 个维度（reasoning + backend-dev），共 10 题

### 交付价值

- ✅ 手动评测单个模型
- ✅ 2 维度分数展示
- ✅ 基础结果查询
- ✅ 结果持久化存储

### 包含功能

#### 1. 数据集适配器（简化版）

**reasoning 维度**：
- ✅ GSM8K 最难的 5 题
- ✅ ExactMatchScorer（答案精确匹配）
- ✅ 从 HuggingFace 下载并缓存

**backend-dev 维度**：
- ✅ BigCodeBench-Hard 5 题
- ✅ ExecutionScorer（代码执行验证）
- ✅ subprocess 沙箱执行

#### 2. LLM API 调用适配器

- ✅ 从 `models.yaml` 加载配置
- ✅ 支持 GLM、OpenAI 兼容接口
- ✅ temperature=0（确定性评测）
- ✅ 超时和重试机制

#### 3. 评分引擎（简化版）

- ✅ functional_score（执行验证或精确匹配）
- ✅ 短路规则（score=0 直接返回）
- ❌ quality_score（v3 实现）

#### 4. SQLite 存储

- ✅ eval_runs 表
- ✅ eval_results 表

#### 5. CLI 基础命令

```bash
python -m benchmark evaluate --model glm-4.7 --dimension reasoning
python -m benchmark list-datasets
python -m benchmark export --format json
```

#### 6. Streamlit 基础界面

- ✅ 结果列表展示
- ✅ 按 model/dimension/date 过滤
- ✅ 单题详情查看

### 验收标准

```bash
# 1. 评测 reasoning 维度（5 题）
python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 5

# 预期输出
# Loading GSM8K (5 tasks)...
# [1/5] Task ID: xxx | Score: 85
# [2/5] Task ID: yyy | Score: 0  (输出答案错误)
# ...
# Average Score: 75.6

# 2. 查看 Streamlit
streamlit run benchmark/visualization/app.py
# 可以看到列表和详情

# 3. 导出结果
python -m benchmark export --format json --output results.json
# 输出合法 JSON 文件
```

### 估算工作量

| 任务 | 时间 |
|------|------|
| GSM8K 适配器（简化版） | 2-3 小时 |
| BigCodeBench-Hard 适配器 | 3-4 小时 |
| ExecutionScorer | 3-4 小时 |
| SQLite 存储 | 2-3 小时 |
| CLI 基础命令 | 2-3 小时 |
| Streamlit 基础界面 | 3-4 小时 |
| **总计** | **15-21 小时** |

---

## 🔄 Stage 2: 自动化 + 增强

**目标**：能定时跑，看趋势，有对比

**数据集**：补充 system-architecture + frontend-dev，共 4 维度 20 题

### 交付价值

- ✅ 定时自动化评测
- ✅ 追踪性能变化趋势
- ✅ 跨模型对比（2 个维度）

### 包含功能

#### 1. 定时调度器

```bash
python -m benchmark scheduler start
python -m benchmark scheduler status
python -m benchmark scheduler stop
```

- ✅ cron 配置（schedule.yaml）
- ✅ APScheduler 实现
- ✅ PID 文件管理
- ✅ 调度日志

#### 2. trends 图

- ✅ 时间序列查询
- ✅ 趋势折线图
- ✅ 单模型多维度对比
- ✅ 多模型单维度对比

#### 3. 基础统计

- ✅ 均值、标准差
- ✅ 95% 置信区间（单次计算）
- ❌ Bootstrap（v3）

#### 4. 补充数据集

**system-architecture 维度**：
- ✅ MMLU 最难的 5 题（法律/道德）
- ✅ ExactMatchScorer（选择题匹配）

**frontend-dev 维度**：
- ✅ FrontCode 自建 5 题
- ✅ LLMJudgeScorer（代码质量评估）
- ⚠️ **注意**：这是 v1 中唯一使用 LLM Judge 的维度

#### 5. 多模型配置管理

```yaml
# configs/models.yaml
models:
  glm-4.7:
    provider: "glm"
    api_key: "xxx"
    
  gpt-4:
    provider: "openai"
    api_key: "xxx"
```

### 验收标准

```bash
# 1. 启动定时任务
python -m benchmark scheduler start
# 预期：创建 PID 文件，开始 cron 调度

# 2. 查看 Schedule 配置
cat configs/schedule.yaml
# 预期：包含定时任务定义

# 3. 查看趋势图
streamlit run benchmark/visualization/app.py
# 预期：可以看到分数随时间变化的折线图

# 4. 查看统计结果
# 预期：可以看到均值、标准差、置信区间
```

### 估算工作量

| 任务 | 时间 |
|------|------|
| 定时调度器 | 3-4 小时 |
| Trends 图 | 3-4 小时 |
| 基础统计 | 2-3 小时 |
| MMLU 适配器 | 2-3 小时 |
| FrontCode 数据集 | 3-4 小时 |
| LLMJudgeScorer | 4-5 小时 |
| **总计** | **17-23 小时** |

---

## 🎓 Stage 3: 高级 + 完善

**目标**：完整对比能力，深度分析

**数据集**：补充 tool-use-agentic，共 5 维度 25 题

### 交付价值

- ✅ 完整的 5 维度评测
- ✅ 显著性检验
- ✅ 专业报告（PDF/HTML）

### 包含功能

#### 1. Agent Loop 实现

- ✅ AgentBench 最难的 5 题（WebShop/Mind2Web）
- ✅ 多轮工具调用
- ✅ subprocess 沙箱执行
- ✅ 工具模拟器

#### 2. 高级统计

- ✅ Bootstrap 置信区间
- ✅ t-test 显著性检验
- ✅ 模型对比结论

#### 3. 报告生成

```bash
python -m benchmark report \
  --models glm-4.7,gpt-4 \
  --dimensions reasoning,backend-dev \
  --date-range 2024-01-01,2024-01-31 \
  --output report.pdf
```

- ✅ Jinja2 HTML 模板
- ✅ WeasyPrint PDF 生成
- ✅ 包含趋势图、统计表格、显著性检验

#### 4. SWE-bench 补充（可选）

- ✅ SWE-bench Verified 最难的 5 题
- ✅ 替换或补充 BigCodeBench

### 验收标准

```bash
# 1. Agent Loop 评测
python -m benchmark evaluate --model glm-4.7 --dimension tool-use-agentic
# 预期：可以看到多轮工具调用记录

# 2. 生成对比报告
python -m benchmark report --models glm-4.7,gpt-4 --output report.pdf
# 预期：生成 PDF 报告

# 3. 查看显著性检验
streamlit run benchmark/visualization/app.py
# 预期：可以看到"模型 A 是否显著优于模型 B"

# 4. Bootstrap 分析
# 预期：可以看到置信区间
```

### 估算工作量

| 任务 | 时间 |
|------|------|
| Agent Loop 实现 | 5-6 小时 |
| AgentBench 适配器 | 3-4 小时 |
| Bootstrap 统计 | 3-4 小时 |
| t-test 显著性检验 | 2-3 小时 |
| 报告生成模块 | 4-5 小时 |
| SWE-bench 适配器（可选） | 3-4 小时 |
| **总计** | **17-26 小时** |

---

## 📊 三阶段对比总结

| 阶段 | 数据集 | 题目数 | 核心功能 | 使用场景 | 工作量 |
|------|--------|--------|---------|---------|--------|
| **Stage 1** | reasoning + backend-dev | 10 题 | 手动评测 + 基础界面 | 快速验证系统是否工作 | 15-21 小时 |
| **Stage 2** | + system-architecture + frontend-dev | 20 题 | 定时调度 + 趋势图 | 日常自动化评测 | 17-23 小时 |
| **Stage 3** | + tool-use-agentic | 25 题 | Agent Loop + 报告 | 专业对比分析 | 17-26 小时 |
| **总计** | 5 维度 | 25 题 | 完整功能 | 全场景覆盖 | **49-70 小时** |

---

## 🗂️ 实施顺序建议

**阶段交付节奏**：

1. **Week 1-2**: 完成 Stage 1
   - 验证核心架构是否正确
   - 确认数据集加载和评分逻辑
   - 快速反馈，调整设计

2. **Week 3-4**: 完成 Stage 2
   - 添加定时调度能力
   - 验证多模型对比
   - 用户开始使用

3. **Week 5-6**: 完成 Stage 3
   - 补充高级功能
   - 生成对比报告
   - 项目交付

---

## ❓ FAQ

**Q1: 每个 VP 阶段是否独立可用？**

✅ 是的。每个 VP 都可以独立使用：
- Stage 1：手动评测，查看结果
- Stage 2：自动评测，查看趋势
- Stage 3：深度分析，生成报告

**Q2: 如果时间不够，能否跳过某些阶段？**

✅ 可以。核心是 Stage 1，后续阶段是增强功能。
- 最小可用版本：Stage 1
- 推荐版本：Stage 1 + Stage 2
- 完整版本：Stage 1 + Stage 2 + Stage 3

**Q3: 数据集能否扩展？**

✅ 可以。每个阶段预留了扩展点：
- Stage 1：2 个维度
- Stage 2：+2 个维度
- Stage 3：+1 个维度

**Q4: 为什么只选 5 道题？**

基于以下考虑：
1. **成本控制**：25 题的 API 调用成本可控
2. **评测时间**：25 题在 30 分钟内完成
3. **区分度**：最难问题最能暴露模型差异
4. **快速迭代**：少量题目便于快速验证系统

如果需要更多题目，可以在系统中配置 `--samples` 参数。

---

## 📝 关键决策记录

### 决策 1: 数据集难度优先

**背景**：用户要求每个维度只选最难的 5 题

**决策**：采用官方难度分级或社区共识
- GSM8K：7-8 步题目
- BigCodeBench：Hard 子集
- AgentBench：WebShop/Mind2Web
- MMLU：法律/道德
- FrontCode：自建设计

**理由**：最难问题最能区分模型能力差异

### 决策 2: 阶段交付

**背景**：项目较大，需要拆分

**决策**：按使用场景拆分为 3 个阶段
- Stage 1：核心评测能力
- Stage 2：自动化 + 增强
- Stage 3：高级 + 完善

**理由**：每个阶段都能独立使用，降低风险

### 决策 3: 简化评分引擎

**背景**：Stage 1 只需功能性评分

**决策**：Stage 1 只实现 functional_score
- reasoning：ExactMatchScorer
- backend-dev：ExecutionScorer
- frontend-dev：LLMJudgeScorer（Stage 2）

**理由**：LLM Judge 需要额外设计和调试，放在 Stage 2

---

## ✅ 下一步

1. **用户确认**：确认阶段拆分是否合理
2. **开始实施**：从 Stage 1 开始
3. **快速迭代**：每个阶段独立验证