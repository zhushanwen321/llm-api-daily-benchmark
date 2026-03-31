# Plan 反思报告：对比 Spec 发现的缺失

**审查时间**: 2026-03-31  
**状态**: 发现重大缺失，需要补充

---

## 📊 总体评估

| 指标 | 状态 | 说明 |
|------|------|------|
| **功能完整性** | ⚠️ 缺失 | 缺少定时调度、趋势图、统计检验、报告生成 |
| **任务覆盖** | ⚠️ 不完整 | 缺少 4 个新增任务 |
| **依赖完整性** | ⚠️ 缺失 | 缺少 5 个新增依赖 |
| **配置文件** | ⚠️ 不完整 | 缺少 schedule.yaml，models.yaml 说明不足 |
| **验收标准** | ✅ 清晰 | 已有任务的验收标准完整 |

---

## ❌ 发现的问题

### 1. 缺少定时调度器任务

**Spec 要求**：
- 内置调度器，支持 cron 表达式
- 配置文件驱动（`schedule.yaml`）
- CLI 命令：`scheduler start/stop/status`

**Plan 现状**：
- ❌ 完全缺失定时调度器任务
- ❌ 没有 `benchmark/core/scheduler.py` 的实现任务

**影响**：
- 无法满足"定时自动化评测"的核心需求
- 用户需要手动运行评测，无法定时执行

**需要新增 Task**：
- **Task 22: 定时调度器实现**

---

### 2. 缺少趋势图和统计检验

**Spec 要求**：
- **趋势图**：X 轴时间，Y 轴分数，支持单模型多维度/多模型单维度
- **统计检验**：Bootstrap 置信区间，t-test 显著性检验

**Plan 现状**：
- Task 20: Streamlit 主界面
  - ✅ 基础对比表
  - ❌ 没有趋势图
  - ❌ 没有统计检验
  - ❌ 没有时间序列查询

**影响**：
- 无法满足"追踪大模型每日性能变化"的需求
- 无法进行跨模型统计对比

**需要修改 Task 20**：
- 补充趋势图组件
- 补充统计检验模块

或

**需要新增 Task**：
- **Task 23: 趋势分析模块**（时间序列查询 + matplotlib 绘图）
- **Task 24: 统计检验模块**（Bootstrap + t-test）

---

### 3. 缺少报告生成模块

**Spec 要求**：
- 生成 PDF/HTML 格式的对比分析报告
- 包含趋势图、统计表格、显著性检验结果
- CLI 命令：`python -m benchmark report`

**Plan 现状**：
- Task 19: 结果导出（JSON/CSV）
  - ✅ 支持导出 JSON/CSV
  - ❌ 没有报告生成功能

**影响**：
- 无法满足"生成完整对比报告"的需求（用户选择了 D）

**需要新增 Task**：
- **Task 25: 报告生成模块**
  - Jinja2 HTML 模板
  - WeasyPrint PDF 生成
  - 报告导出 CLI 命令

---

### 4. 配置文件说明不足

#### 4.1 models.yaml

**Spec 要求**：
```yaml
# benchmark/configs/models.yaml
models:
  glm-4.7:
    provider: "glm"
    api_key: "your_glm_api_key_here"  # 明文存储（个人项目）
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
```

**Plan 现状**：
- Task 6 提到了 `configs/default.yaml`
- ❌ 没有明确说明 `models.yaml` 的结构
- ❌ 没有说明多模型配置管理

**需要补充 Task 6**：
- 明确说明 `models.yaml` 的结构
- 说明如何加载和管理多模型配置

#### 4.2 schedule.yaml

**Spec 要求**：
```yaml
# benchmark/configs/schedule.yaml
schedules:
  - name: "daily_glm_evaluation"
    cron: "0 2 * * *"
    models: ["glm-4.7", "glm-4-flash"]
    dimensions: ["all"]
```

**Plan 现状**：
- ❌ 完全缺失 `schedule.yaml`
- ❌ Task 6 的配置系统没有提到定时任务配置

**需要补充 Task 6**：
- 添加 `schedule.yaml` 配置示例
- 说明定时任务配置格式

---

### 5. 缺少新增依赖

**Plan 已有依赖**：
- `pydantic`, `rich`, `streamlit`, `datasets`, `pyyaml`

**Spec 新增依赖**：
- ❌ `apscheduler` - 定时任务调度器
- ❌ `matplotlib` - 趋势图绘制
- ❌ `jinja2` - HTML 报告模板
- ❌ `weasyprint` - PDF 报告生成（可选）
- ❌ `scipy` - 统计检验

**需要补充 Task 1**：
- 更新依赖列表，添加上述库

---

## ✅ 已完成任务（Plan 正确）

1. ✅ **数据集适配器**（Task 8-12）- 6 个数据集，完整覆盖
2. ✅ **评分引擎**（Task 5, 13, 14）- 两阶段评分，短路规则
3. ✅ **数据模型**（Task 2）- SQLite 存储，Schema 清晰
4. ✅ **CLI 框架**（Task 6, 15）- 基础命令完整
5. ✅ **评估恢复**（Task 17）- 断点续传机制
6. ✅ **自定义题目**（Task 18）- YAML 格式，扩展性好
7. ✅ **错误处理**（Task 21）- API 重试，日志系统

---

## 📝 需要的修改

### 修改 Task 1: 补充依赖

**原依赖列表**：
```python
pydantic, rich, streamlit, datasets, pyyaml
```

**补充后**：
```python
pydantic, rich, streamlit, datasets, pyyaml,  # 原有
apscheduler, matplotlib, jinja2, scipy        # 新增
weasyprint  # 可选（PDF 生成）
```

---

### 修改 Task 6: 补充配置文件

**补充内容**：
- 明确说明 `models.yaml` 的结构和用途
- 补充 `schedule.yaml` 配置示例

**示例代码**：
```yaml
# configs/models.yaml
models:
  glm-4.7:
    provider: "glm"
    api_key: "your_glm_api_key_here"
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
```

```yaml
# configs/schedule.yaml
schedules:
  - name: "daily_glm_evaluation"
    cron: "0 2 * * *"
    models: ["glm-4.7", "glm-4-flash"]
    dimensions: ["all"]
```

---

### 修改 Task 3: 补充多模型配置管理

**补充内容**：
- 从 `models.yaml` 加载模型配置
- 支持多种 provider（GLM, OpenAI, Anthropic）

**示例代码**：
```python
class LLMEvalAdapter:
    def __init__(self):
        self.models_config = self._load_models_config()
    
    def _load_models_config(self):
        with open("benchmark/configs/models.yaml") as f:
            return yaml.safe_load(f)
```

---

### 新增 Task 22: 定时调度器

**文件位置**：`benchmark/core/scheduler.py`

**功能**：
- 读取 `schedule.yaml` 配置
- 使用 APScheduler 实现 cron 调度
- 启动/停止调度器
- 记录调度日志

**CLI 命令**：
```bash
python -m benchmark scheduler start
python -m benchmark scheduler status
python -m benchmark scheduler stop
```

**依赖**：Task 6（配置系统），Task 15（CLI 框架）

---

### 新增 Task 23: 趋势分析模块

**文件位置**：`benchmark/visualization/trends.py`

**功能**：
- 时间序列数据查询（从 SQLite）
- matplotlib 绘制趋势图
- Streamlit 图表组件

**需要修改**：
- Task 2: 补充时间序列查询 SQL
- Task 20: 集成趋势图组件

---

### 新增 Task 24: 统计检验模块

**文件位置**：`benchmark/core/statistics.py`

**功能**：
- Bootstrap 置信区间估计
- t-test 显著性检验
- 统计结果格式化

**依赖**：`scipy` 库

---

### 新增 Task 25: 报告生成模块

**文件位置**：`benchmark/core/reporter.py`

**功能**：
- Jinja2 HTML 模板渲染
- WeasyPrint PDF 生成
- CLI 命令：`python -m benchmark report`

**依赖**：`jinja2`, `weasyprint`

---

## 🔢 工作量估算

| 任务 | 复杂度 | 估算时间 |
|------|--------|---------|
| Task 22: 定时调度器 | medium | 2-3 小时 |
| Task 23: 趋势分析 | medium | 2-3 小时 |
| Task 24: 统计检验 | simple | 1-2 小时 |
| Task 25: 报告生成 | medium | 2-3 小时 |
| **总计** | | **7-11 小时** |

---

## 📋 建议的执行顺序

**Wave 5**（Wave 4 之后，新增）：
- Task 22: 定时调度器 [quick]
- Task 25: 报告生成模块 [medium]

**Wave 6**（Wave 5 之后，并行）：
- Task 23: 趋势分析模块 [medium]
- Task 24: 统计检验模块 [simple]

**修改已有 Task**：
- Task 1: 补充依赖
- Task 3: 补充多模型配置管理
- Task 6: 补充 schedule.yaml 配置
- Task 20: 集成趋势图和统计检验

---

## 🎯 关键问题

**Question 1**: 报告生成是必需的吗？
- Spec 要求 D（完整对比报告）
- 但 WeasyPrint 依赖较重
- **建议**：先实现 HTML 报告，PDF 作为可选功能

**Question 2**: 统计检验的实现深度？
- Bootstrap 需要多次采样，计算成本较高
- **建议**：v1 实现简化版（单次计算），v2 实现 Bootstrap

**Question 3**: 趋势图是必需的吗？
- Spec 明确要求追踪"每日性能变化"
- **建议**：必需实现，核心功能

---

## 📊 优先级建议

### P0 - 必须实现（核心需求）

1. **Task 22: 定时调度器** - 满足"定时自动化评测"核心需求
2. **Task 23: 趋势分析** - 满足"追踪性能变化"核心需求
3. **补充 models.yaml 和 schedule.yaml** - 配置文件是基础

### P1 - 应该实现（重要需求）

4. **Task 24: 统计检验** - 跨模型对比的重要功能
5. **Task 25: 报告生成** - 完整对比报告

### P2 - 可选优化（锦上添花）

6. PDF 报告生成（WeasyPrint 较重，可以先用 HTML）
7. 更复杂的统计算法（Bootstrap）

---

## ✅ 下一步建议

**建议**：
1. 先完成 spec 中缺失的新增 Task（22-25）
2. 修改已有 Task（1, 3, 6, 20）
3. 更新依赖列表
4. 更新并行执行 Wave

**是否需要我开始更新 plan？**