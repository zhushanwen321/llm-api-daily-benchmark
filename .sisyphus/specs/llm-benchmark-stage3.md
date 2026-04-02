# LLM Benchmark - Stage 3 规格文档

**版本**: 2.0
**创建日期**: 2026-03-31
**更新日期**: 2026-04-02
**状态**: 规划中
**前置条件**: Stage 2 完成并验收

---

## 概述

### 目标

1. 优化评测维度区分度 -- 更换已饱和数据集，提升题量
2. 重构 BaseScorer 架构 -- 分离评测编排与评分逻辑
3. 实现高级统计分析 -- Bootstrap 置信区间 + t-test 显著性检验
4. 实现报告生成 -- HTML/PDF 专业报告

### 为什么需要这次重构

Stage 1-2 的 4 个维度存在区分度不足的问题：
- GSM8K 对 2025-2026 前沿模型已饱和（SOTA >95%），实测 100% 通过
- MMLU 前沿模型普遍 >85%，5 题样本随机性大
- BigCodeBench Hard 有区分度但 5 题样本太小
- FrontCode 自建题目偏简单，评分标准宽松

同时 BaseScorer 接口将评测编排硬编码在 CLI 中，难以支持未来的多轮交互评测。

### 数据集调整

| 维度 | 旧方案 | 新方案 | 预期 SOTA | 预期区分度 |
|------|--------|--------|-----------|-----------|
| reasoning | GSM8K 5题 | **MATH 数据集 15题** | 50-70% | 极好 |
| backend-dev | BigCodeBench-Hard 5题 | **BigCodeBench-Hard 15题** | ~35% pass@1 | 极好 |
| system-architecture | MMLU 2学科 5题 | **MMLU-Pro 15题** | 60-75% | 好 |
| frontend-dev | FrontCode 自建 5题 | **Web-Bench 15题** | ~25% | 极好 |

### 不在范围内

- Docker 容器化部署（Stage 4）
- 定时调度器（Stage 4）
- Agent Loop / 多轮交互评测
- 浏览器渲染验证
- 分布式评测

---

## 1. BaseScorer 重构

### 1.1 核心思路

将"评测编排"（如何调用模型）与"评分逻辑"（如何判断结果）分离。

当前流程（硬编码在 `cli.py:_evaluate_task()` 中）：
```
tasks -> prompt -> LLM.generate() -> parse_response -> scorer.score()
```

重构后：
```
tasks -> Evaluator.evaluate() -> ScoringContext -> Scorer.score()
```

### 1.2 新增数据结构

**文件**: `benchmark/models/schemas.py`

```python
@dataclass
class ScoringContext:
    """统一的评分上下文"""
    model_answer: str                           # 解析后的答案
    raw_output: str                             # 模型原始输出
    expected: str                               # 期望输出
    task: TaskDefinition                        # 任务定义
    execution_trace: list[dict] | None = None   # 工具调用记录（未来扩展用）
    execution_metrics: dict | None = None       # 执行指标（未来扩展用）
```

### 1.3 新增 Evaluator 抽象

**文件**: `benchmark/core/evaluator.py`

```python
class BaseEvaluator(ABC):
    """评测编排器基类"""

    @abstractmethod
    async def evaluate(
        self,
        task: TaskDefinition,
        model: str,
        llm: LLMEvalAdapter,
    ) -> ScoringContext:
        """执行评测，返回评分上下文"""


class SingleTurnEvaluator(BaseEvaluator):
    """单轮生成：prompt -> generate -> parse"""

    async def evaluate(self, task, model, llm):
        response = await llm.agenerate(task.prompt, model=model)
        parsed = parse_response(response.content, task.dimension)
        return ScoringContext(
            model_answer=parsed.answer,
            raw_output=response.content,
            expected=task.expected_output,
            task=task,
        )
```

### 1.4 BaseScorer 接口变更

**文件**: `benchmark/scorers/base.py`

```python
class BaseScorer(ABC):
    @abstractmethod
    def score(self, ctx: ScoringContext) -> ScoreResult:
        """接收统一上下文，返回评分结果"""

    @abstractmethod
    def get_metric_name(self) -> str:
        """返回指标名称"""
```

### 1.5 现有评分器迁移

4 个现有评分器仅需改签名，逻辑完全不变：

| 评分器 | 变更 |
|--------|------|
| `ExactMatchScorer` | `score(model_output, expected, task)` -> `score(ctx)`, 内部取 `ctx.model_answer` 和 `ctx.expected` |
| `ExecutionScorer` | 同上，内部取 `ctx.model_answer` 和 `ctx.task.metadata` |
| `ChoiceMatchScorer` | 同上 |
| `KeywordMatchScorer` | 同上 |

### 1.6 CLI 注册表更新

```python
# 从 2-tuple 扩展为 3-tuple
DIMENSION_REGISTRY: dict[str, tuple[type, type, type]] = {
    "reasoning":           (MATHAdapter, MathScorer, SingleTurnEvaluator),
    "backend-dev":         (BigCodeBenchAdapter, ExecutionScorer, SingleTurnEvaluator),
    "system-architecture": (MMLUProAdapter, ChoiceMatchScorer, SingleTurnEvaluator),
    "frontend-dev":        (WebBenchAdapter, PlaywrightScorer, SingleTurnEvaluator),
}
```

### 1.7 CLI 评测流程重构

`_evaluate_task()` 核心逻辑简化为：

```python
# 旧代码（硬编码单轮流程）
gen_response = await llm.agenerate(task.prompt, model=model)
parsed = parse_response(gen_response.content, task.dimension)
score_result = scorer.score(parsed.answer, task.expected_output, task)

# 新代码（委托给 Evaluator）
ctx = await evaluator.evaluate(task, model, llm)
score_result = scorer.score(ctx)
```

### 1.8 向后兼容

- `response_parser.py` 需扩展以支持新维度名（MATH 提取 `\boxed{}`，Web-Bench 提取代码）
- `prompt_builder.py` 保持不变
- 数据库 schema 无变化
- 旧的评测结果数据完全兼容

---

## 2. 新增适配器

### 2.1 MATH 数据集适配器

**文件**: `benchmark/adapters/math_adapter.py`

**数据集**: HuggingFace `HuggingFaceM4/MATH`（或 `DigitalLearningGrowth/MATH`）

**数据格式**:
- `problem`: 数学题文本（LaTeX 格式）
- `solution`: 解题过程（LaTeX 格式）
- `type`: 题目类型（Algebra, Counting & Probability, Geometry, etc.）
- `level`: 难度级别（Level 1-5）

**选题策略**:
- 选择 Level 4-5（较难）的题目
- 覆盖多个 type（代数、几何、数论、概率等）
- 随机选 15 题（seed=42 保证可复现）

**expected_output 提取**:
- 从 solution 字段中提取 `\\boxed{...}` 内的表达式
- 数值题直接比较数值
- 代数表达式需要简化后比较

**评分器**: 新增 `MathScorer`

```python
class MathScorer(BaseScorer):
    """数学题评分器

    支持两种匹配模式:
    1. 数值比较: math.isclose (处理浮点差异)
    2. 表达式简化: sympy 简化后字符串比较
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        predicted = self._extract_boxed(ctx.model_answer)
        expected = self._extract_boxed(ctx.expected)

        # 尝试数值比较
        if self._try_numeric_match(predicted, expected):
            return ScoreResult(score=100, passed=True, ...)

        # 尝试符号比较
        if self._try_symbolic_match(predicted, expected):
            return ScoreResult(score=100, passed=True, ...)

        return ScoreResult(score=0, passed=False, ...)

    def _extract_boxed(self, text: str) -> str:
        """从 \\boxed{...} 或纯文本中提取答案"""

    def _try_numeric_match(self, a: str, b: str) -> bool:
        """数值比较"""

    def _try_symbolic_match(self, a: str, b: str) -> bool:
        """符号表达式比较（sympy）"""
```

**Prompt 构造**: Zero-shot + 要求模型输出 `\\boxed{答案}` 格式

### 2.2 MMLU-Pro 适配器

**文件**: `benchmark/adapters/mmlu_pro_adapter.py`

**数据集**: HuggingFace `TIGER-Lab/MMLU-Pro`

**数据格式**:
- `question`: 题目文本
- `options`: 10 个选项（比 MMLU 的 4 个多）
- `answer`: 正确答案索引（0-9）
- `category`: 学科分类

**与 MMLU 的关键差异**:
- 10 个选项（而非 4 个），干扰项更多
- 题目难度显著提高
- 不需要 few-shot，zero-shot 即可

**选题策略**:
- 选择 `computer_science`、`math`、`physics` 等技术相关学科
- 随机选 15 题

**评分器**: 复用 `ChoiceMatchScorer`（已支持多字母匹配）

### 2.3 Web-Bench 适配器

**文件**: `benchmark/adapters/webbench_adapter.py`

**数据集**: HuggingFace `bytedance-research/Web-Bench`

**数据格式**:
- 50 个项目 x 20 个任务 = 1000 个任务
- 每个任务包含: prompt（需求描述）+ 项目上下文（已有代码）
- 涵盖 HTML/CSS/JS + React/Vue/Angular

**选题策略**:
- 选择 3-5 个独立项目（任务间无依赖的）
- 从每个项目中选 3-5 个任务
- 总计 15 题左右

**评分器**: 新增 `PlaywrightScorer`

```python
class PlaywrightScorer(BaseScorer):
    """Web-Bench Playwright 测试评分器

    将模型生成的代码写入项目目录，
    通过 subprocess 调用 Playwright 测试脚本验证。
    """

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def score(self, ctx: ScoringContext) -> ScoreResult:
        # 1. 将模型生成的代码写入临时项目目录
        # 2. 调用 Playwright 测试脚本
        # 3. 解析测试结果
        # 4. 计算通过率
```

**执行环境**:
- 需要 Node.js + Playwright（在本地或 subprocess 中安装）
- 每个任务有独立的测试脚本
- 30-60 秒超时

---

## 3. 高级统计模块

**文件**: `benchmark/core/advanced_statistics.py`

### 3.1 Bootstrap 置信区间

```python
def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Bootstrap 重采样置信区间

    通过有放回抽样计算均值置信区间，
    对样本量小（如 15 题）的情况更稳健。
    """
```

### 3.2 显著性检验

```python
def ttest_significance(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """两模型 t-test 显著性检验

    Returns:
        {
            "p_value": float,
            "is_significant": bool,
            "effect_size": float,  # Cohen's d
            "conclusion": str,
        }
    """
```

### 3.3 模型两两比较

```python
def pairwise_comparison(
    model_scores: dict[str, list[float]],
    alpha: float = 0.05,
) -> list[dict]:
    """多模型两两 t-test 比较

    Returns:
        [
            {"model_a": "glm-5", "model_b": "MiniMax-M2.7",
             "p_value": ..., "is_significant": ..., "conclusion": ...},
            ...
        ]
    """
```

---

## 4. 报告生成

**文件**: `benchmark/core/reporter.py`

### 4.1 HTML 报告

使用 Jinja2 模板生成 HTML：

```python
def generate_html_report(
    run_ids: list[str] | None = None,
    models: list[str] | None = None,
    dimensions: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    output_path: str = "report.html",
) -> str:
    """生成 HTML 报告"""
```

**报告内容**:
1. **概览**: 评测时间范围、模型列表、维度信息、题目数
2. **综合得分表**: 各模型 x 各维度的平均分 + 置信区间
3. **趋势分析**: 各维度分数随时间变化的折线图（Matplotlib 渲染为 base64 图片嵌入 HTML）
4. **统计检验**: 模型两两比较矩阵（显著差异高亮）
5. **详细结果**: 每个题目的分数明细

### 4.2 PDF 报告（可选）

```python
def generate_pdf_report(html_path: str, output_path: str) -> str:
    """从 HTML 生成 PDF（需要 weasyprint）"""
```

### 4.3 CLI 命令

```bash
python -m benchmark report \
  --models glm-5.1,MiniMax-M2.7,kimi-for-coding \
  --dimensions reasoning,backend-dev \
  --date-range 2026-04-01,2026-04-30 \
  --output report.html
```

---

## 5. 依赖更新

**新增**:
- `sympy>=1.12` -- MATH 数据集符号表达式比较
- `jinja2>=3.1` -- HTML 报告模板
- `weasyprint>=60` -- PDF 生成（可选）
- `playwright` -- Web-Bench 测试执行

---

## 6. 实施顺序

### Phase 1: 基础重构（影响全局，先做）

1. 新增 `ScoringContext` 到 `schemas.py`
2. 新增 `evaluator.py`（`BaseEvaluator` + `SingleTurnEvaluator`）
3. 重构 `BaseScorer` 接口
4. 迁移 4 个现有评分器
5. 更新 `cli.py` 使用新接口
6. 运行已有测试确认无回归

### Phase 2: 数据集替换

1. 实现 `MATHAdapter` + `MathScorer`
2. 实现 `MMLUProAdapter`（复用 `ChoiceMatchScorer`）
3. 增大 `BigCodeBenchAdapter` 的默认题量（5 -> 15）
4. 实现 `WebBenchAdapter` + `PlaywrightScorer`
5. 更新 `config.py` 中的维度配置
6. 端到端测试各维度

### Phase 3: 高级统计 + 报告

1. 实现 `advanced_statistics.py`
2. 实现 `reporter.py` + HTML 模板
3. 新增 CLI `report` 命令
4. 在 Streamlit 中集成统计图表

---

## 7. 验收标准

### 7.1 BaseScorer 重构

```bash
# 重构后，现有维度评测流程不变
python -m benchmark evaluate --model zai/glm-5.1 --dimension reasoning --samples 5
# 预期: 行为与重构前完全一致
```

### 7.2 新数据集

```bash
# MATH 数据集
python -m benchmark evaluate --model zai/glm-5.1 --dimension reasoning --samples 15
# 预期: 15 题，通过率 <100%

# MMLU-Pro
python -m benchmark evaluate --model zai/glm-5.1 --dimension system-architecture --samples 15
# 预期: 15 题，通过率 <100%

# Web-Bench
python -m benchmark evaluate --model zai/glm-5.1 --dimension frontend-dev --samples 15
# 预期: 15 题，通过率显著 <100%
```

### 7.3 高级统计

```bash
# 统计模块可独立调用
python -c "
from benchmark.core.advanced_statistics import bootstrap_confidence_interval, ttest_significance
scores_a = [80, 90, 100, 70, 60, 85, 95, 75, 80, 90]
scores_b = [60, 70, 80, 50, 40, 65, 75, 55, 60, 70]
print(bootstrap_confidence_interval(scores_a))
print(ttest_significance(scores_a, scores_b))
"
```

### 7.4 报告生成

```bash
python -m benchmark report \
  --models zai/glm-5.1,minimax/MiniMax-M2.7 \
  --dimensions reasoning,backend-dev \
  --output report.html
# 预期: 生成 HTML 文件，包含趋势图和统计表格
```

---

## 8. 状态追踪

| 组件 | 状态 | 文件 |
|------|------|------|
| ScoringContext | 待实施 | `models/schemas.py` |
| BaseEvaluator / SingleTurnEvaluator | 待实施 | `core/evaluator.py` |
| BaseScorer 重构 | 待实施 | `scorers/base.py` |
| 4 个现有评分器迁移 | 待实施 | `scorers/*.py` |
| CLI 重构 | 待实施 | `cli.py` |
| MATHAdapter + MathScorer | 待实施 | `adapters/math_adapter.py`, `scorers/math_scorer.py` |
| MMLUProAdapter | 待实施 | `adapters/mmlu_pro_adapter.py` |
| BigCodeBench 题量增大 | 待实施 | `adapters/bigcodebench_adapter.py` |
| WebBenchAdapter + PlaywrightScorer | 待实施 | `adapters/webbench_adapter.py`, `scorers/playwright_scorer.py` |
| AdvancedStatistics | 待实施 | `core/advanced_statistics.py` |
| Reporter + Templates | 待实施 | `core/reporter.py`, `templates/report.html` |
| CLI report 命令 | 待实施 | `cli.py` |
