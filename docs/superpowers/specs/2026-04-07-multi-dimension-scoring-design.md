# 多维度评分系统设计

## 1. 概述

将 4 个评测维度的 binary 评分（0/100）升级为多维度细粒度评分，通过多个独立 Scorer 组合计算加权总分。

## 2. 关键决策

| 决策项 | 选择 |
|---|---|
| 数据存储 | 扩展 `details` JSON 字段，不改变表结构 |
| functional_score | 改为各维度加权总分 |
| 通过标准 | 统一阈值 60 分 |
| FrontCode 题目 | LLM 自动生成 15-20 题 + 测试用例 |
| 工具依赖 | 全部安装（pylint, flake8, bandit, semgrep, axe-core, stylelint, eslint, playwright 等） |
| 维度权重 | 使用分析文档建议权重 |
| 代码架构 | 多 Scorer 组合模式（CompositeScorer） |
| 报告展示 | 总分 + 雷达图详情 |
| 向后兼容 | 直接替换，不考虑兼容 |
| MATH 推理正确性 Judge | 固定 zai/glm-5.1 |
| Frontend 执行环境 | Node.js + Playwright |

## 3. 架构设计

### 3.1 Scorer 继承体系

```
BaseScorer (abstract)
├── score(ctx: ScoringContext) -> ScoreResult
├── ascore(ctx: ScoringContext) -> ScoreResult  (异步包装)
└── get_metric_name() -> str

CompositeScorer
├── scorers: list[tuple[float, BaseScorer]]  # [(weight, scorer), ...]
├── score(ctx) -> ScoreResult  # 加权总分 + 各维度明细
└── ascore(ctx) -> ScoreResult
```

### 3.2 CompositeScorer 约束

- 各子 Scorer 权重之和必须等于 1.0
- 某个子 Scorer 异常（工具未安装、执行超时）时，该维度得分默认 100 分，不拉低总分
- `reasoning_content` 为空（非 thinking model）时，依赖推理内容的 Scorer 返回默认 100 分（即不惩罚）

### 3.3 维度注册表

```
DIMENSION_REGISTRY = {
    "reasoning": (MATHAdapter, CompositeScorer([
        (0.40, AnswerCorrectnessScorer),      # 复用 MathScorer 逻辑
        (0.25, ReasoningCompletenessScorer), # reasoning_content 分析
        (0.20, ReasoningValidityScorer),    # LLM-as-a-Judge (glm-5.1)
        (0.10, MethodEleganceScorer),       # 学科关键词 + 冗余检测
        (0.05, DifficultyAdaptationScorer), # level 匹配
    ]), SingleTurnEvaluator),

    "backend-dev": (BigCodeBenchAdapter, CompositeScorer([
        (0.40, TestCoverageScorer),         # 测试通过率 (hypothesis 边界)
        (0.25, PerformanceScorer),          # timeit + 与标准答案对比
        (0.15, CodeStyleScorer),            # pylint + flake8
        (0.10, RobustnessScorer),           # AST 异常处理 + bandit
        (0.05, ArchitectureScorer),         # radon 圈复杂度
        (0.03, SecurityScorer),             # bandit + semgrep
        (0.02, ExtensibilityScorer),       # 硬编码检测
    ]), SingleTurnEvaluator),

    "system-architecture": (MMLUProAdapter, CompositeScorer([
        (0.30, AnswerCorrectnessScorer),   # 复用 ChoiceMatchScorer 逻辑
        (0.25, ReasoningCompletenessScorer),# 正则匹配
        (0.20, OptionAnalysisScorer),      # 错误选项分析检测
        (0.15, ReasoningConfidenceScorer), # 不确定性表述检测
        (0.10, SubjectAdaptationScorer),   # 学科匹配期望长度
    ]), SingleTurnEvaluator),

    "frontend-dev": (FrontCodeAdapter, CompositeScorer([
        (0.30, FunctionalityScorer),        # Node.js 执行 + Playwright
        (0.20, HTMLSemanticScorer),         # BeautifulSoup + W3C Validator
        (0.15, AccessibilityScorer),        # axe-core
        (0.15, CSSQualityScorer),          # Stylelint + AST 分析
        (0.10, CodeOrganizationScorer),     # ESLint
        (0.05, PerformanceScorer),          # Playwright + 静态分析
        (0.05, BrowserCompatScorer),       # AST 前缀分析
    ]), SingleTurnEvaluator),
}
```

### 3.4 ScoreResult 输出格式

```python
ScoreResult(
    score=82.5,           # 加权总分
    passed=True,          # score >= 60
    details={
        "composite": {
            "weights": {"correctness": 0.4, "style": 0.15, ...},
            "scores": {"correctness": 80.0, "style": 90.0, ...},
        },
        # 各子 scorer 的原始输出也保留
        "correctness": {"method": "exact_match", "predicted": "81", "expected": "81"},
        "style": {"pylint_score": 8.5, "flake8_violations": 2},
        ...
    },
    reasoning="",          # 不使用
)
```

### 3.5 数据库写入

```python
# _evaluate_task 中写入
result = EvalResult(
    functional_score=score_result.score,      # 加权总分
    final_score=score_result.score,
    passed=score_result.passed,              # score >= 60
    details=score_result.details,            # JSON: 各维度明细
    ...
)
```

## 4. 各维度详细设计

### 4.1 Backend-Dev (BigCodeBench)

#### TestCoverageScorer (权重 40%)

**输入**: `ctx.task.metadata["test"]`, `ctx.model_answer`

**逻辑**:
1. 将 `model_answer`（JSON 中的 `code` 字段）与 `test` 代码合并为可执行脚本
2. 执行脚本，捕获 unittest 输出
3. 解析 unittest 输出统计通过/失败数
4. `score = (passed / total) * 100`

**边界情况**:
- `model_answer` 为空或无法提取 code → 0 分
- 执行超时 → 0 分
- 执行崩溃 → 0 分

**可复用**: 复用 ExecutionScorer 的执行环境（tempfile + subprocess）

#### PerformanceScorer (权重 25%)

**前置条件**: 需修改 BigCodeBenchAdapter，将 `canonical_solution` 写入 metadata。

**输入**: `ctx.model_answer`, `ctx.task.metadata["canonical_solution"]`

**逻辑**:
1. 对生成代码和标准答案分别用 `timeit.repeat()` 执行 5 次，取平均
2. `time_ratio = gen_time / canon_time`
3. 如果 `time_ratio < 1.5`: score = 100
4. 如果 `time_ratio < 3`: score = 75
5. 如果 `time_ratio < 10`: score = 40
6. 否则: score = 0

**边界情况**:
- 涉及网络/IO 的题目：跳过性能评分，默认 100 分
- 标准答案不可用：跳过，默认 100 分
- 执行错误：跳过，默认 100 分

#### CodeStyleScorer (权重 15%)

**输入**: `ctx.model_answer`

**逻辑**:
1. 将代码写入临时文件
2. 运行 `pylint --disable=all --enable=C,W,R score` 获取评分
3. `score = pylint_score * 10`（pylint 0-10 → 0-100）
4. 运行 `flake8` 统计违规数，每个违规扣 2 分

#### RobustnessScorer (权重 10%)

**输入**: `ctx.model_answer`

**逻辑** (纯 AST + bandit):
1. AST 解析，统计 try-except 块数量
2. 统计风险操作数（文件 open、网络请求、subprocess）
3. `risky_ops > 0 and try_blocks == 0` → 扣 30 分
4. 有文件操作但没用 with 语句 → 扣 20 分
5. bandit 扫描安全问题 → 每个问题扣 10 分

#### ArchitectureScorer (权重 5%)

**输入**: `ctx.model_answer`

**逻辑** (radon):
1. `radon.complexity` 计算圈复杂度
2. CC > 10 → 扣 15 分, CC > 20 → 扣 30 分
3. 函数长度 > 50 行 → 扣 10 分, > 100 行 → 扣 25 分

#### SecurityScorer (权重 3%)

**输入**: `ctx.model_answer`

**逻辑** (bandit + semgrep):
1. 运行 `bandit -r -f json` 扫描
2. 每个 HIGH severity 问题扣 20 分
3. 每个 MEDIUM 扣 10 分
4. 检测 shell 注入风险 → 扣 30 分
5. 运行 `semgrep --config auto` 扫描，每个发现扣 5 分（上限 20 分）

#### ExtensibilityScorer (权重 2%)

**输入**: `ctx.model_answer`

**逻辑** (AST):
1. 检测硬编码常量（数字字面量）
2. `magic_numbers > 3` → 扣 15 分
3. 检测是否有参数化配置

### 4.2 Frontend-Dev (FrontCode)

#### FunctionalityScorer (权重 30%)

**输入**: `ctx.model_answer`, `ctx.task.test_cases`, `ctx.task.metadata["type"]`

**逻辑**:
1. 根据题目类型选择执行策略:
   - HTML/CSS: 用 Playwright 加载到浏览器，执行 `test_cases` 中的 DOM 断言
   - JavaScript: 用 Node.js 执行代码，运行 `test_cases` 中的断言
   - React: 用 Playwright + @testing-library 渲染组件，执行断言
2. 对每个 test_case: 通过 +1 分，失败 0 分
3. `score = (passed_tests / total_tests) * 100`

**执行环境**:
- Playwright chromium 无头模式
- Node.js 执行 JS 代码
- 超时 30 秒

**边界情况**:
- `test_cases` 为空 → 跳过，默认 100 分
- Playwright/Node.js 未安装 → 跳过，默认 100 分
- 执行超时 → 该用例计 0 分，继续执行其余用例

#### HTMLSemanticScorer (权重 20%)

**输入**: `ctx.model_answer`

**逻辑** (BeautifulSoup + W3C):
1. BeautifulSoup 解析 HTML AST
2. 统计语义标签: header, nav, main, article, section, aside, footer
3. `semantic_ratio = semantic_count / total_elements`
4. semantic_ratio >= 0.6 → 100 分
5. semantic_ratio >= 0.3 → 60 分
6. 检查 heading 层级是否正确（h1 → h2, 不跳级）

#### AccessibilityScorer (权重 15%)

**输入**: `ctx.model_answer`

**逻辑** (axe-core):
1. 将 HTML 加载到 Playwright
2. 运行 `axe-core` 可访问性审计
3. `score = (1 - violations_count / expected_max) * 100`
4. 检查 img alt 文本、表单 label 关联、颜色对比度

#### CSSQualityScorer (权重 15%)

**输入**: `ctx.model_answer`

**逻辑** (Stylelint + AST):
1. 运行 `stylelint` 检查 CSS 规范
2. AST 分析检测媒体查询（响应式设计）
3. 检测相对单位使用（rem/em/%/vw/vh）
4. 检测 flexbox/grid 使用
5. 违规数 = 0 → 100 分, 1-3 个 → 80 分, 4+ → 60 分

#### CodeOrganizationScorer (权重 10%)

**输入**: `ctx.model_answer`

**逻辑** (ESLint):
1. 运行 `eslint` 配置 React/Vue 规则
2. 检查组件命名规范（PascalCase）
3. 检查函数复杂度
4. 违规数 = 0 → 100 分, 1-5 → 80 分, 6+ → 60 分

#### PerformanceScorer (权重 5%)

**输入**: `ctx.model_answer`

**前置条件**: 需要 Node.js 环境 + Playwright

**逻辑** (静态分析 + Playwright):
1. 检测是否有不必要的 DOM 操作（document.querySelectorAll 循环）
2. 检测是否有大量同步操作阻塞渲染
3. 检测图片/资源是否指定了尺寸（避免布局偏移）
4. 用 Playwright 加载页面，测量 `window.performance` 指标
5. 基础分 70 分，每检测到一项性能问题扣 10 分

#### BrowserCompatScorer (权重 5%)

**输入**: `ctx.model_answer`

**逻辑** (AST 分析):
1. 检测 CSS 是否包含 -webkit- / -moz- / -ms- 等厂商前缀
2. 检测是否使用标准属性而非前缀属性（如用 `flex` 而非 `-webkit-flex`）
3. 检测 @supports 特性检测
4. 基础分 80 分
5. 无任何前缀 → 100 分（说明使用现代标准属性）
6. 有厂商前缀但无 @supports → 60 分
7. 有 @supports 兼容检测 → +20 分

### 4.3 Reasoning (MATH)

#### AnswerCorrectnessScorer (权重 40%)

**输入**: `ctx.model_answer`, `ctx.expected`

**逻辑**: 复用 MathScorer 的三阶段匹配（精确 → 归一化 → 数值）
- 完全匹配 → 100 分
- 归一化后匹配 → 100 分
- 数值接近（rel_tol=1e-6）→ 100 分
- 都不匹配 → 0 分

#### ReasoningCompletenessScorer (权重 25%)

**输入**: `ctx.reasoning_content`, `ctx.task.metadata["level"]`

**逻辑** (正则 + token 计数):
1. `reasoning_tokens = len(reasoning_content) / 4`（粗略估算）
2. 期望最小长度: {3: 200, 4: 400, 5: 600}
3. 检测逻辑连接词: "因为", "所以", "因此", "由于", "故"
4. 检测结构标记: "步骤", "首先", "其次", "最后"
5. 加权: 长度 40% + 连接词 30% + 结构 30%

#### ReasoningValidityScorer (权重 20%)

**输入**: `ctx.reasoning_content`, `ctx.task.prompt`, `ctx.task.metadata["level"]`

**逻辑** (LLM-as-a-Judge):
1. 构建 judge prompt:
   ```
   你是数学推理质量评判专家。请检查以下解题过程的正确性。
   题目：{problem}
   模型推理过程：{reasoning}
   评分标准：
   1. 逻辑一致性（40分）
   2. 数学事实正确性（40分）
   3. 计算正确性（20分，允许小失误）
   请给出 0-100 分和简要理由。
   ```
2. 调用 `zai/glm-5.1` API
3. 解析 JSON 响应提取分数
4. 结果缓存: 相同题目 + 相似推理 → 复用缓存

**成本控制**:
- 单次 judge 调用 ~500 tokens
- 15 题 × 500 = 7,500 tokens
- 缓存命中后可降至 ~2,000 tokens

#### MethodEleganceScorer (权重 10%)

**输入**: `ctx.reasoning_content`, `ctx.task.metadata["subject"]`, `ctx.task.metadata["level"]`

**逻辑** (关键词匹配 + 冗余检测):
1. 学科关键词库:
   - Algebra: ["因式分解", "对称性", "换元", "韦达定理"]
   - Geometry: ["辅助线", "相似", "勾股定理"]
   - Number Theory: ["模运算", "整除", "同余", "欧拉函数"]
   - Precalculus: ["三角恒等变换", "复数", "向量"]
2. 每命中一个关键词 +10 分（上限 30 分）
3. 冗余惩罚: `reasoning_tokens > level * 300` → 扣分

#### DifficultyAdaptationScorer (权重 5%)

**输入**: `ctx.reasoning_content`, `ctx.task.metadata["level"]`

**逻辑**:
1. 期望推理深度: {3: 3, 4: 5, 5: 7}
2. 推理深度 = 逻辑连接词数 + 步骤标记数
3. `depth_match = 1 - |actual - expected| / expected`
4. `score = depth_match * 100`

### 4.4 System-Architecture (MMLU-Pro)

#### AnswerCorrectnessScorer (权重 30%)

**输入**: `ctx.model_answer`, `ctx.expected`

**逻辑**: 复用 ChoiceMatchScorer 的正则提取 + 字母匹配
- 正则 `\b([A-Z])\b` 提取选项字母
- 取最后一个匹配
- 与 `expected` 比较: 匹配 100 分, 否则 0 分

#### ReasoningCompletenessScorer (权重 25%)

**输入**: `ctx.reasoning_content`, `ctx.task.metadata["category"]`

**逻辑** (正则匹配):
1. 推理长度检查: 根据学科期望最小长度
2. 选项对比检查: 推理中提到 >= 2 个选项字母 → +30 分
3. 结构检查: 有连接词/分析词 → +30 分
4. 长度比例: `min(tokens / expected, 1.0) * 40` 分

#### OptionAnalysisScorer (权重 20%)

**输入**: `ctx.reasoning_content`, `ctx.task.metadata`

**逻辑** (正则匹配):
1. 检测排除法模式: "排除", "不正确", "错误"
2. 检测对比分析: "而.*不是", "但是.*不对", "however"
3. 统计推理中提到的选项字母数量
4. `mentioned >= 2` → +40, 有排除法 → +30, 有对比 → +30

#### ReasoningConfidenceScorer (权重 15%)

**输入**: `ctx.reasoning_content`, `ctx.model_answer`

**逻辑** (正则匹配):
1. 不确定性检测: "可能", "也许", "不确定", "猜测"
2. 每个不确定性表述扣 10 分（上限 40 分）
3. 确定性检测: "根据", "基于", "因为", "答案是"
4. 每个确定性表述加 5 分（上限 20 分）
5. 答案格式明确: `(?:答案|answer)\s*[：:是]\s*([A-J])` → +10 分
6. 基础分 50 分 + 确定性加成 - 不确定性扣除

#### SubjectAdaptationScorer (权重 10%)

**输入**: `ctx.reasoning_content`, `ctx.task.metadata["category"]`

**逻辑** (查表):
1. 学科期望长度映射: {math: 150, physics: 120, "computer science": 100}
   - 仅覆盖 adapter 实际选择的 3 个学科，未知学科默认 100
2. `match_ratio = min(tokens / expected, 2.0)`
3. `< 0.5` → 40, `< 1.5` → 100, `>= 1.5` → 70

## 5. FrontCode 题目生成

### 5.1 题目结构

复用现有 `TaskDefinition`，不新增模型类。test_cases 使用现有的 `list[str]` 格式（每条为一个断言语句字符串），新增字段放入 metadata：

```python
TaskDefinition(
    task_id="frontcode_xxx",
    dimension="frontend-dev",
    dataset="frontcode",
    prompt="...",
    expected_output="",
    test_cases=["assert document.querySelector('header')", ...],  # 断言列表
    metadata={
        "type": "html",           # html, css, javascript, react, typescript, a11y, performance
        "keywords": [...],
        "difficulty": "medium",   # easy, medium, hard
        "source": "frontcode",
    }
)
```

### 5.2 题目分布

| 类型 | 数量 | 覆盖内容 |
|---|---|---|
| HTML | 2 | 语义化结构, 表单 |
| CSS | 3 | 响应式布局, Grid, 动画, Flexbox |
| JavaScript | 3 | debounce, throttle, 策略模式, 防抖, 日期处理 |
| React | 3 | 组件, Hooks, 状态管理, 表单 |
| TypeScript | 2 | 类型定义, 泛型, 接口 |
| Accessibility | 2 | ARIA, 键盘导航, 屏幕阅读器 |
| 综合 | 2 | 完整页面（多技术栈组合） |

### 5.3 生成方式

1. 用 glm-5.1 生成题目 prompt（覆盖不同类型和难度）
2. 人工审核题目质量和准确性
3. 为每道题自动生成测试用例（断言）
4. 存储到 `benchmark/datasets/frontcode/tasks.json`

## 6. 工具依赖

### 6.1 Python 包

| 包 | 用途 | 安装方式 |
|---|---|---|
| pylint | Python 代码质量评分 | `pip install pylint` |
| flake8 | PEP 8 风格检查 | `pip install flake8` |
| bandit | Python 安全扫描 | `pip install bandit` |
| semgrep | Python 静态分析 | `pip install semgrep` |
| radon | 圈复杂度分析 | `pip install radon` |
| beautifulsoup4 | HTML 解析 | `pip install beautifulsoup4` |
| hypothesis | 属性测试/边界测试 | `pip install hypothesis` |

### 6.2 Node.js 包

| 包 | 用途 | 安装方式 |
|---|---|---|
| eslint | JavaScript/React 规范检查 | `npm install -g eslint` |
| stylelint | CSS 规范检查 | `npm install -g stylelint` |
| @axe-core/playwright | 可访问性测试 | `npm install -g @axe-core/playwright` |
| playwright | 浏览器自动化 | `npm install -g playwright` |

### 6.3 Python 执行环境

| 工具 | 用途 |
|---|---|
| node | 执行 JavaScript |
| npx | 运行 npm 包 |
| playwright | 浏览器自动化 |

## 7. 报告展示

### 7.1 排行榜

```
Model           Score  Correctness  Style  Performance  ...
zai/glm-5.1    85.2   92.0        88.0   75.0        ...
kimi/k2.5      72.1   80.0        70.0   68.5        ...
```

### 7.2 详情页雷达图

每个模型/维度组合展示雷达图：
- Backend: 正确性、性能、风格、鲁棒性、架构、安全
- Frontend: 功能、语义化、a11y、CSS、组织、性能、兼容
- Reasoning: 答案、完整性、正确性、优雅度、难度
- System-Architecture: 答案、完整性、选项分析、置信度、学科

## 8. 文件结构

```
benchmark/scorers/
├── base.py                      # BaseScorer (不变)
├── composite.py                 # CompositeScorer (新增)
├── execution_scorer.py          # ExecutionScorer (保留, 不再直接注册)
├── math_scorer.py              # MathScorer (保留, 复用其匹配逻辑)
├── choice_match_scorer.py       # ChoiceMatchScorer (保留, 复用)
├── keyword_match_scorer.py      # KeywordMatchScorer (保留, 不再直接注册)
├── backend/                     # Backend 子 Scorer 目录 (新增)
│   ├── test_coverage.py
│   ├── performance.py
│   ├── code_style.py
│   ├── robustness.py
│   ├── architecture.py
│   ├── security.py
│   └── extensibility.py
├── frontend/                    # Frontend 子 Scorer 目录 (新增)
│   ├── functionality.py
│   ├── html_semantic.py
│   ├── accessibility.py
│   ├── css_quality.py
│   ├── code_organization.py
│   ├── performance.py
│   └── browser_compat.py
├── reasoning/                   # Reasoning 子 Scorer 目录 (新增)
│   ├── answer_correctness.py     # 复用 MathScorer
│   ├── reasoning_completeness.py
│   ├── reasoning_validity.py     # LLM-as-a-Judge
│   ├── method_elegance.py
│   └── difficulty_adaptation.py
└── system_architecture/         # System-Architecture 子 Scorer 目录 (新增)
    ├── answer_correctness.py     # 复用 ChoiceMatchScorer
    ├── reasoning_completeness.py
    ├── option_analysis.py
    ├── reasoning_confidence.py
    └── subject_adaptation.py
```

## 9. 性能影响

### 9.1 评分耗时估算

| 维度 | 当前耗时 | 新增耗时 | 主要耗时来源 |
|---|---|---|---|
| Backend | 5s (执行) | +3s (pylint/flake8) | 静态分析 |
| Frontend | 0s (关键词) | +10s (Playwright) | 浏览器启动 |
| Reasoning | 0s (匹配) | +2s (LLM Judge) | API 调用 |
| System-Arch | 0s (匹配) | 0s (正则) | 无 |

### 9.2 缓存策略

- **LLM Judge 缓存**: 按 `(problem_hash + reasoning_hash)` 缓存，相同题目复用
- **工具结果缓存**: pylint/flake8/bandit 结果按代码内容哈希缓存
- **Playwright 预热**: 评测开始前启动浏览器实例，后续复用

## 10. 实施顺序

### Phase 1: 基础设施 + Backend + System-Architecture (✅ 已完成)
- CompositeScorer 实现
- 修改 BigCodeBenchAdapter，将 `canonical_solution` 写入 metadata
- Backend 7 个子 Scorer（不含 LLM 依赖的维度）
- System-Architecture 5 个子 Scorer（全部正则匹配）
- DIMENSION_REGISTRY 更新（全部 4 个维度统一使用 CompositeScorer）
- 数据库写入逻辑更新

### Phase 2: Frontend + MATH (✅ 已完成)
- Frontend 7 个子 Scorer
- Playwright 执行环境搭建
- MATH 5 个子 Scorer（含 LLM Judge）
- LLM Judge 缓存机制

### Phase 3: 题目扩展 + 报告 (✅ 已完成)
- FrontCode 题目扩展到 22 题（覆盖 7 种类型）
- 雷达图报告生成（SVG 动态渲染）
- 排行榜各维度分数展示

## 11. DIMENSION_REGISTRY

所有维度统一使用 CompositeScorer 工厂函数：

```python
DIMENSION_REGISTRY = {
    "reasoning": (MATHAdapter, create_reasoning_composite, SingleTurnEvaluator),
    "backend-dev": (BigCodeBenchAdapter, create_backend_composite, SingleTurnEvaluator),
    "system-architecture": (MMLUProAdapter, create_sysarch_composite, SingleTurnEvaluator),
    "frontend-dev": (FrontCodeAdapter, create_frontend_composite, SingleTurnEvaluator),
}
```

`_run_evaluation` 中统一创建 scorer：
- `reasoning`：`CompositeScorer(create_reasoning_composite(llm=llm))`（需要 llm 传给 LLM Judge）
- 其他维度：`CompositeScorer(scorer_factory())`

## 12. FrontCode 题目分布

| 类型 | 数量 | 难度 |
|------|------|------|
| html | 3 | 2 easy + 1 medium |
| css | 4 | 1 easy + 3 medium |
| javascript | 4 | 4 medium |
| react | 4 | 4 medium |
| typescript | 2 | 2 medium |
| accessibility | 2 | 2 hard |
| complex | 3 | 1 medium + 2 hard |
| **总计** | **22** | 2 easy + 11 medium + 4 hard + 5 unknown |

所有 22 题均包含 `test_cases` 和 `difficulty` 字段。
