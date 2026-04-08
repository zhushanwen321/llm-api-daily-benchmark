# MATH 数据集多维度数学推理评分方案

## 1. 当前评分机制分析

### 1.1 现有实现

当前 `MathScorer` 采用三阶段匹配策略：
1. **字符串精确匹配**：predicted == expected
2. **空格归一化匹配**：逗号和括号周围空格归一化后比较
3. **数值比较**：LaTeX → Python 表达式 → 安全 eval → 数值比较

### 1.2 核心问题

**无法区分以下情况：**
- 答案对但过程完全错误（蒙对的，Level 5 题目概率虽低但存在）
- 答案错但思路正确（计算失误、符号错误）
- 解题过程的优雅程度（暴力枚举 vs 智巧构造）
- 推理深度（Level 5 应该有更深入的推理，但当前不区分）

**示例：**
```
题目：求 ∑_{j=1}^∞ ∑_{k=1}^∞ 1/(j+k)^3（用 p = ∑ 1/k² 和 q = ∑ 1/k³ 表示）

答案正确的情况：
A: 直接暴力枚举前几项，猜出答案 p - q（运气好）
B: 系统分析 j+k=n 的出现次数，推导出 ∑ (n-1)/n³ = p - q（真懂）

答案错误的情况：
C: 推理过程完美，但最后写成 q - p（符号错误）
D: 完全不会，瞎蒙一个答案

当前评分：A=100, B=100, C=0, D=0
理想评分：A≈70, B=100, C≈80, D=0
```

## 2. 数据集特征分析

### 2.1 MATH 数据集优势

MATH 是构造题（非选择题），天然适合过程评分：
- 每题有 `problem` + `solution` + `answer`
- `solution` 包含完整的解题步骤
- `subject` 标识学科（代数、几何、数论等）
- `level` 标识难度（1-5）

### 2.2 Thinking Model 的独特优势

**关键发现：** `ScoringContext.reasoning_content` 就是天然的解题过程

```python
class ScoringContext(BaseModel):
    reasoning_content: str = ""  # 推理过程（从 API reasoning_content 获取）
```

这是我们的核心优势：
- **非选择题的优势**：MATH 不像 MMLU-Pro（选择题），reasoning_content 真正反映解题思路
- **可解释性**：可以直接检查推理质量，而非只能看答案对错
- **过程完整性**：可以检查是否有关键步骤缺失

## 3. 多维度评分设计

### 3.1 评分维度拆分

#### 维度 1：答案正确性（Answer Correctness）
**权重：40%**

现有三阶段匹配已经很完善，保留但降低权重：
- 数值/符号精确匹配：100%
- 等价形式（如 π/2 vs 90°）：部分分数

#### 维度 2：推理完整性（Reasoning Completeness）
**权重：25%**

**检查点：**
1. **关键步骤覆盖**：是否遗漏关键推理步骤
   - Level 3-5 题目通常需要 3-5 个关键步骤
   - 检查是否跳过了中间步骤

2. **结论链连贯性**：是否有明确的逻辑链条
   - "因为 A → 所以 B → 因此 C" 的结构
   - 避免直接从题目跳到答案

**量化方法：**
```python
def score_reasoning_completeness(reasoning: str, level: int) -> float:
    # 检查推理长度（token 数）
    reasoning_tokens = count_tokens(reasoning)

    # 根据难度设定期望长度
    expected_min_tokens = {
        3: 200,   # Level 3 至少 200 tokens
        4: 400,   # Level 4 至少 400 tokens
        5: 600,   # Level 5 至少 600 tokens
    }

    # 检查关键连接词
    transition_words = ["因为", "所以", "因此", "由于", "故", "从而"]
    transition_count = sum(1 for word in transition_words if word in reasoning)

    # 检查是否有步骤编号或明确的结构
    has_structure = bool(
        re.search(r'(步骤|Step|首先|其次|最后|第一|第二)', reasoning)
    )

    score = 0.0
    if reasoning_tokens >= expected_min_tokens.get(level, 200):
        score += 40
    if transition_count >= 2:
        score += 30
    if has_structure:
        score += 30

    return score
```

#### 维度 3：推理正确性（Reasoning Validity）
**权重：20%**

**检查点：**
1. **逻辑一致性**：推理步骤之间是否自洽
2. **数学事实正确性**：是否使用了错误的定理或公式
3. **计算中间步骤**：关键计算是否正确

**量化方法（创新点）：**
```python
def score_reasoning_validity(
    reasoning: str,
    problem: str,
    answer: str,
    level: int
) -> float:
    """用 LLM-as-a-Judge 检查推理正确性"""

    # 构建 few-shot prompt
    judge_prompt = f"""
你是数学推理质量评判专家。请检查以下解题过程的正确性。

题目：
{problem}

模型推理过程：
{reasoning}

模型答案：{answer}

评分标准：
1. 逻辑一致性（40分）：推理步骤是否自洽，是否存在矛盾
2. 数学事实正确性（40分）：使用的定理、公式、性质是否正确
3. 计算正确性（20分）：关键计算步骤是否正确（允许小的计算失误）

请给出 0-100 分，并说明理由。
"""

    # 使用高质量的 reasoning model 作为 judge
    # 可以用 claude-3.7-sonnet 或 gpt-4o
    judge_score = call_judge_model(judge_prompt)

    return judge_score
```

**优化策略：**
- **缓存 Judge 结果**：相同题目只评判一次
- **批量评判**：一次 API 调用评判多个题目
- **弱监督学习**：用人工标注的小样本训练轻量级 classifier

#### 维度 4：方法优雅度（Method Elegance）
**权重：10%**

**检查点：**
1. **方法选择**：是否使用了最优解法
   - 几何题：是否用了巧妙的辅助线，而非暴力计算
   - 代数题：是否用了因式分解、对称性等技巧
   - 数论题：是否用了模运算、整除性质等

2. **简洁性**：是否避免了冗长的无效推理

**量化方法：**
```python
def score_method_elegance(
    reasoning: str,
    subject: str,
    level: int,
    reasoning_tokens: int
) -> float:
    """评分方法的优雅程度"""

    score = 50.0  # 基础分

    # 奖励：使用巧妙的技巧（根据学科）
    elegant_keywords = {
        "Algebra": ["因式分解", "对称性", "换元", "韦达定理"],
        "Geometry": ["辅助线", "相似", "勾股定理", "圆幂定理"],
        "Number Theory": ["模运算", "整除", "同余", "欧拉函数"],
        "Precalculus": ["三角恒等变换", "复数", "向量"],
    }

    subject_keywords = elegant_keywords.get(subject, [])
    keyword_count = sum(1 for kw in subject_keywords if kw in reasoning)

    # 使用技巧越多，分数越高
    score += min(keyword_count * 10, 30)

    # 惩罚：过度冗长（根据难度调整阈值）
    max_expected_tokens = level * 300  # Level 3: 900, Level 5: 1500
    if reasoning_tokens > max_expected_tokens:
        # 超出部分扣分
        excess_ratio = (reasoning_tokens - max_expected_tokens) / max_expected_tokens
        score -= min(excess_ratio * 20, 30)

    return max(min(score, 100), 0)
```

#### 维度 5：难度适应性（Difficulty Adaptation）
**权重：5%**

**检查点：**
- Level 5 题目应该有更深入的推理
- Level 3 题目允许相对简洁的解答

**量化方法：**
```python
def score_difficulty_adaptation(
    reasoning_tokens: int,
    reasoning_depth: int,
    level: int
) -> float:
    """评分推理深度与题目难度的匹配度"""

    # 推理深度：关键推理步骤的数量
    # 可以通过检测句子数量、逻辑连接词数量来估算

    expected_depth = {3: 3, 4: 5, 5: 7}
    actual_depth = reasoning_depth

    # 深度匹配度
    depth_match = 1 - abs(actual_depth - expected_depth[level]) / expected_depth[level]

    return depth_match * 100
```

### 3.2 综合评分公式

```python
final_score = (
    answer_correctness * 0.40 +
    reasoning_completeness * 0.25 +
    reasoning_validity * 0.20 +
    method_elegance * 0.10 +
    difficulty_adaptation * 0.05
)
```

## 4. 与选择题的差异

### 4.1 MMLU-Pro（选择题）的局限

选择题的 reasoning_content 不太可靠：
- 模型可能直接选答案，reasoning 是事后编造的
- 无法区分"真正推理"和"合理化答案"

### 4.2 MATH（构造题）的优势

构造题的 reasoning_content 更可信：
- 必须通过推理才能构造答案
- reasoning_content 直接反映解题思路
- 可以直接检查推理质量

**结论：** 多维度评分更适合 MATH，而非 MMLU-Pro

## 5. 实现策略

### 5.1 分阶段实施

**Phase 1：答案正确性 + 推理完整性**
- 保留现有 answer matching
- 添加 reasoning completeness 检查
- 目标：区分"蒙对"和"真懂"

**Phase 2：推理正确性（LLM-as-a-Judge）**
- 实现轻量级 judge 模型
- 批量评判，优化成本
- 目标：区分"思路对但算错"和"完全不会"

**Phase 3：方法优雅度 + 难度适应性**
- 实现优雅度评分
- 根据学科定制评分规则
- 目标：鼓励巧妙的解法

### 5.2 性能优化

1. **缓存机制**
   ```python
   # 缓存 judge 结果
   judge_cache = {}
   cache_key = hashlib.md5((problem + reasoning).encode()).hexdigest()
   if cache_key in judge_cache:
       return judge_cache[cache_key]
   ```

2. **批量评判**
   ```python
   # 一次 API 调用评判多个题目
   batch_prompt = "请评判以下 10 个解题过程..."
   results = call_judge_model(batch_prompt)
   ```

3. **异步处理**
   ```python
   # reasoning_validity 可以异步计算
   async def score_async(ctx):
       completeness = score_completeness(ctx)  # 同步，快速
       validity = await async_judge(ctx)        # 异步，慢速
       return combine(completeness, validity)
   ```

### 5.3 成本控制

**LLM-as-a-Judge 的成本估算：**
- 每题评判约 500 tokens（prompt + response）
- 15 题 × 500 tokens = 7,500 tokens
- Claude 3.7 Sonnet: $3/1M tokens (input) + $15/1M tokens (output)
- 单次评测成本：~$0.02

**优化策略：**
- 使用更便宜的 judge 模型（如 GPT-4o-mini）
- 只对答案错误的题目进行 full scoring
- 对答案正确的题目进行 sampling 验证

## 6. 评分示例

### 6.1 完美解答（Level 5）

```
题目：求 ∑_{j=1}^∞ ∑_{k=1}^∞ 1/(j+k)^3（用 p = ∑ 1/k² 和 q = ∑ 1/k³ 表示）

推理过程：
"我们分析 1/n³ 在双重求和中的出现次数。固定 n，每次 j + k = n 时就会出现 1/n³。
满足条件的数对 (j,k) 有 (1, n-1), (2, n-2), ..., (n-1, 1)，共 n-1 对。
因此，原式 = ∑_{n=1}^∞ (n-1)/n³ = ∑ 1/n² - ∑ 1/n³ = p - q"

答案：p - q

评分：
- 答案正确性：100/100（精确匹配）
- 推理完整性：100/100（步骤完整，逻辑清晰）
- 推理正确性：100/100（逻辑严密，数学事实正确）
- 方法优雅度：90/100（巧妙的变量替换，简洁）
- 难度适应性：100/100（深度匹配 Level 5）

总分：95.5
```

### 6.2 运气好（Level 5）

```
推理过程：
"试试前几项：j=1,k=1 → 1/8, j=1,k=2 → 1/27, j=2,k=1 → 1/27...
看起来像是 p - q？"

答案：p - q

评分：
- 答案正确性：100/100（精确匹配）
- 推理完整性：30/100（只有 50 tokens，无明确步骤）
- 推理正确性：40/100（逻辑不完整，缺乏严格证明）
- 方法优雅度：50/100（暴力枚举，非最优解）
- 难度适应性：20/100（深度远低于 Level 5）

总分：56.5
```

### 6.3 符号错误（Level 4）

```
推理过程：
"固定 n，每次 j + k = n 时出现 1/n³，共 n-1 对。
原式 = ∑_{n=1}^∞ (n-1)/n³ = ∑ 1/n² - ∑ 1/n³ = q - p"  # 符号错误

答案：q - p

评分：
- 答案正确性：0/100（符号错误）
- 推理完整性：90/100（步骤完整，逻辑清晰）
- 推理正确性：80/100（方法正确，最后符号错误）
- 方法优雅度：90/100（方法巧妙）
- 难度适应性：90/100（深度匹配 Level 4）

总分：62.5
```

## 7. 实验设计

### 7.1 对比实验

**Baseline：** 现有的 MathScorer（只看答案）

**Proposed：** 多维度评分

**评估指标：**
1. 与人工评分的相关性（Pearson correlation）
2. 区分度：能否区分"蒙对"和"真懂"
3. 稳定性：同一模型同一题目的多次评分是否一致

### 7.2 人工标注

**标注 50 个样本（每个 level 各 10 个）：**
- 完美解答：10 个
- 思路对但有计算错误：15 个
- 运气好答案对：10 个
- 完全错误：15 个

**标注内容：**
- 答案正确性（binary）
- 推理完整性（0-100）
- 推理正确性（0-100）
- 方法优雅度（0-100）
- 总体质量（0-100）

**评估：**
- 计算自动评分与人工评分的 correlation
- 分析每个维度的有效性

## 8. 未来扩展

### 8.1 自适应评分

根据模型的能力调整评分权重：
- 对于弱模型：提高"答案正确性"权重（如 60%）
- 对于强模型：提高"方法优雅度"权重（如 20%）

### 8.2 学科定制

根据不同学科的特点调整评分规则：
- **几何**：更重视图形理解和辅助线技巧
- **代数**：更重视因式分解、对称性等技巧
- **数论**：更重视模运算、整除性质等技巧

### 8.3 教育应用

- **个性化反馈**：根据每个维度的得分，给出具体的改进建议
- **学习路径**：识别模型的薄弱环节，推荐针对性的训练数据

## 9. 结论

MATH 数据集的构造题特性 + thinking model 的 reasoning_content，为多维度评分提供了独特优势。建议：

1. **立即实施** Phase 1（答案正确性 + 推理完整性）
2. **逐步推进** Phase 2-3（推理正确性 + 方法优雅度）
3. **持续优化** 根据实验结果调整权重和算法

这个方案不仅能更准确地评估数学推理能力，还能为模型改进提供更细粒度的反馈。
