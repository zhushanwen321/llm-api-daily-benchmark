# MMLU-Pro 多维度知识问答评分系统架构分析

## 1. 数据集特征分析

### 1.1 MMLU-Pro 数据集特点

**基本信息：**
- 总题量：246,343 题
- 学科分布：14 个学科（数学 1351 题、物理 1299 题、化学 1132 题、法律 1101 题、工程 969 题等）
- 题目格式：多选题，每题 5-10 个选项
- 答案格式：单个字母（A-J）

**题目结构：**
```json
{
  "question_id": 70,
  "question": "Typical advertising regulatory bodies suggest...",
  "options": [
    "Safe practices, Fear, Jealousy, Trivial",
    "Unsafe practices, Distress, Joy, Trivial",
    // ... 7 more options
  ],
  "answer": "I",
  "answer_index": 8,
  "cot_content": "",
  "category": "business",
  "src": "ori_mmlu-business_ethics"
}
```

**关键观察：**
1. **选项数量变化**：5-10 个选项（不固定）
2. **选项复杂度高**：很多选项是完整句子或多个概念的组合
3. **cot_content 为空**：官方提供的 CoT 示例未包含在数据集中
4. **学科差异大**：从数学（抽象推理）到历史（事实记忆）再到计算机科学（实践知识）

### 1.2 当前评分机制分析

**当前实现：`ChoiceMatchScorer`**

```python
class ChoiceMatchScorer(BaseScorer):
    """选择题字母匹配评分器，用于 MMLU 选择题。

    从模型输出中提取最后的选项字母（A/B/C/D/...），
    与 expected_output（正确选项字母）进行不区分大小写的比较。
    匹配成功 score=100，失败 score=0。
    """
```

**评分逻辑：**
1. 从模型输出中提取所有选项字母（A-Z）
2. 取最后一个匹配的字母（假设最终答案在最后）
3. 与正确答案比较，匹配则 100 分，否则 0 分

**核心问题：**

**问题 1：答案匹配的表面性**
- 无法区分"真正理解"和"蒙对"
- 无法区分"思路正确但选错"和"完全不会"

**问题 2：多选题的格式限制**
- 推理过程可能只是"合理化"已选答案
- 模型可能直接选择答案，然后编造推理

**问题 3：学科差异被忽视**
- 数学题需要深度推理
- 历史题可能只需要事实记忆
- 计算机科学题需要实践判断

## 2. 多维度评分设计

### 2.1 可行维度分析

**核心挑战：** 多选题的格式限制了评估深度

**关键区分：**
- MATH（构造题）：reasoning_content 是解题过程
- MMLU-Pro（选择题）：reasoning_content 可能是"合理化"

#### 维度 1：答案正确性（Answer Correctness）
**权重：30%**（降低，因为只是表面指标）

**现有实现已经足够：**
```python
predicted = extract_last_choice_letter(model_output)
expected = correct_answer
score = 100 if predicted == expected else 0
```

**问题：**
- 9 个选项的题目，随机猜测概率 11%
- 无法区分"蒙对"和"真懂"

#### 维度 2：推理完整性（Reasoning Completeness）
**权重：25%**

**适用场景：**
- 检查模型是否提供了推理过程
- 推理过程是否包含关键步骤

**量化方法：**
```python
def score_reasoning_completeness(
    reasoning: str,
    category: str,
    num_options: int
) -> float:
    """评分推理完整性"""

    score = 0.0

    # 检查推理长度（token 数）
    reasoning_tokens = count_tokens(reasoning)
    if reasoning_tokens == 0:
        return 0.0  # 没有推理过程

    # 根据学科和选项数量设定期望长度
    expected_min_tokens = {
        "math": 150,      # 数学需要推理
        "physics": 120,   # 物理需要解释
        "history": 80,    # 历史可以简洁
        "computer science": 100,
    }

    base_score = min(reasoning_tokens / expected_min_tokens.get(category, 100), 1.0) * 40
    score += base_score

    # 检查是否分析了多个选项
    # 真正的推理应该对比多个选项，而非直接选一个
    mentioned_choices = len(set(re.findall(r'\b([A-J])\b', reasoning)))
    if mentioned_choices >= 2:
        score += 30  # 至少提到两个选项，说明有对比

    # 检查是否有明确的结构
    has_structure = bool(
        re.search(r'(因为|所以|因此|由于|故|从而|因为|所以)', reasoning) or
        re.search(r'(分析|排除|考虑|判断)', reasoning)
    )
    if has_structure:
        score += 30

    return score
```

**局限性：**
- 模型可以编造推理过程
- 无法验证推理的真实性

#### 维度 3：选项分析深度（Option Analysis Depth）
**权重：20%**

**核心思想：** 真正的理解应该包含对错误选项的分析

**量化方法：**
```python
def score_option_analysis_depth(
    reasoning: str,
    question: str,
    options: list[str],
    correct_answer: str
) -> float:
    """评分选项分析深度"""

    score = 0.0

    # 检查是否分析了错误选项
    # 真正理解应该知道为什么其他选项是错的
    incorrect_answers = [chr(65+i) for i in range(len(options))
                        if chr(65+i) != correct_answer]

    mentioned_incorrect = sum(1 for ans in incorrect_answers
                             if ans in reasoning)

    # 至少分析一个错误选项
    if mentioned_incorrect >= 1:
        score += 40

    # 检查是否有排除法
    exclusion_patterns = [
        r'排除',
        r'不正确',
        r'错误',
        r'not.*correct',
        r'exclude',
    ]
    has_exclusion = any(re.search(pattern, reasoning, re.IGNORECASE)
                       for pattern in exclusion_patterns)

    if has_exclusion:
        score += 30

    # 检查是否有对比分析
    contrast_patterns = [
        r'而.*不是',
        r'但是.*不对',
        r'however',
        r'although',
        r'选项.*不如.*选项',
    ]
    has_contrast = any(re.search(pattern, reasoning, re.IGNORECASE)
                      for pattern in contrast_patterns)

    if has_contrast:
        score += 30

    return score
```

**创新点：**
- 强制模型分析错误选项
- 鼓励排除法，而非直接选择
- 区分"知道为什么对"和"只知道对"

#### 维度 4：推理置信度（Reasoning Confidence）
**权重：15%**

**核心思想：** 真正理解应该有明确的判断依据

**量化方法：**
```python
def score_reasoning_confidence(
    reasoning: str,
    model_output: str
) -> float:
    """评分推理置信度"""

    score = 50.0  # 基础分

    # 检查是否有不确定的表述
    uncertainty_patterns = [
        r'可能',
        r'也许',
        r'不确定',
        r'猜测',
        r'好像',
        r'perhaps',
        r'maybe',
        r'uncertain',
        r'guess',
    ]

    uncertainty_count = sum(1 for pattern in uncertainty_patterns
                           if re.search(pattern, reasoning, re.IGNORECASE))

    # 不确定性越多，分数越低
    score -= min(uncertainty_count * 10, 40)

    # 检查是否有明确的判断依据
    confidence_patterns = [
        r'根据',
        r'基于',
        r'因为',
        r'由于',
        r'答案是',
        r'based on',
        r'because',
        r'therefore',
        r'answer is',
    ]

    confidence_count = sum(1 for pattern in confidence_patterns
                          if re.search(pattern, reasoning, re.IGNORECASE))

    score += min(confidence_count * 5, 20)

    # 检查最终答案是否明确
    final_answer_pattern = r'(?:答案|answer)\s*[：:是]\s*([A-J])'
    if re.search(final_answer_pattern, model_output, re.IGNORECASE):
        score += 10

    return max(min(score, 100), 0)
```

#### 维度 5：学科适应性（Subject Adaptation）
**权重：10%**

**核心思想：** 不同学科应该有不同的评分标准

**量化方法：**
```python
def score_subject_adaptation(
    reasoning: str,
    category: str,
    reasoning_tokens: int
) -> float:
    """评分学科适应性"""

    # 不同学科的期望推理长度
    expected_tokens = {
        "math": 150,      # 数学需要推理
        "physics": 120,   # 物理需要解释
        "chemistry": 100, # 化学需要解释
        "history": 80,    # 历史可以简洁
        "philosophy": 140,  # 哲学需要深入思考
        "computer science": 100,  # 计算机科学需要实践判断
        "business": 90,   # 商业需要分析
        "law": 110,       # 法律需要论证
        "engineering": 100,  # 工程需要分析
        "economics": 95,  # 经济学需要分析
        "psychology": 100,  # 心理学需要分析
        "health": 90,     # 健康需要分析
        "biology": 100,   # 生物需要分析
        "other": 100,
    }

    expected = expected_tokens.get(category, 100)

    # 推理长度与期望匹配度
    match_ratio = min(reasoning_tokens / expected, 2.0)  # 最多 2 倍
    if match_ratio < 0.5:
        score = 40  # 太短
    elif match_ratio < 1.5:
        score = 100  # 合适
    else:
        score = 70  # 太长，可能有冗余

    return score
```

### 2.2 不可行维度分析

#### 维度 X：推理正确性（Reasoning Validity）
**为什么不可行：**

**问题 1：推理可能是编造的**
```
模型可能的推理过程：
"我选 A，因为...（编造一些理由）"
```
这种推理可能是"合理化"，而非真正的推理路径。

**问题 2：选择题的"捷径"**
- 模型可能直接选择答案
- 然后根据答案编造推理
- 无法区分"先推理再选择"和"先选择再合理化"

**问题 3：验证成本高**
- LLM-as-a-Judge 的成本：每题 ~500 tokens
- 246,343 题的总成本：~123M tokens
- 成本过高，而且可能不准确

#### 维度 Y：方法优雅度（Method Elegance）
**为什么不可行：**

**问题 1：选择题没有"方法"**
- 构造题：可以选择不同的解题方法
- 选择题：只需要选择答案，没有"方法"可言

**问题 2：无法区分推理质量**
```
选项 A：正确答案，模型直接选 A
选项 B：错误答案，模型通过推理排除 B
```
这两种情况在选择题中无法区分，因为最终结果都是"选 A"。

#### 维度 Z：干扰项识别（Distractor Recognition）
**为什么不可行：**

**问题 1：干扰项的性质**
- 有些干扰项是明显错误的（容易排除）
- 有些干扰项是"似是而非"的（需要深入分析）
- 无法量化"识别难度"

**问题 2：识别的深度难以评估**
```
"排除 B，因为 B 明显错误"  vs  "排除 B，因为 B 与 X 矛盾"
```
前者的深度明显不如后者，但难以自动区分。

## 3. 与 MATH 数据集的对比

### 3.1 格式差异

| 维度 | MATH（构造题） | MMLU-Pro（选择题） |
|------|---------------|-------------------|
| 答案格式 | 自由构造（LaTeX 表达式） | 从选项中选择（A-J） |
| 推理过程 | 必须有推理才能构造答案 | 可能先选答案再编造推理 |
| 验证难度 | 可以检查推理步骤 | 只能验证答案对错 |
| 评估深度 | 可以评估方法优雅度 | 只能评估表面指标 |

### 3.2 多维度评分的适用性

**MATH 数据集：适合多维度评分**
- 推理过程可信（必须通过推理构造答案）
- 可以评估推理完整性
- 可以评估方法优雅度
- 可以评估推理正确性

**MMLU-Pro 数据集：不适合深度多维度评分**
- 推理过程不可信（可能是合理化）
- 无法评估方法优雅度（没有"方法"）
- 无法评估推理正确性（验证成本高）

**结论：** MMLU-Pro 的多维度评分应该"轻量化"，重点关注可验证的维度

## 4. 多维度评分方案

### 4.1 推荐维度配置

```python
MMLU_PRO_SCORING_CONFIG = {
    "dimensions": {
        "answer_correctness": {
            "weight": 0.30,
            "description": "答案正确性",
            "implementation": "existing ChoiceMatchScorer"
        },
        "reasoning_completeness": {
            "weight": 0.25,
            "description": "推理完整性",
            "metrics": ["reasoning_length", "option_comparison", "structure"]
        },
        "option_analysis_depth": {
            "weight": 0.20,
            "description": "选项分析深度",
            "metrics": ["incorrect_option_analysis", "exclusion_method", "contrast_analysis"]
        },
        "reasoning_confidence": {
            "weight": 0.15,
            "description": "推理置信度",
            "metrics": ["uncertainty_penalty", "confidence_markers", "final_answer_clarity"]
        },
        "subject_adaptation": {
            "weight": 0.10,
            "description": "学科适应性",
            "metrics": ["reasoning_length_match", "subject_specific_patterns"]
        }
    }
}
```

### 4.2 综合评分公式

```python
final_score = (
    answer_correctness * 0.30 +
    reasoning_completeness * 0.25 +
    option_analysis_depth * 0.20 +
    reasoning_confidence * 0.15 +
    subject_adaptation * 0.10
)
```

### 4.3 及格线设计

```python
# 及格线设计
PASSING_THRESHOLD = 60.0

# 不同等级的划分
EXCELLENT = 85.0  # 优秀
GOOD = 70.0       # 良好
PASSING = 60.0    # 及格
```

## 5. 实现策略

### 5.1 分阶段实施

**Phase 1：答案正确性 + 推理完整性**
- 保留现有的 ChoiceMatchScorer
- 添加 reasoning_completeness 检查
- 目标：鼓励模型提供推理过程

**Phase 2：选项分析深度**
- 添加 option_analysis_depth 检查
- 鼓励模型分析错误选项
- 目标：区分"知道为什么对"和"只知道对"

**Phase 3：推理置信度 + 学科适应性**
- 添加 reasoning_confidence 检查
- 添加 subject_adaptation 检查
- 目标：区分"确定的理解"和"猜测"

### 5.2 性能优化

**优化 1：快速路径**
```python
def score(ctx: ScoringContext) -> ScoreResult:
    # 如果答案错误，可以跳过某些维度的评分
    if not answer_correctness_check(ctx):
        return ScoreResult(
            score=0.0,
            passed=False,
            details={"reason": "wrong_answer"}
        )

    # 答案正确，才进行完整的多维度评分
    return full_multi_dimension_score(ctx)
```

**优化 2：缓存机制**
```python
# 缓存学科特定的模式
SUBJECT_PATTERNS_CACHE = {}

def get_subject_patterns(category: str) -> dict:
    if category not in SUBJECT_PATTERNS_CACHE:
        SUBJECT_PATTERNS_CACHE[category] = compile_subject_patterns(category)
    return SUBJECT_PATTERNS_CACHE[category]
```

**优化 3：并行评分**
```python
async def ascore(ctx: ScoringContext) -> ScoreResult:
    # 并行计算各个维度
    results = await asyncio.gather(
        score_answer_correctness(ctx),
        score_reasoning_completeness(ctx),
        score_option_analysis_depth(ctx),
        score_reasoning_confidence(ctx),
        score_subject_adaptation(ctx),
    )

    return combine_scores(results)
```

### 5.3 成本控制

**成本估算：**
- 答案正确性：~0 tokens（现有实现）
- 推理完整性：~0 tokens（简单的正则匹配）
- 选项分析深度：~0 tokens（简单的正则匹配）
- 推理置信度：~0 tokens（简单的正则匹配）
- 学科适应性：~0 tokens（简单的正则匹配）

**总成本：** ~0 tokens（无需额外的 LLM 调用）

**与 MATH 的对比：**
- MATH：需要 LLM-as-a-Judge，成本高
- MMLU-Pro：不需要 LLM-as-a-Judge，成本低

## 6. 评分示例

### 6.1 完美解答

```
题目：Typical advertising regulatory bodies suggest, for example that adverts
must not: encourage _________, cause unnecessary ________ or _____, and must
not cause _______ offence.

选项：9 个选项（A-I）

正确答案：I

模型推理过程：
"分析题目：广告监管机构对广告的要求是：
1. 不能鼓励某种做法
2. 不能造成不必要的某种情绪或行为
3. 不能造成某种类型的冒犯

分析选项：
A: Safe practices, Fear, Jealousy, Trivial - 鼓励安全 practices 不符合题意
B: Unsafe practices, Distress, Joy, Trivial - Joy 不符合负面情绪
...
I: Unsafe practices, Distress, Fear, Serious - 所有空格都符合题意

排除法：
- A、C、D、F、G 都包含 Safe practices，不符合第一空
- B 包含 Joy，不符合负面情绪
- E、H 包含 Wants，不如 Distress 专业
I 是唯一符合所有条件的选项。

答案：I"
```

**评分：**
- 答案正确性：100/100
- 推理完整性：90/100（结构清晰，分析了多个选项）
- 选项分析深度：95/100（分析了 9 个选项中的 8 个）
- 推理置信度：85/100（有明确的分析过程）
- 学科适应性：80/100（推理长度适中）

**总分：88.5**

### 6.2 直接选择

```
模型推理过程：
"选 I"

答案：I
```

**评分：**
- 答案正确性：100/100
- 推理完整性：10/100（几乎没有推理）
- 选项分析深度：0/100（没有分析其他选项）
- 推理置信度：50/100（答案明确但无推理）
- 学科适应性：30/100（推理过短）

**总分：45.5**

### 6.3 分析错误

```
模型推理过程：
"广告监管要求不能鼓励错误做法，不能造成痛苦或恐惧，不能造成严重冒犯。

选项分析：
A: Safe practices - 安全做法不对，应该是错误做法
B: Unsafe practices - 错误做法，符合
...
I: Unsafe practices, Distress, Fear, Serious - 看起来最合理

答案：B"  # 选择错误
```

**评分：**
- 答案正确性：0/100
- 推理完整性：70/100（有分析过程）
- 选项分析深度：60/100（分析了选项）
- 推理置信度：60/100（有一定推理）
- 学科适应性：70/100（推理长度适中）

**总分：40.0**

### 6.4 蒙对

```
模型推理过程：
"看起来应该是 I 吧，因为其他选项不太对"

答案：I
```

**评分：**
- 答案正确性：100/100
- 推理完整性：20/100（推理过短）
- 选项分析深度：10/100（没有具体分析）
- 推理置信度：30/100（不确定性高）
- 学科适应性：20/100（推理过短）

**总分：43.0**

## 7. 实验设计

### 7.1 对比实验

**Baseline：** 现有的 ChoiceMatchScorer（只看答案）

**Proposed：** 多维度评分

**评估指标：**
1. 与人工评分的相关性（Pearson correlation）
2. 区分度：能否区分"蒙对"和"真懂"
3. 稳定性：同一模型同一题目的多次评分是否一致

### 7.2 人工标注

**标注 100 个样本（每个主要学科 10 个）：**
- 完美解答：20 个
- 良好解答：30 个
- 运气好答案对：20 个
- 分析错误但答案对：15 个
- 完全错误：15 个

**标注内容：**
- 答案正确性（binary）
- 推理完整性（0-100）
- 选项分析深度（0-100）
- 推理置信度（0-100）
- 总体质量（0-100）

**评估：**
- 计算自动评分与人工评分的 correlation
- 分析每个维度的有效性
- 调整权重和算法

## 8. 与 MATH 的互补关系

### 8.1 MMLU-Pro 的优势

**广度：**
- 14 个学科，覆盖知识面广
- 246,343 题，数据量大
- 包含专业领域的知识（法律、医学、商业等）

**格式：**
- 选择题，评分标准化
- 适合大规模评估
- 适合评估知识广度

### 8.2 MATH 的优势

**深度：**
- 构造题，需要深度推理
- 可以评估推理质量
- 可以评估方法优雅度

**格式：**
- 构造题，允许自由发挥
- 适合评估推理深度
- 适合评估创新能力

### 8.3 互补关系

**MMLU-Pro：评估知识广度**
- 模型知道什么
- 模型知道多少
- 模型的知识覆盖面

**MATH：评估推理深度**
- 模型如何推理
- 模型的推理质量
- 模型的创新能力

**结论：** 两个数据集应该配合使用，而非替代关系

## 9. 未来扩展

### 9.1 自适应评分

根据模型的能力调整评分权重：
- 对于弱模型：提高"答案正确性"权重（如 50%）
- 对于强模型：提高"选项分析深度"权重（如 30%）

### 9.2 学科定制

根据不同学科的特点调整评分规则：
- **数学/物理**：更重视推理完整性
- **历史/文学**：更重视事实准确性
- **计算机科学**：更重视实践判断

### 9.3 动态难度调整

根据题目的难度调整评分标准：
- **简单题**：更重视答案正确性
- **难题**：更重视推理完整性

## 10. 结论

### 10.1 核心发现

1. **多选题的格式限制了评估深度**
   - 无法像构造题那样评估推理质量
   - 推理过程可能是"合理化"，而非真实推理

2. **但仍然有可改进的空间**
   - 推理完整性：鼓励模型提供推理过程
   - 选项分析深度：鼓励模型分析错误选项
   - 推理置信度：区分确定的理解和猜测

3. **成本可控**
   - 所有维度都可以用简单的正则匹配实现
   - 无需额外的 LLM 调用
   - 适合大规模评估

### 10.2 实施建议

**Phase 1（立即实施）：**
- 实现答案正确性 + 推理完整性
- 目标：鼓励模型提供推理过程

**Phase 2（逐步推进）：**
- 实现选项分析深度 + 推理置信度
- 目标：区分"真懂"和"蒙对"

**Phase 3（持续优化）：**
- 实现学科适应性
- 目标：根据学科特点调整评分标准

### 10.3 与 MATH 的配合

- **MMLU-Pro**：评估知识广度
- **MATH**：评估推理深度
- 两者配合，提供更全面的评估

### 10.4 最终建议

**不要对多维度评分期望过高**
- 多选题的格式限制了评估深度
- 有些维度（如推理正确性）在选择题中不可行
- 应该专注于可验证的维度

**但也不应该放弃改进**
- 即使是简单的改进，也比单纯的答案匹配好
- 推理完整性、选项分析深度等维度是有价值的
- 可以为模型改进提供更细粒度的反馈

**关键在于平衡**
- 在成本和效果之间找到平衡
- 在可行性和理想性之间找到平衡
- 在 MMLU-Pro 和 MATH 之间找到平衡
