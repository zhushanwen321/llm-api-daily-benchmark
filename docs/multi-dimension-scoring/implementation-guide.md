# 多维度推理评分实现指南

## 1. 架构设计

### 1.1 新的 Scorer 类

```python
# benchmark/scorers/math_reasoning_scorer.py

class MathReasoningScorer(BaseScorer):
    """MATH 数据集的多维度推理评分器."""

    def __init__(self, config: ScoringConfig):
        self.config = config
        self.answer_scorer = MathScorer()  # 复用现有的答案评分
        self.judge_client = None  # 延迟初始化

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """同步评分入口."""
        return asyncio.run(self.ascore(ctx))

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """异步评分入口（推荐使用）."""

        # 并行计算各个维度
        results = await asyncio.gather(
            self._score_answer_correctness(ctx),
            self._score_reasoning_completeness(ctx),
            self._score_reasoning_validity(ctx),  # 可能需要 LLM judge
            self._score_method_elegance(ctx),
            self._score_difficulty_adaptation(ctx),
            return_exceptions=True
        )

        # 处理异常
        dimension_scores = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Dimension {i} failed: {result}")
                dimension_scores[i] = 0.0
            else:
                dimension_scores[i] = result

        # 加权汇总
        final_score = self._combine_scores(dimension_scores)

        return ScoreResult(
            score=final_score,
            passed=final_score >= 60.0,  # 及格线
            details={
                "dimension_scores": dimension_scores,
                "weights": self.config.weights,
            },
            reasoning=self._generate_reasoning(dimension_scores)
        )
```

### 1.2 配置管理

```python
# benchmark/models/schemas.py

@dataclass
class ScoringConfig:
    """多维度评分配置."""

    weights: dict[str, float] = field(default_factory=lambda: {
        "answer_correctness": 0.40,
        "reasoning_completeness": 0.25,
        "reasoning_validity": 0.20,
        "method_elegance": 0.10,
        "difficulty_adaptation": 0.05,
    })

    # 推理完整性阈值
    min_reasoning_tokens: dict[int, int] = field(default_factory=lambda: {
        3: 200, 4: 400, 5: 600
    })

    # 方法优雅度关键词
    elegance_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "Algebra": ["因式分解", "对称性", "换元", "韦达定理"],
        "Geometry": ["辅助线", "相似", "勾股定理"],
        "Number Theory": ["模运算", "整除", "同余"],
    })

    # Judge 模型配置
    judge_model: str = "claude-3.5-sonnet"  # 或 gpt-4o
    judge_cache_enabled: bool = True
    judge_batch_size: int = 10
```

## 2. 各维度实现

### 2.1 答案正确性（复用现有）

```python
async def _score_answer_correctness(self, ctx: ScoringContext) -> float:
    """复用现有的 MathScorer."""
    result = self.answer_scorer.score(ctx)
    return result.score  # 0.0 or 100.0
```

### 2.2 推理完整性

```python
async def _score_reasoning_completeness(self, ctx: ScoringContext) -> float:
    """评分推理完整性."""
    reasoning = ctx.reasoning_content
    if not reasoning:
        return 0.0

    level = ctx.task.metadata.get("level", 3)

    # 检查 1：长度检查
    reasoning_tokens = len(reasoning.split())  # 粗略估计
    min_tokens = self.config.min_reasoning_tokens.get(level, 200)
    length_score = min(100, reasoning_tokens / min_tokens * 40)

    # 检查 2：逻辑连接词
    transitions = ["因为", "所以", "因此", "由于", "故", "从而", "于是"]
    transition_count = sum(1 for t in transitions if t in reasoning)
    transition_score = min(100, transition_count / 3 * 30)

    # 检查 3：结构化程度
    has_structure = bool(re.search(
        r'(步骤|Step|首先|其次|最后|第一|第二|综上)',
        reasoning
    ))
    structure_score = 30 if has_structure else 0

    return length_score + transition_score + structure_score
```

### 2.3 推理正确性（LLM-as-a-Judge）

```python
async def _score_reasoning_validity(self, ctx: ScoringContext) -> float:
    """用 LLM Judge 评分推理正确性."""

    # 构建评判 prompt
    judge_prompt = self._build_judge_prompt(ctx)

    # 检查缓存
    cache_key = self._get_cache_key(ctx)
    if self.config.judge_cache_enabled:
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

    # 调用 Judge 模型
    judge_response = await self._call_judge_model(judge_prompt)

    # 解析分数
    score = self._parse_judge_score(judge_response)

    # 保存缓存
    if self.config.judge_cache_enabled:
        self._save_to_cache(cache_key, score)

    return score

def _build_judge_prompt(self, ctx: ScoringContext) -> str:
    """构建 Judge prompt."""
    return f"""你是数学推理质量评判专家。请检查以下解题过程的正确性。

【题目】
{ctx.task.prompt}

【模型推理过程】
{ctx.reasoning_content}

【模型答案】
{ctx.model_answer}

【标准答案】
{ctx.expected}

评分标准（总分 100）：
1. 逻辑一致性（40分）：推理步骤是否自洽，是否存在矛盾
2. 数学事实正确性（40分）：使用的定理、公式、性质是否正确
3. 计算正确性（20分）：关键计算步骤是否正确

请只返回一个 0-100 之间的数字，不要其他内容。"""
```

### 2.4 方法优雅度

```python
async def _score_method_elegance(self, ctx: ScoringContext) -> float:
    """评分方法优雅度."""
    reasoning = ctx.reasoning_content
    subject = ctx.task.metadata.get("subject", "")
    level = ctx.task.metadata.get("level", 3)

    # 基础分
    score = 50.0

    # 奖励：使用技巧
    keywords = self.config.elegance_keywords.get(subject, [])
    keyword_count = sum(1 for kw in keywords if kw in reasoning)
    score += min(keyword_count * 10, 30)

    # 惩罚：过度冗长
    reasoning_tokens = len(reasoning.split())
    max_tokens = level * 300
    if reasoning_tokens > max_tokens:
        excess_ratio = (reasoning_tokens - max_tokens) / max_tokens
        score -= min(excess_ratio * 20, 30)

    return max(0, min(100, score))
```

### 2.5 难度适应性

```python
async def _score_difficulty_adaptation(self, ctx: ScoringContext) -> float:
    """评分难度适应性."""
    reasoning = ctx.reasoning_content
    level = ctx.task.metadata.get("level", 3)

    # 估算推理深度（句子数量）
    sentences = re.split(r'[。！？.!?]', reasoning)
    sentences = [s.strip() for s in sentences if s.strip()]
    actual_depth = len(sentences)

    # 期望深度
    expected_depth = {3: 5, 4: 8, 5: 12}
    expected = expected_depth.get(level, 5)

    # 匹配度
    depth_match = 1 - abs(actual_depth - expected) / expected
    return depth_match * 100
```

## 3. 性能优化

### 3.1 缓存机制

```python
import hashlib
import json
from pathlib import Path

class JudgeCache:
    """Judge 结果缓存."""

    def __init__(self, cache_dir: str = "cache/judge"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, ctx: ScoringContext) -> str:
        """生成缓存 key."""
        content = f"{ctx.task.prompt}{ctx.reasoning_content}{ctx.model_answer}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, cache_key: str) -> float | None:
        """读取缓存."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return data.get("score")
        return None

    def set(self, cache_key: str, score: float):
        """写入缓存."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {"score": score, "timestamp": time.time()}
        cache_file.write_text(json.dumps(data))
```

### 3.2 批量 Judge

```python
class BatchJudgeClient:
    """批量 Judge 客户端."""

    def __init__(self, model: str, batch_size: int = 10):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
        self.results = {}

    async def judge(self, ctx: ScoringContext) -> float:
        """加入队列，等待批量处理."""
        cache_key = self._get_cache_key(ctx)

        # 检查缓存
        if cache_key in self.results:
            return self.results[cache_key]

        # 加入队列
        future = asyncio.Future()
        self.queue.append((ctx, future))

        # 队列满了，触发批量处理
        if len(self.queue) >= self.batch_size:
            await self._process_batch()

        return await future

    async def _process_batch(self):
        """批量处理."""
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]

        # 构建 batch prompt
        prompt = self._build_batch_prompt([ctx for ctx, _ in batch])

        # 调用模型
        response = await self._call_model(prompt)

        # 解析结果
        scores = self._parse_batch_response(response)

        # 设置 Future
        for (ctx, future), score in zip(batch, scores):
            cache_key = self._get_cache_key(ctx)
            self.results[cache_key] = score
            future.set_result(score)
```

### 3.3 异步并发

```python
async def score_batch(
    self,
    contexts: list[ScoringContext]
) -> list[ScoreResult]:
    """批量评分（并发处理）."""

    # 限制并发数
    semaphore = asyncio.Semaphore(10)

    async def score_with_semaphore(ctx):
        async with semaphore:
            return await self.ascore(ctx)

    # 并发评分
    tasks = [score_with_semaphore(ctx) for ctx in contexts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {contexts[i].task.task_id} failed: {result}")
            final_results.append(ScoreResult(
                score=0.0,
                passed=False,
                details={"error": str(result)},
                reasoning="Scoring failed"
            ))
        else:
            final_results.append(result)

    return final_results
```

## 4. 集成到现有系统

### 4.1 修改 MATHAdapter

```python
# benchmark/adapters/math_adapter.py

from benchmark.scorers.math_reasoning_scorer import MathReasoningScorer

class MATHAdapter(DatasetAdapter):
    # ... 现有代码 ...

    def get_scorer(self) -> BaseScorer:
        """返回推荐使用的 Scorer."""
        return MathReasoningScorer(
            config=ScoringConfig(
                weights={
                    "answer_correctness": 0.40,
                    "reasoning_completeness": 0.25,
                    "reasoning_validity": 0.20,
                    "method_elegance": 0.10,
                    "difficulty_adaptation": 0.05,
                }
            )
        )
```

### 4.2 CLI 集成

```python
# benchmark/cli.py

@app.command()
def evaluate(
    model: str,
    dataset: str = "math",
    scorer: str = "reasoning",  # 新增：reasoning 或 simple
):
    """运行评测."""

    # 选择 Scorer
    if dataset == "math":
        if scorer == "reasoning":
            from benchmark.scorers.math_reasoning_scorer import MathReasoningScorer
            scorer_cls = MathReasoningScorer
        else:
            from benchmark.scorers.math_scorer import MathScorer
            scorer_cls = MathScorer

    # 运行评测
    # ...
```

## 5. 测试策略

### 5.1 单元测试

```python
# tests/test_math_reasoning_scorer.py

def test_answer_correctness():
    """测试答案正确性评分."""
    scorer = MathReasoningScorer()
    ctx = ScoringContext(
        model_answer="\\frac{14}{3}",
        expected="\\frac{14}{3}",
        # ...
    )
    result = asyncio.run(scorer._score_answer_correctness(ctx))
    assert result == 100.0

def test_reasoning_completeness():
    """测试推理完整性评分."""
    scorer = MathReasoningScorer()

    # 完整推理
    ctx_complete = ScoringContext(
        reasoning_content="首先，我们分析... 因此... 故答案为...",
        task=TaskDefinition(metadata={"level": 4}),
        # ...
    )
    score = asyncio.run(scorer._score_reasoning_completeness(ctx_complete))
    assert score > 70

    # 不完整推理
    ctx_incomplete = ScoringContext(
        reasoning_content="答案是 42",
        # ...
    )
    score = asyncio.run(scorer._score_reasoning_completeness(ctx_incomplete))
    assert score < 30
```

### 5.2 集成测试

```python
def test_end_to_end_scoring():
    """端到端测试."""
    adapter = MATHAdapter()
    tasks = adapter.load()
    scorer = MathReasoningScorer()

    for task in tasks[:3]:  # 测试前 3 题
        ctx = ScoringContext(
            model_answer=task.expected_output,  # 假设答案正确
            expected=task.expected_output,
            reasoning_content="详细的推理过程...",
            task=task,
        )
        result = scorer.score(ctx)
        assert 0 <= result.score <= 100
        assert "dimension_scores" in result.details
```

## 6. 监控和调试

### 6.1 日志记录

```python
import logging

logger = logging.getLogger(__name__)

async def ascore(self, ctx: ScoringContext) -> ScoreResult:
    logger.info(f"Scoring task {ctx.task.task_id}")
    logger.debug(f"Reasoning tokens: {len(ctx.reasoning_content.split())}")

    start_time = time.time()
    result = await self._score_internal(ctx)
    duration = time.time() - start_time

    logger.info(
        f"Scored {ctx.task.task_id}: {result.score:.1f} "
        f"in {duration:.2f}s"
    )

    return result
```

### 6.2 指标收集

```python
from prometheus_client import Histogram, Counter

# 评分耗时
scoring_duration = Histogram(
    'math_reasoning_scoring_duration_seconds',
    'Time spent scoring',
    ['dimension']
)

# Judge 调用次数
judge_calls = Counter(
    'math_reasoning_judge_calls_total',
    'Total judge model calls',
    ['cached']  # cached="true" or "false"
)

# 使用
async def _score_reasoning_validity(self, ctx: ScoringContext):
    with scoring_duration.labels('reasoning_validity').time():
        # ...
        if not cached:
            judge_calls.labels(cached='false').inc()
```

## 7. 渐进式部署

### 7.1 A/B 测试

```python
class ABTestScorer(BaseScorer):
    """A/B 测试 Scorer."""

    def __init__(self, control_scorer, treatment_scorer, treatment_ratio=0.5):
        self.control = control_scorer
        self.treatment = treatment_scorer
        self.treatment_ratio = treatment_ratio

    def score(self, ctx: ScoringContext) -> ScoreResult:
        # 根据 task_id 决定用哪个 Scorer
        hash_val = int(hashlib.md5(ctx.task.task_id.encode()).hexdigest(), 16)
        if hash_val % 100 < self.treatment_ratio * 100:
            result = self.treatment.score(ctx)
            result.details["ab_group"] = "treatment"
        else:
            result = self.control.score(ctx)
            result.details["ab_group"] = "control"
        return result
```

### 7.2 灰度发布

```python
# 阶段 1：10% 流量使用新 Scorer
scorer = ABTestScorer(
    control_scorer=MathScorer(),
    treatment_scorer=MathReasoningScorer(),
    treatment_ratio=0.1
)

# 阶段 2：50% 流量
scorer = ABTestScorer(
    control_scorer=MathScorer(),
    treatment_scorer=MathReasoningScorer(),
    treatment_ratio=0.5
)

# 阶段 3：100% 流量
scorer = MathReasoningScorer()
```

## 8. 成本估算

### 8.1 Token 消耗

```
单题 Judge 估算：
- Prompt: ~400 tokens（题目 + 推理 + 答案）
- Response: ~100 tokens（分数 + 简短理由）
- 总计: ~500 tokens

15 题一次评测：
- 总 tokens: 15 × 500 = 7,500 tokens

成本（Claude 3.5 Sonnet）：
- Input: 6,750 × $3/1M = $0.02
- Output: 750 × $15/1M = $0.01
- 总计: ~$0.03/次

优化后（缓存 + 批量）：
- 重复题: $0（缓存）
- 新题: $0.03
- 平均: $0.015/次
```

### 8.2 性能影响

```
现有 MathScorer：~10ms/题
新 MathReasoningScorer：
- 答案正确性：10ms
- 推理完整性：5ms
- 推理正确性：2-5s（Judge 模型）
- 方法优雅度：5ms
- 难度适应性：5ms
- 总计：~2-5s/题

优化后：
- 缓存命中：~30ms
- 缓存未命中：~500ms（批量 Judge）
- 平均：~100ms/题
```
