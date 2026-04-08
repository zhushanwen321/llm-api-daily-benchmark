# 性能修复设计规格

日期: 2026-04-06
分支: fix/perf-and-429-investigation → fix/perf-async-refactor

## 背景

调度器单次触发 6 模型 x 4 维度 = 24 个 evaluation run，理论耗时 ~75min，实际 2.5h+。
根因是 asyncio 单线程模型下三个因素叠加：subprocess 阻塞事件循环、429 重试空占 semaphore、
全量并发导致队列过深。详细调查见 `docs/benchmark-perf-investigation.md`。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Scorer 异步接口 | 新增 `ascore()`，默认 `asyncio.to_thread()` | 只有 ExecutionScorer 有真实异步需求，其余 3 个纯计算 <1ms |
| 死代码 | 本次一并清理 | `_provider_async_limiters` 从未使用，`generate()` 仅测试调用 |
| DB 写入 | 用 `asyncio.to_thread()` 包装 | 改动最小，不需要引入 aiosqlite |

## 修复项

| 优先级 | 文档 | 核心改动 | 改动文件 |
|--------|------|---------|---------|
| P0 | [01-event-loop-blocking](../fix-plan/01-event-loop-blocking.md) | subprocess → `asyncio.create_subprocess_exec()` | scorers/base.py, execution_scorer.py, cli.py |
| P0 | [02-semaphore-retry](../fix-plan/02-semaphore-retry.md) | 重试前释放 semaphore | llm_adapter.py |
| P1 | [03-evaluation-grouping](../fix-plan/03-evaluation-grouping.md) | 按 provider 分组串行 | cli.py |
| P1 | [04-db-async-and-cleanup](../fix-plan/04-db-async-and-cleanup.md) | DB 异步写入 + 删除死代码 | database.py, llm_adapter.py, cli.py |

## 接口变更

### BaseScorer 新增 `ascore()`

```python
class BaseScorer(ABC):
    @abstractmethod
    def score(self, ctx: ScoringContext) -> ScoreResult: ...

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        return await asyncio.to_thread(self.score, ctx)
```

- MathScorer / ChoiceMatchScorer / KeywordMatchScorer：不改动，继承默认实现
- ExecutionScorer：重写 `ascore()`，用原生 async subprocess

### Database 新增 `asave_result()` / `asave_metrics()`

```python
class Database:
    async def asave_result(self, result: EvalResult) -> str:
        return await asyncio.to_thread(self.save_result, result)

    async def asave_metrics(self, metrics: ApiCallMetrics) -> str:
        return await asyncio.to_thread(self.save_metrics, metrics)
```

### LLMEvalAdapter.agenerate() 重构

semaphore 粒度从"整个重试周期"变为"单次 API 请求"：
重试 sleep 前释放 semaphore，sleep 后重新 acquire。

### _run_multi_evaluation() 按 provider 分组

同 provider 的 evaluation run 串行，不同 provider 并行。

## 删除清单

| 删除项 | 文件 | 行号 |
|--------|------|------|
| `_provider_async_limiters` | llm_adapter.py | ~43 |
| `_provider_sync_semaphores` | llm_adapter.py | ~41 |
| `_get_or_create_sync_semaphore()` | llm_adapter.py | ~57-66 |
| `generate()` 方法 | llm_adapter.py | ~83-470 |
| 相关 sync 测试 | tests/test_llm_adapter.py | 待定 |

## 验证计划

1. `pytest tests/` 全量通过
2. 单维度冒烟测试：`benchmark evaluate --model zai/glm-5 --dimension backend-dev --samples 5`
3. 全量评测（调度器触发），对比 GANTT 日志
4. 预期：总耗时从 2.5h+ 降至 ~75min，429 雪崩消除，ReadError 消除

## 不在本次范围

- aiosqlite 替换 sqlite3（`asyncio.to_thread` 已足够）
- scheduler 重构（当前只改调度策略，不改 APScheduler）
- 前端/Streamlit 展示优化
