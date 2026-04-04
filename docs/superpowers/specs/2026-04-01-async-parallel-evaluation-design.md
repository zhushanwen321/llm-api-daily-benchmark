# 异步并行评测设计

## 目标

将 benchmark evaluate 从串行执行改为 asyncio 并发执行，通过 per-provider Semaphore 控制大模型 API 并发数，非 API 调用部分（scorer、DB 写入）并行不受限。

## 核心决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 并发模型 | asyncio + httpx | 流式 SSE 解析天然适合 async for，单线程避免 DB 并发问题 |
| 限流机制 | per-provider asyncio.Semaphore | 语义从 QPS 变为最大并发数，复用 models.yaml rate_limit 字段 |
| 并行范围 | 跨 dimension 也支持 | CLI 可一次跑多个 dimension，共享 Semaphore |
| DB 写入 | 直接写，不加锁 | asyncio 单线程事件循环，同步 DB 写入不会互相打断 |
| 兼容性 | 保留同步 generate() | 不破坏现有接口 |

## 涉及文件

| 文件 | 变更 |
|------|------|
| `benchmark/core/llm_adapter.py` | 新增 `async def agenerate()`，用 httpx 异步流式请求，Semaphore 限流 |
| `benchmark/cli.py` | evaluate 命令内用 `asyncio.gather` 并发执行 task |
| `pyproject.toml` | 添加 httpx 依赖 |

不变更：rate_limiter.py、database.py、schemas.py、adapters/、scorers/、config.py。

## 详细设计

### 1. llm_adapter.py — agenerate()

在 LLMEvalAdapter 中新增：

```python
class LLMEvalAdapter:
    _provider_semaphores: dict[str, asyncio.Semaphore] = {}

    async def agenerate(self, prompt, model, temperature=0.0, max_tokens=None):
        cfg = self._get_model_config(model)
        sem = self._get_or_create_semaphore(model)
        async with sem:
            return await self._do_agenerate(cfg, prompt, model, temperature, max_tokens)
```

- `_get_or_create_semaphore` 根据 provider 从 models.yaml 读取 rate_limit 作为 Semaphore 容量
- `_do_agenerate` 包含完整的重试逻辑，与 generate() 一致，但用 httpx + asyncio.sleep
- SSE 解析用 `async for line in response.aiter_lines()`
- 保留 TTFT、tokens_per_second、流空闲超时等所有现有逻辑

### 2. cli.py — 并发执行

```
evaluate() → asyncio.run(_run_evaluation(...))
_run_evaluation():
  1. 初始化 adapter、scorer、llm、db
  2. 加载 tasks，创建 run 记录
  3. asyncio.gather(*[_evaluate_task(...) for task in tasks], return_exceptions=True)
  4. 汇总统计，更新 run 状态
```

每个 `_evaluate_task` 协程内部：
1. `await llm.agenerate(...)` — 受 Semaphore 限流
2. `scorer.score(...)` — 同步执行，不占 Semaphore
3. `db.save_result()` + `db.save_metrics()` — 直接写

进度显示用 Rich Live，每个 task 完成时刷新。

### 3. 错误处理

- `asyncio.gather(..., return_exceptions=True)`：单 task 失败不影响其他 task
- 失败 task 在结果中标记 failed，计入统计
- 全部 task 完成后统一更新 run 状态

### 4. rate_limit 语义变更

models.yaml 中 `rate_limit` 含义从"每秒最多 N 个请求"变为"最大并发 API 请求数"。

例如 `rate_limit: 2` 表示该 provider 最多 2 个并发 API 请求在途。

### 5. httpx 依赖

添加 httpx 到项目依赖。httpx 的流式 API 与 requests 类似，迁移成本低。
