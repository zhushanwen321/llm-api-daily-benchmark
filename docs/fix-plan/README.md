# 性能修复计划

## 问题摘要

调度器单次触发 6 模型 x 4 维度 = 24 个 evaluation run，理论耗时 ~75min，实际耗时 2.5h+。

根因：asyncio 单线程模型下，三个因素叠加形成正反馈循环，导致全局吞吐量降至理论值的 ~50%。

### 根因关系

```
根因 A (全局影响)          根因 B (provider 局部)       根因 C (调度层)
subprocess.run()          429 重试持有 semaphore       24 run 全量并发
冻结事件循环              浪费并发 slot                provider 间无隔离
影响所有 provider         影响 single provider         所有 provider 同时争抢
```

**A 是主要矛盾**：每次 subprocess 阻塞 5-30s，45 个 bigcodebench 任务累计冻结 675s。
期间所有 provider 的所有协程停转，包括 API streaming、semaphore 操作、DB 写入。

**B 放大 A**：429 重试期间 semaphore 被空占（单次可空占 150s），有效并发从 2 降到 1 甚至 0。

**C 放大 B**：zai 的 150 个任务同时争抢 semaphore(2)，队列深度 75，任何延迟都被放大。

### 验证结论

- Semaphore 按 provider 隔离：设计和实现一致（`AsyncConcurrencyLimiter._instances` 按 provider name 存储）
- `subprocess.run()` 阻塞是全局的：冻结整个事件循环，不区分 provider
- `_provider_async_limiters` 声明但从未使用（`_get_or_create_async_limiter` 直接委托 `AsyncConcurrencyLimiter.get_or_create`）
- `_provider_sync_semaphores` 和 `generate()` 方法仅被测试调用，生产代码未使用

### 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Scorer 异步接口 | 新增 `ascore()`，默认用 `asyncio.to_thread()` 包装 | 只有 ExecutionScorer 有真实异步需求，其余 3 个 scorer 是纯计算 <1ms |
| 死代码清理 | 本次一并清理 | `_provider_async_limiters`、`_provider_sync_semaphores`、`generate()` 均无生产调用 |
| sqlite3 写入 | 一并修复，用 `asyncio.to_thread()` 包装 | 单次 ~3ms 但高并发下文件锁争抢可导致排队 |

## 修复计划

按优先级和模块拆分，每个子文档包含：问题、修复方案、改动范围、验证方式。

| 优先级 | 文档 | 核心改动 | 影响范围 |
|--------|------|---------|---------|
| P0 | [01-event-loop-blocking.md](01-event-loop-blocking.md) | subprocess 改异步 | scorers/base.py, execution_scorer.py, cli.py |
| P0 | [02-semaphore-retry.md](02-semaphore-retry.md) | 重试前释放 semaphore | llm_adapter.py |
| P1 | [03-evaluation-grouping.md](03-evaluation-grouping.md) | 按 provider 分组串行 | cli.py |
| P1 | [04-db-async-and-cleanup.md](04-db-async-and-cleanup.md) | DB 异步写入 + 死代码清理 | database.py, llm_adapter.py |

## 执行顺序

1. P0-01：消除事件循环阻塞（最大性能瓶颈）
2. P0-02：重试释放 semaphore（减少 429 级联）
3. P1-03：按 provider 分组调度（减少排队深度）
4. P1-04：DB 异步写入 + 死代码清理
5. 全部完成后，重建容器运行完整评测，对比 GANTT 日志

## 预期效果

| 指标 | 修复前 | 修复后预期 |
|------|--------|-----------|
| zai 单次总耗时 | 2.5h+ | ~75min |
| 429 雪崩次数 | 5+ 次/run | 0-1 次/run |
| ReadError 失败 | 3-5 个/run | 0 个 |
| 有效并发 | ~1.0 | ~1.8 |
| 事件循环冻结总时间 | ~675s | 0s |
