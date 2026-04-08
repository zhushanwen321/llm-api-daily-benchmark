# P1-04: DB 异步写入 + 死代码清理

## 问题

### DB 写入阻塞事件循环

`database.py` 的 `save_result()` / `save_metrics()` 使用同步 sqlite3 操作，
在 `_evaluate_task`（async 上下文）中直接调用。

**当前代码** (`benchmark/models/database.py`):
```python
def save_result(self, result: EvalResult) -> str:
    conn = self._get_conn()
    conn.execute("INSERT INTO eval_results ...", (...))
    conn.commit()
    return result.result_id
```

**影响量化**:
- 单次写入 ~3ms
- 150 个并发任务争抢 sqlite 文件锁，高并发下排队可达 10-50ms
- 不如 subprocess 严重，但仍是不必要的事件循环阻塞

### 死代码

| 变量/方法 | 位置 | 原因 |
|-----------|------|------|
| `_provider_async_limiters` | `llm_adapter.py:43` | 声明但从未写入/读取，`_get_or_create_async_limiter` 直接调用 `AsyncConcurrencyLimiter.get_or_create()` |
| `_provider_sync_semaphores` | `llm_adapter.py:41` | 仅被 `generate()` 使用，而 `generate()` 仅测试调用 |
| `generate()` 方法 | `llm_adapter.py:83-470` | 生产代码全部使用 `agenerate()`，`generate()` 仅测试调用 |

## 修复方案

### DB 异步写入

在 `Database` 新增 `async asave_result()` / `async asave_metrics()`，
内部用 `asyncio.to_thread()` 包装同步操作。

**选择 `asyncio.to_thread()` 而非 aiosqlite 的原因**:
- `Database` 类已封装完整的 SQL 逻辑，改用 aiosqlite 需要重写连接管理
- `asyncio.to_thread()` 对现有代码改动最小（~10 行）
- sqlite3 的阻塞在毫秒级，`to_thread` 足以消除事件循环冻结

### 死代码清理

直接删除三个死代码块。

### 改动文件

#### 1. `benchmark/models/database.py`
- 新增 `async asave_result()` 和 `async asave_metrics()`
- 保留同步方法供测试使用

#### 2. `benchmark/cli.py` (`_evaluate_task`)
- `db.save_result(result)` → `await db.asave_result(result)`
- `db.save_metrics(metrics)` → `await db.asave_metrics(metrics)`

#### 3. `benchmark/core/llm_adapter.py`
- 删除 `_provider_async_limiters`（第 43 行）
- 删除 `_provider_sync_semaphores`（第 41 行）
- 删除 `_get_or_create_sync_semaphore()`（第 57-66 行）
- 删除整个 `generate()` 方法（第 83-470 行）
- 删除 `_provider_async_limiters` 相关的 import（如果清理后无使用）

### 关键实现

```python
# database.py - 新增异步方法
class Database:
    async def asave_result(self, result: EvalResult) -> str:
        return await asyncio.to_thread(self.save_result, result)

    async def asave_metrics(self, metrics: ApiCallMetrics) -> str:
        return await asyncio.to_thread(self.save_metrics, metrics)
```

### 对测试的影响

- `generate()` 删除后，`tests/test_llm_adapter.py` 中的 `test_sync_generate` 等测试需要删除或改为测试 `agenerate()`
- 其余测试（scorer 测试等）不受影响，它们仍调用同步 `score()`

## 验证方式

1. `pytest tests/` 确认所有测试通过（除已删除的 sync generate 测试）
2. 运行评测，检查 DB 写入无报错
3. 确认 `_provider_async_limiters` / `_provider_sync_semaphores` 无残留引用
