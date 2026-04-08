# P0-02: 重试期间释放 Semaphore

## 问题

`agenerate()` 在 `acquire()` 后，整个 `_do_agenerate()`（含重试循环）完成后才 `release()`。
重试的 `await asyncio.sleep(backoff)` 期间 semaphore 被持有但不做有用功。

**当前代码** (`benchmark/core/llm_adapter.py:488-501`):
```python
async def agenerate(self, ...):
    limiter = self._get_or_create_async_limiter(model)
    if limiter is not None:
        await limiter.acquire()              # 获取 semaphore
        try:
            return await self._do_agenerate(...)  # 含重试循环，429 时 sleep 10/20/40/80s
        finally:
            limiter.release()                # 整个 _do_agenerate 完成才释放
```

**日志证据**:
```
15:40:57  zai/glm-5  429 attempt 1, sleep 10s   ← semaphore 空占
15:44:11  zai/glm-5  429 attempt 2, sleep 20s   ← 继续空占
15:44:32  zai/glm-5  429 attempt 3, sleep 40s   ← 继续空占
15:45:12  zai/glm-5  429 attempt 4, sleep 80s   ← 继续空占
15:46:33  zai/glm-5  429 重试耗尽               ← 150s 后才释放 semaphore
```

**影响**: 单次 429 雪崩浪费 150s semaphore 时间，有效并发从 2 降到 1。

## 修复方案

### 方案：每次 API 请求独立 acquire/release，重试间隔不持有 semaphore

将 semaphore 的粒度从"整个 agenerate 调用"细化到"单次 API 请求"。

### 改动文件

#### `benchmark/core/llm_adapter.py`

重构 `agenerate()` 和 `_do_agenerate()` 的职责划分：

```
修改前:
  agenerate() → acquire → _do_agenerate(含重试循环) → release

修改后:
  agenerate() → 重试循环 {
    acquire → 单次 _do_api_request → release
    if 失败且可重试: sleep(backoff)  ← 此时已释放 semaphore
  }
```

### 关键实现

```python
async def agenerate(self, prompt, model, temperature=0.0, max_tokens=None):
    limiter = self._get_or_create_async_limiter(model)
    cfg = self._get_model_config(model)
    # ... 构建 payload ...

    last_error = None
    for attempt in range(self.max_retries):
        # 每次 attempt 独立获取/释放 semaphore
        if limiter is not None:
            await limiter.acquire()
        try:
            result = await self._do_single_request(cfg, payload, model)
            return result  # 成功则直接返回
        except RetryableError as exc:
            last_error = exc
            # semaphore 已在 finally 中释放
            if attempt < self.max_retries - 1:
                wait = calc_backoff(exc, attempt)
                logger.warning(f"[{model}] attempt {attempt+1} 失败, {wait}s 后重试...")
                await asyncio.sleep(wait)  # 此时 semaphore 已释放，其他任务可以使用
        except NonRetryableError:
            raise
        finally:
            if limiter is not None:
                limiter.release()

    raise ConnectionError(f"重试 {self.max_retries} 次后仍失败: {last_error}")
```

### 注意事项

1. **429 的 backoff 策略不变**：仍使用指数退避（10/20/40/80s）
2. **semaphore 语义不变**：仍限制同时进行的 API 请求为 max_concurrency
3. **仅改变 semaphore 持有时机**：从"整个重试周期"变为"单次请求"

### 并发模型对比

```
修改前 (150 tasks, semaphore=2):
  Slot 1: [req1][sleep 10s][retry][sleep 20s][retry][sleep 40s]... ← 150s 空转
  Slot 2: [req2][done][req3][done]...
  等效并发: ~1.0 (大量空转)

修改后:
  Slot 1: [req1][fail][release] → [req5][done] → [req8][done]...
  Slot 2: [req2][done] → [req3][done] → [req4][done]...
  req1 在 sleep 期间不占 slot，其他任务可以使用
  等效并发: ~1.8 (接近理论值 2)
```

## 验证方式

1. 模拟 429 场景：确认 sleep 期间其他任务能获取 semaphore
2. 检查 PERF 日志：`semaphore_wait` 应显著降低
3. 检查 GANTT 日志：不再出现单个任务耗时 > 200s 的情况
