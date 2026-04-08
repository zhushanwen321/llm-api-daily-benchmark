# P1: Evaluation Run 按 Provider 分组调度

## 问题

当前 `_run_multi_evaluation` 用 `asyncio.gather` 同时启动所有 evaluation run：
```python
coros = [
    _run_evaluation(model, dim, samples, debug)
    for model in models
    for dim in dimensions
]
await asyncio.gather(*coros)  # 24 个 run 同时启动
```

这导致同一 provider 下的任务全部同时争抢 semaphore：
- zai: 3 模型 x 4 维度 = 12 个 run，共 ~150 个 task 争抢 semaphore(2)
- opencode-go-gk: 2 模型 x 4 维度 = 8 个 run，共 ~100 个 task 争抢 semaphore(2)

队列深度过大，任务排队时间长，`execution_time` 指标被 semaphore 等待时间严重膨胀。

## 修复方案

### 方案：同 provider 的 evaluation run 串行，不同 provider 并行

```
修改前:
  asyncio.gather(zai-math, zai-code, zai-mmlu, zai-front, kimi-math, kimi-code, ...)
  → 所有 24 个 run 同时创建协程，争抢各自的 provider semaphore

修改后:
  provider_groups = {
    "zai":           [run(zai/glm-4.7, reasoning), run(zai/glm-4.7, backend-dev), ...],
    "kimi":          [run(kimi/kimi-for-coding, reasoning), ...],
    "opencode-go-gk":[run(opencode-go-gk/glm-5, reasoning), ...],
  }
  # 不同 provider 之间并行
  await asyncio.gather(
      _run_provider_group(provider_groups["zai"]),        # zai 内串行
      _run_provider_group(provider_groups["kimi"]),       # kimi 内串行
      _run_provider_group(provider_groups["opencode-go-gk"]),  # ogk 内串行
  )
```

### 为什么不在 run 内部串行 task

`_run_evaluation` 内部已经用 `asyncio.as_completed` 并发执行所有 task，
配合 semaphore(2) 限制 API 并发。问题不是单个 run 内的并发，而是多个 run
同时往同一个 semaphore 里灌任务。

### 改动文件

#### `benchmark/cli.py`

新增 `_run_provider_group()` 和 `_group_by_provider()` 函数，修改 `_run_multi_evaluation`。

### 关键实现

```python
def _group_by_provider(
    models: list[str], dimensions: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """按 provider 分组：(model, dimension) 列表。"""
    groups: dict[str, list[tuple[str, str]]] = {}
    for model in models:
        provider = model.split("/", 1)[0]
        for dim in dimensions:
            groups.setdefault(provider, []).append((model, dim))
    return groups


async def _run_provider_group(
    tasks: list[tuple[str, str]], samples: int, debug: bool
) -> None:
    """同一 provider 内的 evaluation run 串行执行。"""
    for model, dim in tasks:
        await _run_evaluation(model, dim, samples, debug)


async def _run_multi_evaluation(
    models: list[str], dimensions: list[str], samples: int, debug: bool
) -> None:
    groups = _group_by_provider(models, dimensions)
    coros = [
        _run_provider_group(tasks, samples, debug)
        for tasks in groups.values()
    ]
    await asyncio.gather(*coros)
```

### 效果分析

```
修改前 (zai semaphore 队列):
  T=0: 150 个 task 同时 acquire → 2 个执行, 148 个排队
  排队深度 75 → 平均等待 ~75 x 60s = 75 min

修改后 (zai 内串行, 12 个 run 依次执行):
  T=0:  run1 启动, 15 个 task acquire → 2 执行, 13 排队
  T=15: run1 完成, run2 启动, 15 个新 task
  排队深度 ~13 → 平均等待 ~13 x 60s = 13 min

  但 zai 和 kimi/opencode-go-gk 之间是并行的，不影响总 wall time
```

### 为什么这只是 P1

P0-01 和 P0-02 修复后，subprocess 不再阻塞事件循环，429 不再空占 semaphore。
此时即使 150 个 task 同时排队，有效并发也能接近 2.0，吞吐量不会严重下降。

P1 的收益是减少 semaphore 队列深度，改善 `execution_time` 指标的准确性，
以及减少极端情况下的 429 触发概率。

## 验证方式

1. 运行 `benchmark evaluate --model zai/glm-4.7,zai/glm-5,kimi/kimi-for-coding --dimension all --samples 5`
2. 检查日志：同一 provider 的 run 应串行完成，不同 provider 应并行
3. 检查 `execution_time`：不应包含大量 semaphore 等待时间
