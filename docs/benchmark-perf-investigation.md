# Benchmark 性能与问题调查记录

日期: 2026-04-06
分支: fix/perf-and-429-investigation

## 运行概况

调度器配置:
- cron: `0 0,4,8,10,14,18,22 * * *` (每天8次)
- models: `zai/glm-4.7`, `zai/glm-5`, `zai/glm-5.1`, `kimi/kimi-for-coding`, `opencode-go-gk/glm-5`, `opencode-go-gk/kimi-k2.5`
- dimensions: all (reasoning, backend-dev, system-architecture, frontend-dev)
- samples: 15

单次调度总任务量:
- 6 模型 x 4 维度 = 24 个 evaluation run
- zai provider: 3 模型 x (15+15+15+5) = **150 个任务** -> 共享 semaphore(2)
- kimi provider: 1 模型 x 50 = **50 个任务** -> 共享 semaphore(2)
- opencode-go-gk provider: 2 模型 x 50 = **100 个任务** -> 共享 semaphore(2)

---

## 发现的问题

### P0 - 致命/严重影响

#### 问题1: 24个 evaluation run 全量并发，信号量队列过深

**现象:**
- 14:00:00 触发，24 个 `_run_evaluation` 通过 `asyncio.gather` 同时启动
- 所有任务的协程同时创建并争抢 semaphore
- zai provider 的 150 个任务争抢 semaphore(2)，平均排队 75 轮

**影响:**
- zai 总执行时间: 2.5 小时+ (预期 ~1.5 小时)
- 任务 `execution_time` 包含大量信号量等待时间，不具有参考价值

**根因:**
- `_run_multi_evaluation` 用 `asyncio.gather` 同时启动所有 evaluation
- 每个 `_run_evaluation` 内部又用 `asyncio.as_completed` 同时启动所有 task
- 两层并发叠加，导致同一 provider 下的任务数 = 模型数 x 维度任务数

**状态:** 待修复

---

#### 问题2: 重试期间持有信号量，造成 429 雪崩

**现象:**
```
15:35:25 - zai/glm-5 429 (attempt 1, sleep 10s, 持有 semaphore)
15:35:37 - 重试成功
15:40:57 - zai/glm-5 429 (attempt 1, sleep 10s)
15:44:11 - 429 (attempt 2, sleep 20s)
15:44:32 - 429 (attempt 3, sleep 40s)
15:45:12 - 429 (attempt 4, sleep 80s)
15:46:33 - 429 重试 5 次后失败!
```

**影响:**
- 重试退避期间 (10+20+40+80=150s) semaphore 被占用但不做有用功
- 其他任务无法获取 semaphore，吞吐量急剧下降
- 429 和信号量持有形成正反馈：越限流 -> 越多重试 -> semaphore 占用越久 -> 吞吐越低

**根因:**
- `agenerate()` 在 `acquire()` 后整个 `_do_agenerate()`（含重试循环）完成后才 `release()`
- 重试的 `await asyncio.sleep(backoff)` 期间 semaphore 被持有

**kimi 同样受影响:**
```
14:25:39 - kimi/kimi-for-coding 429 重试5次后失败 (mmlu_pro_7896)
14:26:45 - kimi/kimi-for-coding 429 重试5次后失败 (mmlu_pro_7972)
...
```

**状态:** 待修复

---

#### 问题3: ExecutionScorer.subprocess.run() 阻塞 asyncio 事件循环

**现象:**
- BigCodeBench 维度 (backend-dev) 使用 `ExecutionScorer` 评分
- `ExecutionScorer.score()` 调用 `subprocess.run()` 同步执行代码，最长阻塞 30 秒
- 这在 async 上下文中会阻塞整个事件循环

**影响:**
- 事件循环阻塞期间：
  - 所有协程暂停（包括正在流式接收的 API 响应）
  - semaphore 的 release/acquire 无法处理
  - 可能导致流超时 (ReadError / Server disconnected)
- zai 下有 3 模型 x 15 = 45 个 bigcodebench 任务
- 每个 subprocess.run() 阻塞 ~5-30s，总计 225-1350s 的事件循环阻塞

**根因:**
- `scorer.score()` 是同步方法，在 `async def _evaluate_task()` 中直接调用
- `subprocess.run()` 完全同步，不释放 GIL 给 asyncio

**这可能是 ReadError 的间接原因:**
```
15:33:01 - math_test/counting_and_probability/870.json 失败: ReadError
15:33:29 - bigcodebench_hard_12 失败: Server disconnected without sending a response
```
当一个 bigcodebench 任务在 subprocess.run() 中阻塞事件循环时，其他正在接收流式响应的任务可能因缓冲区满而断连。

**状态:** 待修复（优先级高）

---

### P1 - 中等影响

#### 问题4: ReadError / Server disconnected 导致任务失败但未充分重试

**现象:**
```
15:33:01 - math_test/counting_and_probability/870.json | ERROR: ReadError:
15:33:29 - bigcodebench_hard_12 | ERROR: RemoteProtocolError: Server disconnected
15:45:18 - bigcodebench_hard_11 | ERROR: ReadError
```

**影响:**
- 多个任务因网络层错误失败，降低评测完整性
- 可能是问题3（事件循环阻塞）的间接后果

**根因:**
- `_do_agenerate` 的重试循环应捕获 `httpx.StreamError`（含 ReadError）
- 但 `RemoteProtocolError` 可能在异常层级上未被完全覆盖
- 需要确认具体异常传播路径

**状态:** 待验证

---

#### 问题5: kimi/kimi-for-coding 大量 429 失败

**现象:**
```
14:25:39 - kimi/kimi-for-coding 429 重试5次后失败 (mmlu_pro_7896)
14:26:45 - kimi/kimi-for-coding 429 重试5次后失败 (mmlu_pro_7972)
... (连续 7+ 个任务因 429 失败)
```

**影响:**
- kimi 的 system-architecture 维度大面积失败
- 问题2的重试占用 semaphore + kimi API 的限流更严格

**根因:** 同问题2，信号量持有 + API 限流

**状态:** 待修复

---

### P2 - 低影响

#### 问题6: TTFT-C 异常高

**现象:**
```
mmlu_pro_9109 | TTFT-C: 493.80s (8.2 分钟)
mmlu_pro_9109 | TTFT-C: 664.86s (11.1 分钟)
mmlu_pro_8190 | TTFT-C: 405.56s (6.8 分钟)
mmlu_pro_9222 | TTFT-C: 281.75s (4.7 分钟)
```

**影响:** 这些任务的 TTFT-C 远超正常值（通常 10-60s），说明 API 侧延迟严重

**根因:**
- 可能是 API 侧排队（服务端推理时间包含在 TTFT-C 中）
- 也可能是事件循环阻塞导致首 content token 时间计算被延迟

**状态:** 待验证（需计时日志区分排队延迟 vs API 延迟）

---

#### 问题7: kimi-k2.5 偶发空内容

**现象:**
```
[opencode-go-gk/kimi-k2.5] 内容异常: token 计数非零但内容为空 (chunks=6857, completion_tokens=4140), 可能被安全过滤
[opencode-go-gk/kimi-k2.5] 内容异常: 模型返回空内容 (chunks=1743, completion_tokens=0)
```

**影响:** 个别任务需重试，但不严重

**状态:** 已有重试机制，低优先级

---

#### 问题8: zai/glm-5 偶发 500 Internal Server Error

**现象:**
```
15:55:39 - zai/glm-5 500 Internal Server Error (attempt 1, 2s 后重试)
16:04:32 - zai/glm-5 500 Internal Server Error (attempt 1, 2s 后重试)
```

**影响:** 重试后成功，影响有限

**状态:** 已有重试机制，低优先级

---

## 根因分析摘要

### 核心瓶颈: 三因素叠加

```
                        ┌──────────────────────┐
                        │  24 runs 全量并发     │
                        │  zai: 150 tasks      │
                        │  semaphore: 2        │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  semaphore 队列过深           │
                    │  平均排队 75 轮 x 75s/轮      │
                    │  = ~1.5h 理论最小时间         │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  + 429 重试占用 semaphore               │
              │  每次重试 sleep 期间 semaphore 空转      │
              │  实际吞吐 < 2 并发                      │
              └────────────────────┬────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │  + subprocess.run() 阻塞事件循环            │
          │  45 个 bigcodebench 评分各阻塞 5-30s        │
          │  期间所有协程暂停，streaming 中断            │
          └─────────────────────────────────────────────┘
                                 │
                                 ▼
                    实际运行时间: 2.5h+ (理论 1.5h)
```

### 修复优先级

| 优先级 | 问题 | 修复方向 |
|--------|------|----------|
| P0-1 | subprocess.run() 阻塞 | 改用 `asyncio.create_subprocess_exec()` |
| P0-2 | 重试占用 semaphore | 重试前释放 semaphore，重试后重新获取 |
| P0-3 | 全量并发 | 按 provider 分组串行，provider 内并发 |
| P1 | ReadError 未重试 | 排查异常传播路径 |
| P2 | TTFT-C/空内容/500 | 添加计时日志后进一步分析 |
