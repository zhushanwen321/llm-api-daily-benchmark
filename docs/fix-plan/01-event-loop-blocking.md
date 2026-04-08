# P0-01: 消除事件循环阻塞

## 问题

`ExecutionScorer.score()` 调用 `subprocess.run()` 是同步阻塞操作。
在 asyncio 单线程模型下，这会冻结整个事件循环，影响**所有 provider** 的所有协程。

**当前代码** (`benchmark/scorers/execution_scorer.py:63`):
```python
result = subprocess.run(
    [sys.executable, script_path],
    capture_output=True, text=True,
    timeout=self.timeout,  # 最长 30s
)
```

**影响量化**:
- zai: 3 模型 x 15 bigcodebench = 45 次 subprocess 调用
- 平均每次阻塞 ~15s，总计 ~675s 事件循环冻结
- 冻结期间：API streaming 暂停（→ ReadError）、semaphore 操作暂停、DB 写入暂停

## 修复方案

### 方案：`subprocess.run()` → `asyncio.create_subprocess_exec()`

将 `ExecutionScorer.score()` 改为异步方法，使用 asyncio 原生子进程 API。
调用方 `_evaluate_task` 已在 async 上下文中，改动链路最短。

### 改动文件

#### 1. `benchmark/scorers/base.py`
- `BaseScorer` 新增 `async ascore()` 方法
- 默认实现用 `asyncio.to_thread(self.score, ctx)` 包装同步 `score()`
- 保持 `score()` 不变，向后兼容测试代码

#### 2. `benchmark/scorers/execution_scorer.py`
- 重写 `ascore()` 方法，用 `asyncio.create_subprocess_exec()` 替代 `subprocess.run()`
- 提取公共逻辑 `_evaluate_result(returncode, stdout, stderr)` 供 `score()` 和 `ascore()` 复用
- `score()` 保留同步实现（仅测试使用），内部调 `_run_and_score` 不变

#### 3. 其余 scorer（math / choice_match / keyword_match）
- 无需改动，继承 BaseScorer 的默认 `ascore()`
- 这些 scorer 都是纯计算，耗时 < 1ms，`asyncio.to_thread()` 的开销可忽略

#### 4. `benchmark/cli.py` (`_evaluate_task`)
- 将 `scorer.score(ctx)` 改为 `await scorer.ascore(ctx)`

### 关键实现细节

```python
# base.py - 默认 ascore 用 to_thread 包装同步 score
class BaseScorer(ABC):
    @abstractmethod
    def score(self, ctx: ScoringContext) -> ScoreResult: ...

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """异步评分。默认用线程池包装同步 score()。"""
        return await asyncio.to_thread(self.score, ctx)

# execution_scorer.py - 重写 ascore 用原生异步 subprocess
class ExecutionScorer(BaseScorer):
    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        if not ctx.model_answer.strip():
            return ScoreResult(score=0.0, passed=False, ...)
        # ... 构建代码、写入临时文件（同 score）...
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return self._evaluate_result(0, "", f"Timeout after {self.timeout}s")
        return self._evaluate_result(proc.returncode, stdout, stderr)
```

### 为什么选 `ascore()` 而非改 `score()` 为 async

- 测试代码（5 个 scorer 测试文件）全部同步调用 `score()`，改动面大
- 4 个 scorer 中只有 ExecutionScorer 有真实异步需求（subprocess 5-30s）
- 其余 3 个是纯计算 <1ms，用 `asyncio.to_thread()` 包装即可

## 验证方式

1. 单独测试：`pytest tests/` 确认 ExecutionScorer 的 ascore 行为正确
2. 集成测试：运行 `benchmark evaluate --model zai/glm-5 --dimension backend-dev --samples 5`
3. 检查 GANTT 日志：确认 `score_block` 不再出现 > 1s 的值
4. 检查事件循环冻结：bigcodebench 评分期间，其他维度的 streaming 应正常进行
