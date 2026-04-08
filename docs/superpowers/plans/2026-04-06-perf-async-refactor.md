# 性能修复实施计划

> Spec: `docs/superpowers/specs/2026-04-06-perf-fix-design.md`
> 分支: `fix/perf-async-refactor`

## 任务概览

| Task | 内容 | 依赖 | 改动文件数 |
|------|------|------|-----------|
| 1 | BaseScorer + ExecutionScorer 异步接口 | 无 | 3 |
| 2 | Database 异步写入方法 | 无 | 2 |
| 3 | LLMEvalAdapter semaphore 重构 | 无 | 2 |
| 4 | cli.py 集成：ascore + asave | 1,2 | 1 |
| 5 | _run_multi_evaluation 按 provider 分组 | 4 | 1 |
| 6 | 死代码清理 + 测试更新 | 1-5 | 3 |

## 提交策略

每个 Task 一个 commit，commit message 格式：
```
fix(perf): [简述]

Refs: docs/superpowers/specs/2026-04-06-perf-fix-design.md
```

---

### Task 1: BaseScorer 新增 ascore() + ExecutionScorer 原生异步

**目标**: 消除 ExecutionScorer 中 subprocess.run() 对事件循环的阻塞。

**Files:**
- Modify: `benchmark/scorers/base.py`
- Modify: `benchmark/scorers/execution_scorer.py`
- Modify: `tests/test_base_scorer.py`
- Modify: `tests/test_execution_scorer.py`

- [ ] **Step 1: 在 BaseScorer 新增 ascore() 默认实现**

`benchmark/scorers/base.py` — 在 `get_metric_name` 方法后新增：

```python
import asyncio

class BaseScorer(ABC):
    # ... 现有 score() 和 get_metric_name() 不变 ...

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """异步评分。默认用线程池包装同步 score()，子类可重写。"""
        return await asyncio.to_thread(self.score, ctx)
```

- [ ] **Step 2: 在 ExecutionScorer 新增 ascore() 重写**

`benchmark/scorers/execution_scorer.py` — 新增 `import asyncio`，在 `score()` 方法后新增 `ascore()` 方法。
同时提取 `_evaluate_result()` 辅助方法供 `score()` 和 `ascore()` 复用评分逻辑。

`score()` 方法保持不变（仅测试使用）。新增的 `ascore()` 用 `asyncio.create_subprocess_exec` 替代 `subprocess.run`：

```python
import asyncio

class ExecutionScorer(BaseScorer):
    # ... __init__ 不变 ...

    def score(self, ctx: ScoringContext) -> ScoreResult:
        # ... 现有实现完全不变 ...

    async def ascore(self, ctx: ScoringContext) -> ScoreResult:
        """异步评分：用 asyncio 原生子进程替代 subprocess.run。"""
        if not ctx.model_answer.strip():
            return ScoreResult(score=0.0, passed=False,
                details={"error": "Empty model output"},
                reasoning="Model produced no code")

        test_code = ctx.task.metadata.get("test", "")
        entry_point = ctx.task.metadata.get("entry_point", "")
        full_code = self._build_executable(ctx.model_answer, test_code, entry_point)

        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="bench_exec_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(full_code)
            return await self._async_run_and_score(temp_path, ctx.task.task_id)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _async_run_and_score(self, script_path: str, task_id: str) -> ScoreResult:
        """异步执行脚本并评分。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ScoreResult(score=0.0, passed=False,
                    details={"error": f"Timeout after {self.timeout}s"},
                    reasoning=f"Execution timed out after {self.timeout} seconds")

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                return ScoreResult(score=100.0, passed=True,
                    details={"stdout": stdout[-500:]},
                    reasoning="All test cases passed")

            return ScoreResult(score=0.0, passed=False,
                details={"returncode": proc.returncode, "stderr": stderr[-1000:]},
                reasoning=f"Execution failed with return code {proc.returncode}")

        except Exception as exc:
            return ScoreResult(score=0.0, passed=False,
                details={"error": str(exc)},
                reasoning=f"Execution error: {exc}")
```

- [ ] **Step 3: 更新测试**

`tests/test_base_scorer.py` — 新增异步测试验证默认 `ascore()` 委托给 `score()`：

```python
import asyncio

def test_base_scorer_ascore_delegates_to_score():
    """ascore() 默认实现应委托给 score()。"""
    scorer = DummyScorer()
    ctx = _make_ctx("42", "42")
    result = asyncio.run(scorer.ascore(ctx))
    assert result.passed is True
    assert result.score == 100.0
```

`tests/test_execution_scorer.py` — 新增异步测试：

```python
import asyncio

def test_execution_ascore_correct_code():
    """ascore 应正确执行通过的代码。"""
    code = "def add(a, b):\n    return a + b"
    test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
    scorer = ExecutionScorer(timeout=10)
    result = asyncio.run(scorer.ascore(_make_ctx(code, test, "add")))
    assert result.passed is True
    assert result.score == 100.0


def test_execution_ascore_wrong_code():
    """ascore 应正确检测失败代码。"""
    code = "def add(a, b):\n    return a - b"
    test = """
import unittest
class Test(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
if __name__ == "__main__":
    unittest.main()
"""
    scorer = ExecutionScorer(timeout=10)
    result = asyncio.run(scorer.ascore(_make_ctx(code, test, "add")))
    assert result.passed is False
    assert result.score == 0.0


def test_execution_ascore_empty_output():
    """ascore 对空输出应返回 0 分。"""
    scorer = ExecutionScorer()
    result = asyncio.run(scorer.ascore(_make_ctx("")))
    assert result.passed is False
    assert result.details["error"] == "Empty model output"
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_base_scorer.py tests/test_execution_scorer.py -v
```

预期：所有测试通过，包括新增的 4 个异步测试。

---

### Task 2: Database 新增 asave_result / asave_metrics

**目标**: 消除 sqlite3 写入对事件循环的阻塞。

**Files:**
- Modify: `benchmark/models/database.py`
- Modify: `tests/` (无现有 DB 测试文件，需新建)

- [ ] **Step 1: 在 Database 新增异步方法**

`benchmark/models/database.py` — 在 `save_metrics()` 方法后、`get_results()` 方法前新增：

```python
import asyncio

class Database:
    # ... 现有方法不变 ...

    async def asave_result(self, result: EvalResult) -> str:
        """异步保存单题评测结果。"""
        return await asyncio.to_thread(self.save_result, result)

    async def asave_metrics(self, metrics: ApiCallMetrics) -> str:
        """异步保存 API 调用指标。"""
        return await asyncio.to_thread(self.save_metrics, metrics)
```

- [ ] **Step 2: 验证**

```bash
python -c "
import asyncio
from benchmark.models.database import Database
from benchmark.models.schemas import EvalResult, ApiCallMetrics
from datetime import datetime

async def test():
    db = Database(':memory:')
    r = EvalResult(result_id='t1', run_id='r1', task_id='q1',
        task_content='test', model_output='out', model_think='',
        model_answer='ans', functional_score=100, final_score=100,
        passed=True, execution_time=1.0, created_at=datetime.now())
    rid = await db.asave_result(r)
    assert rid == 't1'

    m = ApiCallMetrics(result_id='t1', prompt_tokens=10,
        completion_tokens=5, duration=1.0, tokens_per_second=5.0,
        created_at=datetime.now())
    mid = await db.asave_metrics(m)
    assert mid == 't1'
    print('OK')

asyncio.run(test())
"
```

预期输出: `OK`

---

### Task 3: LLMEvalAdapter.agenerate() semaphore 重构

**目标**: 429 重试期间释放 semaphore，避免空占并发 slot。

**Files:**
- Modify: `benchmark/core/llm_adapter.py`
- Modify: `tests/test_llm_adapter.py`

- [ ] **Step 1: 重构 agenerate() — semaphore 粒度细化**

当前 `agenerate()` 的结构：
```
acquire → _do_agenerate(含重试循环) → release
```

改为：
```
重试循环 { acquire → 单次请求 → release → if 失败: sleep }
```

`benchmark/core/llm_adapter.py` — 将 `agenerate()` 和 `_do_agenerate()` 重构为：
`agenerate()` 管理重试循环，每次 attempt 独立 acquire/release semaphore。
新增 `_single_api_request()` 处理单次 HTTP 请求（无重试）。

具体改动：

1. 删除 `agenerate()` 中的 semaphore acquire/release（第 488-501 行）
2. 删除整个 `_do_agenerate()` 方法（第 503-871 行）
3. 新增 `_single_api_request()` 方法：包含当前 `_do_agenerate` 中的单次请求逻辑（SSE 解析、响应处理），但不包含重试循环
4. 重写 `agenerate()`：外层是重试循环，每次 attempt 调用 `acquire → _single_api_request → release`，失败时 sleep

```python
async def agenerate(self, prompt: str, model: str,
                    temperature: float = 0.0,
                    max_tokens: int | None = None) -> GenerateResponse:
    cfg = self._get_model_config(model)
    api_key = cfg["api_key"]
    api_base = cfg["api_base"].rstrip("/")
    model_max_tokens = cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)
    effective_max_tokens = (
        min(max_tokens, model_max_tokens) if max_tokens is not None
        else model_max_tokens
    )

    if api_base.endswith("/chat/completions") or api_base.endswith("/messages"):
        url = api_base
    else:
        url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": os.getenv("LLM_USER_AGENT", "claude-code/1.0.0"),
    }
    payload: dict[str, Any] = {
        "model": model.split("/", 1)[1] if "/" in model else model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    thinking_cfg = cfg.get("thinking", {})
    if thinking_cfg.get("enabled") and thinking_cfg.get("request_params"):
        payload.update(thinking_cfg["request_params"])

    limiter = self._get_or_create_async_limiter(model)

    last_error: Exception | None = None
    for attempt in range(self.max_retries):
        try:
            if limiter is not None:
                await limiter.acquire()
            try:
                return await self._single_api_request(
                    url, headers, payload, model, thinking_cfg
                )
            finally:
                if limiter is not None:
                    limiter.release()
        except _NonRetryableError:
            raise
        except (TimeoutError, httpx.HTTPStatusError, httpx.StreamError,
                httpx.ConnectError, httpx.TimeoutException, ConnectionError) as exc:
            last_error = exc
            if attempt < self.max_retries - 1:
                wait = self._calc_backoff(exc, attempt)
                logger.warning(
                    f"[{model}] async attempt {attempt + 1}/{self.max_retries} "
                    f"失败 ({type(exc).__name__}): {exc}. {wait}s 后重试..."
                )
                await asyncio.sleep(wait)

    raise ConnectionError(
        f"[{model}] 重试 {self.max_retries} 次后仍失败: {last_error}"
    ) from last_error
```

需要定义 `_NonRetryableError` 异常和 `_calc_backoff()` 辅助方法：

```python
class _NonRetryableError(Exception):
    """4xx 客户端错误（非 429），不重试。"""

def _calc_backoff(self, exc: Exception, attempt: int) -> int:
    """根据异常类型计算退避时间。"""
    is_rate_limited = (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 429
    )
    if is_rate_limited:
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(int(retry_after), 120)
            except ValueError:
                pass
        return min(10 * (2 ** attempt), 120)
    return min(2 * (2 ** attempt), 120)
```

在 `_single_api_request()` 中，遇到 4xx（非 429）时抛出 `_NonRetryableError` 而非 `ConnectionError`：
```python
if 400 <= status < 500 and status != 429:
    raise _NonRetryableError(f"客户端错误 ({status})")
```

`_single_api_request()` 的其余逻辑（SSE 解析、流处理、响应构造）与当前 `_do_agenerate` 中的单次请求部分完全相同。

- [ ] **Step 2: 更新测试**

`tests/test_llm_adapter.py` — 删除 sync 相关测试（`test_generate_*`、`test_sync_uses_threading_semaphore`），
保留 `test_async_uses_concurrency_limiter` 和 `test_no_max_concurrency_no_limiter`。
新增测试验证 semaphore 在 sleep 期间被释放：

```python
@patch("benchmark.core.llm_adapter.get_model_config")
async def test_semaphore_released_during_retry(mock_config):
    """429 重试期间 semaphore 应被释放。"""
    mock_config.return_value = {
        "api_key": "k", "api_base": "https://api.test.com/v1",
        "provider": "test", "max_tokens": 4096, "max_concurrency": 1,
    }
    # 验证方式：在 sleep 期间另一个协程能获取 semaphore
    # （具体实现需要 mock httpx 返回 429 然后返回 200）
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_llm_adapter.py -v
```

---

### Task 4: cli.py 集成 — ascore + asave

**目标**: 在 `_evaluate_task` 中使用异步版本的 score 和 DB 写入。

**Files:**
- Modify: `benchmark/cli.py`

- [ ] **Step 1: 替换同步调用为异步调用**

`benchmark/cli.py` `_evaluate_task()` 中两处改动：

Phase 2（评分）:
```python
# 修改前
score_result = scorer.score(ctx)
# 修改后
score_result = await scorer.ascore(ctx)
```

Phase 3（DB 写入）:
```python
# 修改前
db.save_result(result)
db.save_metrics(ApiCallMetrics(...))
# 修改后
await db.asave_result(result)
await db.asave_metrics(ApiCallMetrics(...))
```

- [ ] **Step 2: 验证**

```bash
python -m pytest tests/ -v
```

---

### Task 5: _run_multi_evaluation 按 provider 分组

**目标**: 同 provider 的 evaluation run 串行，不同 provider 并行。

**Files:**
- Modify: `benchmark/cli.py`

- [ ] **Step 1: 新增分组函数和串行执行函数**

`benchmark/cli.py` — 在 `_run_multi_evaluation` 前新增：

```python
def _group_by_provider(
    models: list[str], dimensions: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """按 provider 分组 (model, dimension) 对。"""
    groups: dict[str, list[tuple[str, str]]] = {}
    for model in models:
        provider = model.split("/", 1)[0]
        for dim in dimensions:
            groups.setdefault(provider, []).append((model, dim))
    return groups


async def _run_provider_group(
    tasks: list[tuple[str, str]], samples: int, debug: bool
) -> None:
    """同一 provider 内串行执行 evaluation run。"""
    for model, dim in tasks:
        await _run_evaluation(model, dim, samples, debug)
```

- [ ] **Step 2: 重写 _run_multi_evaluation**

```python
async def _run_multi_evaluation(
    models: list[str], dimensions: list[str], samples: int, debug: bool
) -> None:
    """多模型 x 多维度评测。同 provider 串行，不同 provider 并行。"""
    groups = _group_by_provider(models, dimensions)
    coros = [
        _run_provider_group(tasks, samples, debug)
        for tasks in groups.values()
    ]
    await asyncio.gather(*coros)
```

- [ ] **Step 3: 验证**

```bash
python -c "
from benchmark.cli import _group_by_provider
groups = _group_by_provider(
    ['zai/glm-5', 'zai/glm-4.7', 'kimi/kimi-for-coding'],
    ['reasoning', 'backend-dev']
)
assert set(groups.keys()) == {'zai', 'kimi'}
assert len(groups['zai']) == 4  # 2 models x 2 dims
assert len(groups['kimi']) == 2  # 1 model x 2 dims
print('OK')
"
```

---

### Task 6: 死代码清理

**目标**: 删除生产代码中未使用的同步路径和死变量。

**Files:**
- Modify: `benchmark/core/llm_adapter.py`
- Modify: `tests/test_llm_adapter.py`

- [ ] **Step 1: 清理 llm_adapter.py**

删除以下内容：
1. `_provider_sync_semaphores` 类变量（约第 41 行）
2. `_provider_async_limiters` 类变量（约第 43 行）
3. `_get_or_create_sync_semaphore()` 方法（约第 57-66 行）
4. 整个 `generate()` 方法（约第 83-470 行）
5. 清理 `requests` 相关 import（`requests`, `requests.exceptions`）
6. 清理 `threading` import（如仅用于 sync semaphore）

保留：`_get_or_create_async_limiter()` 方法（仍被 agenerate 使用）。

- [ ] **Step 2: 更新测试**

`tests/test_llm_adapter.py` — 删除以下测试：
- `test_generate_returns_generate_response`
- `test_generate_handles_missing_usage`
- `test_sync_uses_threading_semaphore`

保留：
- `test_async_uses_concurrency_limiter`
- `test_no_max_concurrency_no_limiter`

- [ ] **Step 3: 全量测试**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: 验证 CLI 可用**

```bash
python -m benchmark.cli --help
python -m benchmark.cli list-datasets
```
