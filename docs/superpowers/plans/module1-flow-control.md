# 模块 1：流量控制优化（Task 1-4）

### Task 1: 创建 AsyncConcurrencyLimiter + 测试

**Files:**
- Create: `benchmark/core/concurrency.py`
- Create: `tests/test_concurrency.py`

- [ ] **Step 1: 编写 AsyncConcurrencyLimiter 测试**

```python
# tests/test_concurrency.py
"""AsyncConcurrencyLimiter 测试。"""
import asyncio
import pytest
from benchmark.core.concurrency import AsyncConcurrencyLimiter


class TestAsyncConcurrencyLimiter:
    def setup_method(self):
        """每个测试前清理单例缓存，避免测试间污染。"""
        AsyncConcurrencyLimiter._instances.clear()

    def test_get_or_create_returns_same_instance(self):
        a = AsyncConcurrencyLimiter.get_or_create("provider_a", 5)
        b = AsyncConcurrencyLimiter.get_or_create("provider_a", 999)  # max_concurrency 被忽略
        assert a is b

    def test_get_or_create_different_providers(self):
        a = AsyncConcurrencyLimiter.get_or_create("p1", 2)
        b = AsyncConcurrencyLimiter.get_or_create("p2", 3)
        assert a is not b

    def test_concurrency_limit_respected(self):
        """同时运行 max_concurrency 个任务后，第 max_concurrency+1 个必须等待。"""
        max_conc = 2
        limiter = AsyncConcurrencyLimiter.get_or_create("test", max_conc)
        running = 0
        peak = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal running, peak
            await limiter.acquire()
            async with lock:
                running += 1
                peak = max(peak, running)
            await asyncio.sleep(0.05)
            async with lock:
                running -= 1
            limiter.release()

        async def run():
            await asyncio.gather(*[worker() for _ in range(5)])

        asyncio.run(run())
        assert peak <= max_conc

    def test_release_without_acquire_no_error(self):
        """release 多次不应抛异常（Semaphore 行为）。"""
        limiter = AsyncConcurrencyLimiter(max_concurrency=1)
        limiter.release()  # 不抛异常即可

    def test_max_concurrency_one_serial(self):
        """max_concurrency=1 时任务串行执行。"""
        limiter = AsyncConcurrencyLimiter.get_or_create("serial", 1)
        order = []

        async def worker(idx):
            await limiter.acquire()
            order.append(f"start-{idx}")
            await asyncio.sleep(0.02)
            order.append(f"end-{idx}")
            limiter.release()

        async def run():
            await asyncio.gather(*[asyncio.create_task(worker(i)) for i in range(3)])

        asyncio.run(run())
        # max_concurrency=1，第一个 start 必须对应第一个 end
        starts = [i for i, x in enumerate(order) if x.startswith("start")]
        ends = [i for i, x in enumerate(order) if x.startswith("end")]
        for s, e in zip(starts, ends):
            assert s < e
```

- [ ] **Step 2: 验证测试失败（模块不存在）**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_concurrency.py -v`
Expected: `ModuleNotFoundError: No module named 'benchmark.core.concurrency'`

- [ ] **Step 3: 编写 AsyncConcurrencyLimiter 实现**

```python
# benchmark/core/concurrency.py
"""异步并发控制器。按 provider 维度控制同时进行的 API 请求数。"""
from __future__ import annotations

import asyncio

class AsyncConcurrencyLimiter:
    """provider 级并发流控制器。

    使用 asyncio.Semaphore 限制同一 provider 的并发请求数，
    通过 get_or_create 工厂方法实现 provider 维度的单例管理。
    """

    _instances: dict[str, AsyncConcurrencyLimiter] = {}

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self._sem = asyncio.Semaphore(max_concurrency)

    @classmethod
    def get_or_create(cls, provider: str, max_concurrency: int) -> AsyncConcurrencyLimiter:
        """获取或创建指定 provider 的并发控制器。

        已存在的实例会忽略 max_concurrency 参数，保证运行时行为一致。
        """
        if provider not in cls._instances:
            cls._instances[provider] = cls(max_concurrency)
        return cls._instances[provider]

    async def acquire(self) -> None:
        await self._sem.acquire()

    def release(self) -> None:
        self._sem.release()
```

- [ ] **Step 4: 验证测试通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_concurrency.py -v`
Expected: 全部 5 个测试通过

- [ ] **Step 5: 提交**

```
git add benchmark/core/concurrency.py tests/test_concurrency.py
git commit -m "feat(core): 新增 AsyncConcurrencyLimiter 并发控制器"
```

---

### Task 2: 更新 config.py（max_concurrency + 兼容 rate_limit）+ 更新测试

**Files:**
- Modify: `benchmark/config.py:69-121`（get_model_config 函数）
- Modify: `tests/test_config.py`

- [ ] **Step 1: 更新测试文件**

在 `tests/test_config.py` 的 `TestGetModelConfig` 类中，将 `test_rate_limit_returned` 和 `test_no_rate_limit_returns_none` 替换为：

```python
def test_max_concurrency_returned(self, tmp_path):
    cfg_path = _write_test_config(tmp_path, {
        "providers": {
            "glm": {
                "api_key": "k", "api_base": "https://api.test.com/v1/",
                "max_concurrency": 5, "models": {"glm-4.7": {}},
            }
        }
    })
    result = get_model_config("glm/glm-4.7", models_path=cfg_path)
    assert result["max_concurrency"] == 5
    assert "rate_limit" not in result

def test_no_max_concurrency_returns_none(self, tmp_path):
    cfg_path = _write_test_config(tmp_path, {
        "providers": {
            "glm": {
                "api_key": "k", "api_base": "https://api.test.com/v1/",
                "models": {"glm-4.7": {}},
            }
        }
    })
    result = get_model_config("glm/glm-4.7", models_path=cfg_path)
    assert result["max_concurrency"] is None

def test_rate_limit_deprecated_mapped_to_max_concurrency(self, tmp_path):
    """旧配置 rate_limit 应映射到 max_concurrency 并打印 deprecation warning。"""
    cfg_path = _write_test_config(tmp_path, {
        "providers": {
            "glm": {
                "api_key": "k", "api_base": "https://api.test.com/v1/",
                "rate_limit": 3, "models": {"glm-4.7": {}},
            }
        }
    })
    with pytest.warns(DeprecationWarning, match="rate_limit.*deprecated.*max_concurrency"):
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
    assert result["max_concurrency"] == 3
    assert "rate_limit" not in result

def test_max_concurrency_negative_raises(self, tmp_path):
    cfg_path = _write_test_config(tmp_path, {
        "providers": {
            "glm": {
                "api_key": "k", "api_base": "https://api.test.com/v1/",
                "max_concurrency": 0, "models": {"glm-4.7": {}},
            }
        }
    })
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        get_model_config("glm/glm-4.7", models_path=cfg_path)
```

- [ ] **Step 2: 验证测试失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_config.py -v`
Expected: 新测试因返回 `rate_limit` 而非 `max_concurrency` 失败

- [ ] **Step 3: 修改 config.py 的 get_model_config 函数**

将 `benchmark/config.py` 中 return 语句之前的 `rate_limit` 逻辑替换为：

```python
    # 并发控制：优先读取 max_concurrency，兼容旧配置 rate_limit
    max_concurrency = None
    if "max_concurrency" in provider_cfg:
        max_concurrency = provider_cfg["max_concurrency"]
    elif "rate_limit" in provider_cfg:
        import warnings
        warnings.warn(
            f"Provider '{provider_name}': 'rate_limit' 配置已废弃，"
            f"请改用 'max_concurrency'。当前值 {provider_cfg['rate_limit']} "
            f"将自动映射为 max_concurrency。",
            DeprecationWarning,
            stacklevel=2,
        )
        max_concurrency = provider_cfg["rate_limit"]

    if max_concurrency is not None:
        max_concurrency = int(max_concurrency)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

    return {
        "provider": provider_name,
        "api_key": _resolve_env_var(provider_cfg["api_key"], "api_key"),
        "api_base": provider_cfg["api_base"],
        "max_tokens": model_cfg.get("max_tokens", default_max_tokens),
        "max_concurrency": max_concurrency,
        "thinking": model_cfg.get("thinking", {}),
    }
```

同时更新 `get_model_config` 的 docstring，将 `rate_limit` 改为 `max_concurrency`。

- [ ] **Step 4: 验证测试通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_config.py -v`
Expected: 全部 config 测试通过

- [ ] **Step 5: 提交**

```
git add benchmark/config.py tests/test_config.py
git commit -m "refactor(config): rate_limit 迁移为 max_concurrency，保留旧配置兼容"
```

---

### Task 3: 更新 llm_adapter.py（替换并发控制）+ 更新测试

**Files:**
- Modify: `benchmark/core/llm_adapter.py:21,39-42,44-56,58-78,119-121,471-487`
- Modify: `tests/test_llm_adapter.py`

- [ ] **Step 1: 新增并发控制测试**

在 `tests/test_llm_adapter.py` 中新增：

```python
import threading
from benchmark.core.concurrency import AsyncConcurrencyLimiter

@patch("benchmark.core.llm_adapter.get_model_config")
def test_sync_uses_threading_semaphore(mock_config):
    """sync 路径应使用 threading.Semaphore。"""
    mock_config.return_value = {
        "api_key": "k", "api_base": "https://api.test.com/v1",
        "provider": "test", "max_tokens": 4096, "max_concurrency": 2,
    }
    adapter = LLMEvalAdapter()
    sem = adapter._get_or_create_sync_semaphore("test/model")
    assert sem is not None
    assert isinstance(sem, threading.Semaphore)

@patch("benchmark.core.llm_adapter.get_model_config")
def test_async_uses_concurrency_limiter(mock_config):
    """async 路径应使用 AsyncConcurrencyLimiter。"""
    mock_config.return_value = {
        "api_key": "k", "api_base": "https://api.test.com/v1",
        "provider": "test", "max_tokens": 4096, "max_concurrency": 3,
    }
    adapter = LLMEvalAdapter()
    limiter = adapter._get_or_create_async_limiter("test/model")
    assert limiter is not None
    assert isinstance(limiter, AsyncConcurrencyLimiter)

@patch("benchmark.core.llm_adapter.get_model_config")
def test_no_max_concurrency_no_limiter(mock_config):
    """max_concurrency 为 None 时不创建 limiter。"""
    mock_config.return_value = {
        "api_key": "k", "api_base": "https://api.test.com/v1",
        "provider": "test", "max_tokens": 4096, "max_concurrency": None,
    }
    adapter = LLMEvalAdapter()
    assert adapter._get_or_create_sync_semaphore("test/model") is None
    assert adapter._get_or_create_async_limiter("test/model") is None
```

- [ ] **Step 2: 验证测试失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_llm_adapter.py -v`
Expected: `ImportError` 或 `AttributeError`

- [ ] **Step 3: 修改 llm_adapter.py**

**导入（第21行）：** 删除 `from benchmark.core.rate_limiter import TokenBucketRateLimiter`，新增：

```python
import threading
from benchmark.core.concurrency import AsyncConcurrencyLimiter
```

**类变量（第39-42行）：**

```python
    _provider_sync_semaphores: dict[str, threading.Semaphore] = {}
    _provider_async_limiters: dict[str, AsyncConcurrencyLimiter] = {}
```

**__init__（第44-56行）：** 删除 `self._limiter` 和 `_get_or_create_limiter` 调用：

```python
    def __init__(self, model=None, max_retries=5, timeout=300):
        self.max_retries = max_retries
        self.timeout = timeout
        self._model_cache: dict[str, dict[str, Any]] = {}
        if model:
            self._model_cache[model] = get_model_config(model)
```

**删除 `_get_or_create_limiter` 和 `_get_or_create_semaphore`（第58-78行），替换为：**

```python
    def _get_or_create_sync_semaphore(self, model: str) -> threading.Semaphore | None:
        cfg = self._get_model_config(model)
        max_conc = cfg.get("max_concurrency")
        if max_conc is None:
            return None
        provider = cfg["provider"]
        if provider not in self._provider_sync_semaphores:
            self._provider_sync_semaphores[provider] = threading.Semaphore(max_conc)
        return self._provider_sync_semaphores[provider]

    def _get_or_create_async_limiter(self, model: str) -> AsyncConcurrencyLimiter | None:
        cfg = self._get_model_config(model)
        max_conc = cfg.get("max_concurrency")
        if max_conc is None:
            return None
        provider = cfg["provider"]
        return AsyncConcurrencyLimiter.get_or_create(provider, max_conc)
```

**generate() 中的限流（第119-121行）：** 将 `if self._limiter is not None: self._limiter.acquire()` 替换为在函数开头获取信号量，将整个 for 循环包裹在信号量上下文中：

```python
    def generate(self, prompt, model, temperature=0.0, max_tokens=None):
        cfg = self._get_model_config(model)
        sem = self._get_or_create_sync_semaphore(model)
        # ... 构建 headers, payload 等（不变） ...

        def _do_request():
            last_error = None
            for attempt in range(self.max_retries):
                # ... 原有的 requests.post 调用和重试逻辑 ...
                pass
            raise ConnectionError(f"[{model}] 重试 {self.max_retries} 次后仍失败: {last_error}") from last_error

        if sem is not None:
            with sem:
                return _do_request()
        return _do_request()
```

实际实现时将 generate() 方法重构为：payload 构建部分不变，将 `last_error` 循环及之后的逻辑提取到 `_do_request()` 内部闭包中，外层用 `with sem:` 包裹。

**agenerate()（第471-487行）：**

```python
    async def agenerate(self, prompt, model, temperature=0.0, max_tokens=None):
        limiter = self._get_or_create_async_limiter(model)
        if limiter is not None:
            await limiter.acquire()
            try:
                return await self._do_agenerate(prompt, model, temperature, max_tokens)
            finally:
                limiter.release()
        return await self._do_agenerate(prompt, model, temperature, max_tokens)
```

- [ ] **Step 4: 验证测试通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_llm_adapter.py -v`
Expected: 全部 llm_adapter 测试通过

- [ ] **Step 5: 提交**

```
git add benchmark/core/llm_adapter.py tests/test_llm_adapter.py
git commit -m "refactor(llm_adapter): 用并发控制器替换令牌桶限流器"
```

---

### Task 4: 删除 rate_limiter.py + 删除旧测试

**Files:**
- Delete: `benchmark/core/rate_limiter.py`
- Delete: `tests/test_rate_limiter.py`

- [ ] **Step 1: 确认没有残留引用**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && grep -r "rate_limiter\|TokenBucketRateLimiter" --include="*.py" benchmark/ tests/`
Expected: 无输出

- [ ] **Step 2: 删除文件**

Run:
```bash
rm benchmark/core/rate_limiter.py
rm tests/test_rate_limiter.py
```

- [ ] **Step 3: 运行全量测试确认无回归**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest -v`
Expected: 全部测试通过

- [ ] **Step 4: 提交**

```
git add -A
git commit -m "chore: 删除已废弃的令牌桶限流器模块及测试"
```
