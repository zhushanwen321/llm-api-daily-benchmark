# provider/model 标识 + provider 级令牌桶限流 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将模型标识改为 `provider/model` 格式，并为每个 provider 添加令牌桶限流。

**Architecture:** 修改 `config.py` 的模型查找逻辑从遍历改为直接定位；新建 `rate_limiter.py` 实现令牌桶；`llm_adapter.py` 在 HTTP 请求前调用限流器。不兼容旧数据库。

**Tech Stack:** Python 3.11+, threading.Lock, logging, pytest

---

## File Structure

| 文件 | 职责 |
|------|------|
| `tests/__init__.py` | 空文件，标记 tests 为包 |
| `tests/test_rate_limiter.py` | 令牌桶限流器单元测试 |
| `tests/test_config.py` | config.py 的 provider/model 解析测试 |
| `benchmark/core/rate_limiter.py` | 新建，令牌桶限流器实现 |
| `benchmark/config.py` | 修改 `get_model_config()` 解析 `provider/model` |
| `benchmark/core/llm_adapter.py` | 集成限流器 |
| `benchmark/cli.py` | 更新 `--model` help 文案，model 字段存完整标识 |
| `benchmark/configs/models.yaml.example` | 添加 `rate_limit` 字段 |
| `quickstart.md` | 更新命令示例 |

---

### Task 1: 搭建测试基础设施

**Files:**
- Create: `tests/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 pytest 开发依赖**

在 `pyproject.toml` 末尾添加：

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: 安装 pytest**

Run: `uv sync --extra dev`
Expected: pytest 安装成功

- [ ] **Step 3: 创建 tests 包**

创建 `tests/__init__.py`（空文件）。

- [ ] **Step 4: 验证 pytest 可运行**

Run: `uv run pytest --co`
Expected: 无报错，显示 "no tests collected"

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py pyproject.toml uv.lock
git commit -m "chore: 添加 pytest 测试基础设施"
```

---

### Task 2: 实现令牌桶限流器

**Files:**
- Create: `tests/test_rate_limiter.py`
- Create: `benchmark/core/rate_limiter.py`

- [ ] **Step 1: 写限流器测试**

创建 `tests/test_rate_limiter.py`：

```python
"""令牌桶限流器测试。"""

import time
from unittest.mock import patch

from benchmark.core.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_acquire_within_rate_no_wait(self):
        """rate=10 时，连续 acquire 10 次不应等待。"""
        limiter = TokenBucketRateLimiter(rate=10)
        start = time.monotonic()
        for _ in range(10):
            limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_acquire_exceeds_rate_waits(self):
        """rate=10 时，第 11 次 acquire 应等待约 0.1s。"""
        limiter = TokenBucketRateLimiter(rate=10)
        for _ in range(10):
            limiter.acquire()
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08  # 允许小幅误差

    def test_rate_limit_one_slow(self):
        """rate=1 时，连续 3 次调用至少耗时约 2s。"""
        limiter = TokenBucketRateLimiter(rate=1)
        start = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 1.8

    def test_warning_on_wait(self):
        """等待时应打印 warning。"""
        limiter = TokenBucketRateLimiter(rate=2)
        limiter.acquire()
        limiter.acquire()
        with patch("benchmark.core.rate_limiter.logger.warning") as mock_warn:
            limiter.acquire()
            mock_warn.assert_called_once()
            assert "Rate limited" in mock_warn.call_args[0][0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_rate_limiter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.core.rate_limiter'`

- [ ] **Step 3: 实现令牌桶限流器**

创建 `benchmark/core/rate_limiter.py`：

```python
"""令牌桶限流器。控制每个 provider 的 API 调用频率。"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """令牌桶限流器。

    Args:
        rate: 每秒允许的请求数。桶容量等于 rate。
    """

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError(f"rate_limit must be positive, got {rate}")
        self._rate = rate
        self._tokens = rate
        self._max_tokens = rate
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """获取一个令牌。桶空时阻塞等待并打印 warning。"""
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                return
            wait = (1 - self._tokens) / self._rate

        logger.warning("Rate limited, waiting %.1fs", wait)
        time.sleep(wait)

        with self._lock:
            self._refill()
            self._tokens -= 1

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_rate_limiter.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: 实现令牌桶限流器 TokenBucketRateLimiter"
```

---

### Task 3: 修改 config.py 解析 provider/model 格式

**Files:**
- Create: `tests/test_config.py`
- Modify: `benchmark/config.py:49-83`

- [ ] **Step 1: 写 config 解析测试**

创建 `tests/test_config.py`：

```python
"""config.py 的 provider/model 解析测试。"""

import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from benchmark.config import get_model_config


def _write_test_config(tmp_path: Path, config: dict) -> str:
    """写入临时 models.yaml 并返回路径。"""
    p = tmp_path / "models.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return str(p)


class TestGetModelConfig:
    def test_valid_provider_model(self, tmp_path):
        """glm/glm-4.7 应正确解析。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "test-key",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {"max_tokens": 4096}},
                }
            }
        })
        with patch("benchmark.config._CONFIG_DIR", tmp_path):
            result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["provider"] == "glm"
        assert result["api_key"] == "test-key"
        assert result["max_tokens"] == 4096

    def test_rate_limit_returned(self, tmp_path):
        """有 rate_limit 时应包含在返回值中。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "rate_limit": 3,
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with patch("benchmark.config._CONFIG_DIR", tmp_path):
            result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["rate_limit"] == 3.0

    def test_no_rate_limit_returns_none(self, tmp_path):
        """无 rate_limit 时返回 None。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with patch("benchmark.config._CONFIG_DIR", tmp_path):
            result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["rate_limit"] is None

    def test_bare_model_name_raises(self, tmp_path):
        """裸名（如 glm-4.7）应报错提示格式。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with patch("benchmark.config._CONFIG_DIR", tmp_path):
            with pytest.raises(ValueError, match="provider/model"):
                get_model_config("glm-4.7", models_path=cfg_path)

    def test_unknown_provider_raises(self, tmp_path):
        """provider 不存在时报错。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with patch("benchmark.config._CONFIG_DIR", tmp_path):
            with pytest.raises(ValueError, match="Provider 'openai' not found"):
                get_model_config("openai/gpt-4", models_path=cfg_path)

    def test_unknown_model_raises(self, tmp_path):
        """provider 下无该 model 时报错。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with patch("benchmark.config._CONFIG_DIR", tmp_path):
            with pytest.raises(ValueError, match="Model 'glm-4-flash' not found under provider 'glm'"):
                get_model_config("glm/glm-4-flash", models_path=cfg_path)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — 当前 `get_model_config` 不解析 `provider/model` 格式

- [ ] **Step 3: 修改 config.py**

将 `benchmark/config.py` 中的 `get_model_config` 函数（第 49-83 行）替换为：

```python
def get_model_config(model_name: str, models_path: str | Path | None = None) -> dict[str, Any]:
    """获取指定模型的完整配置（合并 provider 和 model 层级的配置）。

    Args:
        model_name: 模型标识，格式为 provider/model（如 glm/glm-4.7）。
        models_path: 模型配置路径。为 None 时使用 configs/models.yaml。

    Returns:
        合并后的配置字典，包含 provider, api_key, api_base, max_tokens, rate_limit 字段。

    Raises:
        ValueError: 格式错误、provider 不存在或 model 不存在。
    """
    if "/" not in model_name:
        raise ValueError(
            f"Model identifier must be 'provider/model' format, got '{model_name}'. "
            "Example: glm/glm-4.7"
        )

    provider_name, model_id = model_name.split("/", 1)

    cfg = load_models_config(models_path)
    providers = cfg.get("providers", {})

    if provider_name not in providers:
        available = ", ".join(providers.keys()) or "none"
        raise ValueError(
            f"Provider '{provider_name}' not found. Available: {available}"
        )

    provider_cfg = providers[provider_name]
    models = provider_cfg.get("models", {})

    if model_id not in models:
        available = ", ".join(models.keys()) or "none"
        raise ValueError(
            f"Model '{model_id}' not found under provider '{provider_name}'. "
            f"Available: {available}"
        )

    model_cfg = models[model_id] or {}
    return {
        "provider": provider_name,
        "api_key": provider_cfg["api_key"],
        "api_base": provider_cfg["api_base"],
        "max_tokens": model_cfg.get("max_tokens", 4096),
        "rate_limit": float(provider_cfg["rate_limit"]) if "rate_limit" in provider_cfg else None,
    }
```

注意：`load_models_config` 函数签名不变，不需要改动。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/config.py tests/test_config.py
git commit -m "feat(config): get_model_config 解析 provider/model 格式"
```

---

### Task 4: 集成限流器到 LLMEvalAdapter

**Files:**
- Modify: `benchmark/core/llm_adapter.py`

- [ ] **Step 1: 修改 llm_adapter.py**

将 `benchmark/core/llm_adapter.py` 完整替换为：

```python
"""LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from benchmark.config import get_model_config
from benchmark.core.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

# 当模型配置未指定 max_tokens 时的默认值
_DEFAULT_MAX_TOKENS = 4096


class LLMEvalAdapter:
    """LLM 调用适配器.

    从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
    支持重试（最多 max_retries 次，指数退避）。
    支持 provider 级令牌桶限流。
    """

    # provider -> limiter 实例缓存，同一 provider 的所有模型共享
    _provider_limiters: dict[str, TokenBucketRateLimiter] = {}

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 5,
        timeout: int = 300,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._limiter: TokenBucketRateLimiter | None = None
        if model:
            self._model_cache[model] = get_model_config(model)
            self._limiter = self._get_or_create_limiter(model)

    def _get_or_create_limiter(self, model: str) -> TokenBucketRateLimiter | None:
        """获取或创建 provider 级限流器。"""
        cfg = self._get_model_config(model)
        rate = cfg.get("rate_limit")
        if rate is None:
            return None
        provider = cfg["provider"]
        if provider not in self._provider_limiters:
            self._provider_limiters[provider] = TokenBucketRateLimiter(rate=rate)
        return self._provider_limiters[provider]

    def _get_model_config(self, model: str) -> dict[str, Any]:
        """获取模型配置，带缓存."""
        if model not in self._model_cache:
            self._model_cache[model] = get_model_config(model)
        return self._model_cache[model]

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """调用 LLM 生成文本.

        Args:
            prompt: 输入提示.
            model: 模型标识（provider/model 格式）.
            temperature: 温度参数（评测时固定为 0）.
            max_tokens: 最大输出 token 数.

        Returns:
            模型生成的文本.

        Raises:
            ValueError: 模型未配置.
            ConnectionError: 重试耗尽后仍失败.
        """
        cfg = self._get_model_config(model)
        api_key = cfg["api_key"]
        api_base = cfg["api_base"].rstrip("/")
        model_max_tokens = cfg.get("max_tokens", max_tokens)

        # 限流
        if self._limiter is not None:
            self._limiter.acquire()

        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": min(max_tokens, model_max_tokens),
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    # 429 限频用更长退避，其他错误用标准退避
                    is_rate_limited = (
                        isinstance(exc, requests.exceptions.HTTPError)
                        and exc.response is not None
                        and exc.response.status_code == 429
                    )
                    base = 10 if is_rate_limited else 2
                    wait = min(base * 2**attempt, 120)
                    time.sleep(wait)

        raise ConnectionError(
            f"Failed after {self.max_retries} retries for model '{model}': {last_error}"
        ) from last_error
```

- [ ] **Step 2: 验证导入正常**

Run: `uv run python -c "from benchmark.core.llm_adapter import LLMEvalAdapter; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmark/core/llm_adapter.py
git commit -m "feat(llm_adapter): 集成 provider 级令牌桶限流"
```

---

### Task 5: 更新 CLI 和配置文件

**Files:**
- Modify: `benchmark/cli.py:55-56`
- Modify: `benchmark/configs/models.yaml.example`

- [ ] **Step 1: 更新 cli.py 的 --model help 文案**

将 `benchmark/cli.py` 第 55-56 行：

```python
@click.option(
    "--model", required=True, help="模型名称（需在 configs/models.yaml 中配置）"
)
```

改为：

```python
@click.option(
    "--model", required=True, help="模型标识，格式: provider/model（如 glm/glm-4.7）"
)
```

- [ ] **Step 2: 更新 models.yaml.example**

将 `benchmark/configs/models.yaml.example` 替换为：

```yaml
# 模型配置模板
# 复制为 models.yaml 并填入真实 API key
# models.yaml 已加入 .gitignore，不会被提交
#
# 模型标识格式: provider/model（如 glm/glm-4.7）
# rate_limit: 可选，每秒允许的请求数，不配置则不限流

providers:
  glm:
    api_key: "YOUR_GLM_API_KEY_HERE"
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
    rate_limit: 2
    models:
      glm-4.7:
        max_tokens: 4096
      glm-4-flash: {}

  openai:
    api_key: "YOUR_OPENAI_API_KEY_HERE"
    api_base: "https://api.openai.com/v1/"
    models:
      gpt-4:
        max_tokens: 4096
      gpt-4o: {}
```

- [ ] **Step 3: 验证 CLI help 输出**

Run: `uv run python -m benchmark evaluate --help`
Expected: `--model` 帮助文本显示 "格式: provider/model"

- [ ] **Step 4: Commit**

```bash
git add benchmark/cli.py benchmark/configs/models.yaml.example
git commit -m "feat(cli): 更新 --model 参数为 provider/model 格式"
```

---

### Task 6: 更新 quickstart.md

**Files:**
- Modify: `quickstart.md`

- [ ] **Step 1: 更新文档中所有模型标识为 provider/model 格式**

将 `quickstart.md` 中所有 `--model glm-4.7` 替换为 `--model glm/glm-4.7`，`--model my-model` 替换为 `--model my-provider/my-model`。

具体改动：

1. YAML 示例中添加 `rate_limit` 字段注释
2. `--model glm-4.7` → `--model glm/glm-4.7`（第 91、94、97、109 行）
3. `--model my-model` → `--model my-provider/my-model`（第 145 行）
4. 项目结构中 core/ 下添加 `rate_limiter.py` 条目

- [ ] **Step 2: Commit**

```bash
git add quickstart.md
git commit -m "docs(quickstart): 更新为 provider/model 格式"
```

---

### Task 7: 清理旧数据库，运行全量测试

**Files:**
- Delete: `benchmark/data/results.db`（如果存在）

- [ ] **Step 1: 删除旧数据库**

旧数据库中 model 字段存储的是裸名，与新格式不兼容。

Run: `rm -f benchmark/data/results.db`

- [ ] **Step 2: 运行全量测试**

Run: `uv run pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 3: 端到端验证（需要真实 API key）**

确保 `benchmark/configs/models.yaml` 中配置了正确的 API key，然后：

Run: `uv run python -m benchmark evaluate --model glm/glm-4.7 --dimension reasoning --samples 1`
Expected: 评测正常运行，输出含 Score

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: 清理旧数据库，完成 provider/model 迁移"
```
