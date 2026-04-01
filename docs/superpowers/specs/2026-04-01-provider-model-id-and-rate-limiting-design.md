# 设计：provider/model 标识 + provider 级令牌桶限流

## 背景

当前模型以裸名标识（如 `glm-4.7`），在多个 provider 下存在同名模型时无法区分。且无客户端限流机制，高频调用会触发 API 429 错误。

## 变更概述

1. 模型唯一标识改为 `provider/model` 格式（如 `glm/glm-4.7`）
2. 每个 provider 支持 `rate_limit` 配置（每秒请求数），通过令牌桶限流
3. 不兼容旧数据

## 详细设计

### 1. 模型标识：provider/model

**配置文件（models.yaml）：**

```yaml
providers:
  glm:
    api_key: "..."
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
    rate_limit: 2              # 可选，每秒请求数，不配置则不限流
    models:
      glm-4.7: { max_tokens: 4096 }
      glm-4-flash: {}

  openai:
    api_key: "..."
    api_base: "https://api.openai.com/v1/"
    models:
      gpt-4: { max_tokens: 4096 }
```

**CLI 用法：**

```bash
uv run python -m benchmark evaluate --model glm/glm-4.7 --dimension reasoning
```

**config.py 改动：**

`get_model_config(model_name)` 解析逻辑：
- 检测 `model_name` 是否包含 `/`
- 包含：拆分为 `provider_name` 和 `model_name`，直接定位
- 不包含：抛出 ValueError 提示使用 `provider/model` 格式

返回值增加 `provider` 和 `rate_limit` 字段：

```python
{
    "provider": "glm",
    "api_key": "...",
    "api_base": "...",
    "max_tokens": 4096,
    "rate_limit": 2.0,  # 新增，未配置时为 None
}
```

### 2. 令牌桶限流

**新建文件：`benchmark/core/rate_limiter.py`**

```python
class TokenBucketRateLimiter:
    """令牌桶限流器。rate 表示每秒允许的请求数。"""

    def __init__(self, rate: float):
        self._rate = rate
        self._tokens = rate        # 桶容量 = rate
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
            # 计算等待时间
            wait = (1 - self._tokens) / self._rate
        logger.warning("Rate limited, waiting %.1fs", wait)
        time.sleep(wait)
        with self._lock:
            self._refill()
            self._tokens -= 1  # 消耗等待补充的令牌

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now
```

**集成到 LLMEvalAdapter：**

- 新增类变量 `_provider_limiters: dict[str, TokenBucketRateLimiter]` 缓存每个 provider 的限流器
- `__init__` 中：如果配置有 `rate_limit`，创建或复用该 provider 的 limiter
- `generate()` 中：HTTP 请求前调用 `self._limiter.acquire()`

### 3. 影响范围

| 文件 | 改动 |
|------|------|
| `benchmark/config.py` | `get_model_config()` 解析 `provider/model` 格式 |
| `benchmark/core/rate_limiter.py` | 新建，令牌桶实现 |
| `benchmark/core/llm_adapter.py` | 集成限流器 |
| `benchmark/cli.py` | `--model` help 文案，移除旧的 `generate(prompt, model)` 双参数中 model 的裸名用法 |
| `benchmark/configs/models.yaml.example` | 添加 `rate_limit` 字段 |
| `quickstart.md` | 更新命令示例为 `provider/model` 格式 |

### 4. 错误处理

- `provider/model` 格式错误（缺少 `/` 或 provider 不存在）：抛出 ValueError，明确提示可用格式
- `rate_limit` 为 0 或负数：配置加载时报错
- `rate_limit` 未配置：不限流，limiter 为 None
