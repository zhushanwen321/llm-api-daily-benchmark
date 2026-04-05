# 镜像精简 + 流量控制优化设计

日期：2026-04-05
状态：草案

## 背景

### 镜像体积

多阶段构建后镜像仍有 953 MB。分析 site-packages 发现三大块占 583 MB：

| 包 | 大小 | 实际使用 |
|---|---|---|
| pyarrow | 149 MB | `datasets.load_dataset()` 的内部存储格式，未直接使用 |
| scipy | 170 MB | 仅用 `t.ppf()` 和 `ttest_ind()` |
| numpy | 98 MB | 仅在 bootstrap 中用 `np.array/mean/var/percentile` |
| sympy | 74 MB | math_scorer 的 fallback，已 try/except 条件导入 |
| pandas | 72 MB | 仅 `pd.read_sql_query()` + `pd.notna()` |
| pydeck | 15 MB | streamlit 间接依赖，未使用 |

### 流量控制

当前 `agenerate()` 用 `Semaphore(rate_limit)` 做并发控制。存在两个问题：

1. **语义混乱**：`rate_limit` 同时表示"速率"和"并发数"，但 LLM 调用是 streaming（30-60s），这两个概念不等价
2. **TokenBucket 仅用于 sync 路径**：async 路径不走 TokenBucket，sync 路径的 `time.sleep` 阻塞线程

评测场景中所有调用走 async 路径（`agenerate`），Semaphore 控制的是"同时在飞的流数"。stream 结束后 semaphore 释放，下一个等待的协程立即获得——这是正确的行为，不需要额外的速率控制。

## 设计目标

1. 镜像体积从 953 MB 降至 ~370 MB
2. 流量控制语义清晰：只用 `max_concurrency` 控制 provider 级并发流数
3. 评测结果数值不变（同样的输入产生同样的分数）

---

## Part 1：流量控制优化

### 1.1 移除 TokenBucketRateLimiter

删除 `benchmark/core/rate_limiter.py` 中的 `TokenBucketRateLimiter` 类。

理由：sync 路径不用于评测主流程，TokenBucket 的 `time.sleep` 在 async 环境下无意义。

### 1.2 新增 AsyncConcurrencyLimiter

```python
# benchmark/core/concurrency.py

class AsyncConcurrencyLimiter:
    """provider 级并发流控制器"""

    _instances: dict[str, AsyncConcurrencyLimiter] = {}

    def __init__(self, max_concurrency: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)

    @classmethod
    def get_or_create(cls, provider: str, max_concurrency: int) -> AsyncConcurrencyLimiter:
        if provider not in cls._instances:
            cls._instances[provider] = cls(max_concurrency)
        return cls._instances[provider]

    async def acquire(self):
        await self._sem.acquire()

    def release(self):
        self._sem.release()
```

使用方式（在 `agenerate` 中）：

```python
async def agenerate(self, ...):
    limiter = AsyncConcurrencyLimiter.get_or_create(provider, max_concurrency)
    await limiter.acquire()
    try:
        return await self._do_agenerate(...)
    finally:
        limiter.release()
```

### 1.3 配置变更

models.yaml 中 `rate_limit` 重命名为 `max_concurrency`：

```yaml
providers:
  glm:
    api_key: "${GLM_API_KEY}"
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
    max_concurrency: 2   # 同一 provider 最多 2 个并发流
```

`config.py` 中兼容处理：读到 `rate_limit` 时映射到 `max_concurrency`，打印 deprecation warning。

### 1.4 LLMEvalAdapter 变更

- 删除 `_provider_limiters`（TokenBucket）和 `_provider_semaphores`
- 新增 `_provider_concurrency`：`dict[str, AsyncConcurrencyLimiter]`
- `generate()`（sync 路径）：使用 `threading.Semaphore(max_concurrency)` 代替 TokenBucket
- `agenerate()`（async 路径）：使用 `AsyncConcurrencyLimiter`
- 删除对 `rate_limiter.py` 的 import

---

## Part 2：镜像精简

### 2.1 datasets + pyarrow → requests + json（-154 MB）

5 个 adapter（MATH, GSM8K, MMLU, MMLU-Pro, BigCodeBench）都用 `datasets.load_dataset()` 加载数据。

替换方案：
- 用 `requests.get()` 直接调用 HuggingFace API 下载原始数据文件
- HuggingFace 数据集的原始文件可通过 `https://huggingface.co/datasets/{repo}/resolve/{branch}/{path}` 获取
- 用标准库 `json.loads()` 解析
- 保留 `cache_dir` 离线缓存机制：下载后存为本地 JSON，优先读本地缓存

每个 adapter 的 `load()` 接口不变，返回 `list[TaskDefinition]`。

新增工具函数：

```python
# benchmark/adapters/hf_loader.py

def load_hf_dataset(
    repo: str,
    split: str,
    cache_dir: str,
    *,
    config_name: str | None = None,
    filename: str | None = None,
) -> list[dict]:
    """从 HuggingFace 下载数据集文件，带本地 JSON 缓存"""
```

需要为每个数据集确认其 HuggingFace 上的原始文件路径（parquet 或 jsonl），这需要在实现阶段逐个确认。

### 2.2 scipy → 纯 Python（-170 MB）

`statistics.py` 中仅用 `scipy.stats.t.ppf()`，用 `math.lgamma` 近似 t 分布 ppf：

```python
import math

def _t_ppf(p: float, df: int) -> float:
    """t 分布百分位数的近似实现"""
    # 基于 Cornish-Fisher 展开或正态近似
    # 对 df >= 15（评测场景），误差 < 0.001
```

`advanced_statistics.py` 中 bootstrap 和 t-test 用纯 Python 重写：
- `np.random.default_rng.choice` → `random.choices`
- `np.mean` → `statistics.mean`
- `np.var` → `statistics.variance`
- `np.percentile` → 排序后取分位数
- `scipy.stats.ttest_ind` → 手动计算 t 统计量 + 上述 `_t_ppf` 得 p-value

### 2.3 numpy 移除（-98 MB）

随 scipy 一起移除。所有 `np.array/mean/var` 调用替换为 `statistics` 模块。

### 2.4 sympy 移除（-74 MB）

`math_scorer.py` 中 `_try_sympy_match()` 已是 try/except 条件导入的 fallback。删除该函数和调用点，只保留字符串精确匹配 + 数值比较。

### 2.5 pandas → sqlite3 + list[dict]（-72 MB）

`app.py` 中两处使用：
1. `pd.read_sql_query()` → `sqlite3` 原生查询，返回 `list[sqlite3.Row]`，转为 `list[dict]`
2. `pd.notna(x)` → `x is not None`
3. `st.dataframe(df)` → `st.dataframe(list_of_dict)`（Streamlit 原生支持）

### 2.6 pydeck 清理（-15 MB）

在 Dockerfile 的运行阶段删除未使用的 pydeck：

```dockerfile
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip \
    && rm -rf /usr/local/lib/python3.13/site-packages/pydeck*
```

### 2.7 pyproject.toml 变更

```toml
dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
    "streamlit>=1.28",
    "pyyaml>=6.0",
    "requests>=2.31",
    "httpx>=0.27",
    "python-dotenv>=1.0",
    "matplotlib>=3.7",
    "jinja2>=3.1",
    "apscheduler>=3.10",
]
```

移除：`datasets`, `pandas`, `scipy`, `sympy`

---

## Part 3：数据兼容性保证

### 3.1 数值精度

t 分布 ppf 近似在 df >= 15 时误差 < 0.001，对置信区间的实际影响可忽略（置信区间本身就只是参考指标）。

bootstrap 的随机种子固定为 42，用 `random.Random(42)` 替换 `np.random.default_rng(42)` 后，结果会有微小差异（不同的随机数生成器），但统计结论不变。

### 3.2 离线缓存

新的 `hf_loader.py` 维护与当前 `datasets` 库相同的离线缓存语义：
- 缓存文件存在 → 直接读取，不发起网络请求
- 配合现有 `benchmark/datasets/.download-complete` 标志位

---

## 文件改动清单

| 文件 | 变更类型 |
|---|---|
| `benchmark/core/rate_limiter.py` | 删除 |
| `benchmark/core/concurrency.py` | 新增 |
| `benchmark/core/llm_adapter.py` | 重写并发控制 |
| `benchmark/config.py` | 读取 `max_concurrency`，兼容 `rate_limit` |
| `benchmark/core/statistics.py` | 移除 scipy，纯 Python |
| `benchmark/core/advanced_statistics.py` | 移除 numpy/scipy，纯 Python |
| `benchmark/scorers/math_scorer.py` | 删除 sympy fallback |
| `benchmark/adapters/hf_loader.py` | 新增 |
| `benchmark/adapters/math_adapter.py` | 用 hf_loader 替换 datasets |
| `benchmark/adapters/gsm8k_adapter.py` | 用 hf_loader 替换 datasets |
| `benchmark/adapters/mmlu_adapter.py` | 用 hf_loader 替换 datasets |
| `benchmark/adapters/mmlu_pro_adapter.py` | 用 hf_loader 替换 datasets |
| `benchmark/adapters/bigcodebench_adapter.py` | 用 hf_loader 替换 datasets |
| `benchmark/visualization/app.py` | 移除 pandas，用 sqlite3 |
| `pyproject.toml` | 移除 4 个依赖 |
| `configs/models.yaml.example` | `rate_limit` → `max_concurrency` |
| `Dockerfile` | 添加 pydeck 清理 |

## 不在范围内

- 同步路径 `generate()` 的完整重写（仅最小化改动以适配新并发控制）
- Streamlit 可视化功能扩展
- 跨 provider 负载均衡 / 故障转移
- scheduler 调度逻辑变更
