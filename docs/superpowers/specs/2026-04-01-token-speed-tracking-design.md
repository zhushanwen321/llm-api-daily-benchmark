# Token 生成速度追踪

## 背景

当前 benchmark 系统仅记录每次评测的 `execution_time`（端到端耗时），没有记录 API 返回的 token 用量信息。token 生成速度（completion_tokens / duration）是衡量模型 API 性能的重要指标，需要持久化记录并展示。

## 目标

- 记录每次 API 调用的 token 用量和生成速度
- 数据关联到具体的 benchmark 结果，方便后续统计
- UI 页面展示 token 速度指标

## 设计

### 数据层：新表 api_call_metrics

独立于 `eval_results`，通过 `result_id` 一对一关联。

```sql
CREATE TABLE IF NOT EXISTS api_call_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id TEXT UNIQUE NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    duration REAL NOT NULL DEFAULT 0,
    tokens_per_second REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (result_id) REFERENCES eval_results(result_id)
)
```

新增 Pydantic 模型 `ApiCallMetrics`，字段与表列对应。

### LLM Adapter 层

`llm_adapter.py` 的 `generate()` 方法当前返回 `str`，改为返回 `GenerateResponse` 对象：

```python
class GenerateResponse:
    content: str           # 模型输出文本
    prompt_tokens: int     # API usage.prompt_tokens
    completion_tokens: int # API usage.completion_tokens
```

从 API 响应 `data["usage"]` 中提取 token 用量，不再丢弃。`usage` 字段不存在时（部分 API 可能不返回），默认为 0。

### CLI 层

`cli.py` 的 `evaluate` 命令中，每次 API 调用后：
1. 从 `GenerateResponse` 获取 `completion_tokens`
2. 用已有的 `execution_time` 计算 `tokens_per_second`
3. 保存 `EvalResult` 后，额外写入 `api_call_metrics` 记录

### UI 层

`app.py` 中：
- 结果表格新增 "Token Speed" 列，显示格式如 `45.2 tok/s`
- 详情页新增 metric 卡片展示 prompt_tokens、completion_tokens、tokens_per_second

### 不变的部分

- `eval_results` 表结构不变，不破坏现有数据
- `EvalResult` schema 不变
- 评分逻辑不变

## 涉及文件

| 文件 | 改动 |
|---|---|
| `benchmark/core/llm_adapter.py` | 新增 `GenerateResponse`，`generate()` 返回类型改为 `GenerateResponse`，提取 `usage` |
| `benchmark/models/schemas.py` | 新增 `ApiCallMetrics` 模型 |
| `benchmark/models/database.py` | 新增 `api_call_metrics` 表，新增 `save_metrics()` 方法 |
| `benchmark/cli.py` | 适配 `GenerateResponse`，保存 token 速度记录 |
| `benchmark/visualization/app.py` | 表格新增 token speed 列，详情页新增 metric 卡片 |
