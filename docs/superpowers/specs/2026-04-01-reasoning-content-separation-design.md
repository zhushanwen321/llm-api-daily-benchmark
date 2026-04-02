# reasoning_content / content 分离采集设计

## 背景

当前 `llm_adapter.py` 在 SSE 流式解析中用 `or` 合并了 `reasoning_content` 和 `content`：

```python
delta_content = delta.get("content") or delta.get("reasoning_content")
```

导致推理过程和最终回答混入同一个字符串，无法区分。后置的 `response_parser.py` 只能用正则猜测 `<think >` 标签来拆分，不可靠。

## 目标

1. 在 API 调用层原生分离 `reasoning_content` 和 `content`
2. 支持不同 provider 的思考模式开启方式和字段名差异
3. 采集完整指标：TTFT-R、TTFT-C、reasoning_tokens
4. 简化下游链路（prompt_builder、response_parser）

## 涉及文件

| 文件 | 操作 |
|------|------|
| `benchmark/models/schemas.py` | 修改：GenerateResponse / ApiCallMetrics 增加字段 |
| `benchmark/core/llm_adapter.py` | 修改：分离收集 reasoning/content，注入 thinking 参数 |
| `benchmark/configs/models.yaml` | 修改：模型级 thinking 配置 |
| `benchmark/core/prompt_builder.py` | 修改：简化 JSON schema |
| `benchmark/core/response_parser.py` | 修改：去掉 strip_think_tags，简化解析 |
| `benchmark/models/database.py` | 修改：api_call_metrics 增加列 |
| `benchmark/cli.py` | 修改：优先使用 adapter 层返回的 reasoning_content |
| `benchmark/visualization/app.py` | 修改：从 metrics 表读取 reasoning_content 展示 |

## 设计细节

### 1. 数据模型变更

#### GenerateResponse

```python
class GenerateResponse(BaseModel):
    content: str                    # 最终回答（原 content）
    reasoning_content: str = ""     # 推理过程
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0       # 推理 token 数
    duration: float = 0.0
    tokens_per_second: float = 0.0
    ttft: float = 0.0               # 首 token 延迟（TTFT-R，含推理）
    ttft_content: float = 0.0       # 首 content token 延迟（TTFT-C）
    truncated: bool = False
    finish_reason: str = ""
```

#### ApiCallMetrics

```python
class ApiCallMetrics(BaseModel):
    result_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0       # 新增
    reasoning_content: str = ""     # 新增
    duration: float = 0.0
    tokens_per_second: float = 0.0
    ttft_content: float = 0.0       # 新增
    created_at: datetime
```

#### api_call_metrics 表增加列

```sql
reasoning_tokens INTEGER DEFAULT 0,
reasoning_content TEXT DEFAULT '',
ttft_content REAL DEFAULT 0.0
```

### 2. llm_adapter.py 流式解析重构

#### SSE delta 解析

替换原来的单 `content_parts` 列表，改为双列表分别收集：

```python
reasoning_parts: list[str] = []
content_parts: list[str] = []
t_first_token: float | None = None          # TTFT-R
t_first_content_token: float | None = None  # TTFT-C

# 根据模型配置确定 reasoning 字段名
thinking_cfg = cfg.get("thinking", {})
reasoning_field = thinking_cfg.get("reasoning_field", "reasoning_content")

delta_reasoning = delta.get(reasoning_field)
delta_content = delta.get("content")

if delta_reasoning:
    reasoning_parts.append(delta_reasoning)
    if t_first_token is None:
        t_first_token = time.monotonic()

if delta_content:
    content_parts.append(delta_content)
    if t_first_content_token is None:
        t_first_content_token = time.monotonic()
```

#### payload 注入 thinking 参数

```python
payload = {
    "model": ...,
    "messages": [...],
    "temperature": ...,
    "max_tokens": ...,
    "stream": True,
    "stream_options": {"include_usage": True},
}
if thinking_cfg.get("enabled") and thinking_cfg.get("request_params"):
    payload.update(thinking_cfg["request_params"])
```

#### usage 提取 reasoning_tokens

```python
details = usage.get("completion_tokens_details", {})
reasoning_tokens = details.get("reasoning_tokens", 0)
```

部分 provider（如 Kimi、GLM）不返回 `completion_tokens_details`，此时 `reasoning_tokens` 为 0。

#### 返回值

```python
return GenerateResponse(
    content="".join(content_parts),
    reasoning_content="".join(reasoning_parts),
    reasoning_tokens=reasoning_tokens,
    ttft=t_first_token - t_start if t_first_token else 0.0,
    ttft_content=t_first_content_token - t_start if t_first_content_token else 0.0,
    ...
)
```

同步 `generate()` 和异步 `agenerate()` 逻辑一致，同步改两次。

### 3. models.yaml 模型级 thinking 配置

每个推理模型配置完整的 `thinking` 块，非推理模型不配：

```yaml
providers:
  zai:
    models:
      glm-4.7: {}                      # 非推理模型，不配置 thinking
      glm-5:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
      glm-5.1:
        thinking:
          enabled: true
          reasoning_field: reasoning_content

  minimax:
    models:
      MiniMax-M2.7:
        thinking:
          enabled: true
          reasoning_field: reasoning_details
          request_params:
            reasoning_split: true
      MiniMax-M2.5-highspeed:
        thinking:
          enabled: true
          reasoning_field: reasoning_details
          request_params:
            reasoning_split: true

  kimi:
    models:
      kimi-for-coding:
        thinking:
          enabled: true
          reasoning_field: reasoning_content

  opencode-go-gk:
    models:
      glm-5:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
      kimi-k2.5:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
```

### 4. prompt_builder.py 简化

去掉 JSON schema 中的 `reasoning`/`explanation` 字段要求，因为推理过程已由 API 层 `reasoning_content` 天然提供。

**reasoning 维度**：
```json
{"answer": "42"}
```

**backend-dev 维度**：
```json
{"code": "def task_func(...):\n    ..."}
```

### 5. response_parser.py 简化

去掉 `strip_think_tags()`——adapter 层已分离，不需要正则匹配 `<think >` 标签。

简化后 `parse_response` 流程：
1. 尝试 `extract_json_object()` 提取 JSON
2. JSON 成功 → 按 dimension 提取 answer/code 字段
3. JSON 失败 + backend-dev → 尝试 `extract_python_code()` 提取代码块
4. 最终 fallback → 原文整体作为 answer

### 6. cli.py 调整

```python
gen_response = await llm.agenerate(task.prompt, model=model)

# 推理内容直接从 API 层获取
think_content = gen_response.reasoning_content

# content 仍需解析（可能包含 JSON 格式）
parsed = parse_response(gen_response.content, task.dimension)

result = EvalResult(
    model_output=gen_response.content,
    model_think=think_content,
    model_answer=parsed.answer,
    ...
)
```

### 7. 前端 app.py 适配

Thinking 折叠区优先从 `api_call_metrics.reasoning_content` 读取，fallback 到 `eval_results.model_think`。

### 8. DB 迁移

`_init_db()` 检测 `api_call_metrics` 是否缺少 `reasoning_tokens` 列，若缺少则 drop `api_call_metrics` 表重建（不影响 `eval_results` 和 `eval_runs`）。

## 不在范围内

- OpenAI o1/o3 的 `reasoning.effort` 参数支持（当前 provider 无 OpenAI 推理模型）
- Anthropic 的 `thinking.budget` 参数
- 非推理模型的 reasoning_content 处理（不配置 thinking 块的模型走原逻辑）
