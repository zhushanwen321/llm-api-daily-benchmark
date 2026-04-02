# reasoning_content / content 分离采集 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 API 调用层原生分离 reasoning_content 和 content，支持不同 provider 的思考模式配置，采集 TTFT-R/TTFT-C/reasoning_tokens 完整指标。

**Architecture:** models.yaml 增加模型级 `thinking` 配置块 → config.py 传递 thinking 配置到 adapter → llm_adapter.py 分离收集 reasoning/content → 全链路传递到 DB 和前端。prompt_builder 和 response_parser 简化（不再需要正则猜测拆分 think 标签）。

**Tech Stack:** Python 3.12+, httpx/requests (SSE streaming), SQLite, Pydantic, Streamlit

---

## File Structure

| 文件 | 职责 | 操作 |
|------|------|------|
| `benchmark/models/schemas.py` | GenerateResponse、ApiCallMetrics 增加字段 | 修改 |
| `benchmark/config.py` | get_model_config() 传递 thinking 配置 | 修改 |
| `benchmark/configs/models.yaml` | 模型级 thinking 配置 | 修改 |
| `benchmark/core/llm_adapter.py` | 分离 reasoning/content 收集 + 注入 thinking 参数 | 修改 |
| `benchmark/core/prompt_builder.py` | 简化 JSON schema（去掉 reasoning/explanation） | 修改 |
| `benchmark/core/response_parser.py` | 去掉 strip_think_tags，简化解析 | 修改 |
| `benchmark/models/database.py` | api_call_metrics 增加列 + 迁移 | 修改 |
| `benchmark/cli.py` | 优先使用 adapter 返回的 reasoning_content | 修改 |
| `benchmark/visualization/app.py` | 从 metrics 表读取 reasoning_content 展示 | 修改 |

---

### Task 1: schemas.py — 增加 reasoning 相关字段

**Files:**
- Modify: `benchmark/models/schemas.py`

- [ ] **Step 1: 给 GenerateResponse 增加 3 个字段**

在 `GenerateResponse` 类中，在 `content: str` 后增加 `reasoning_content`，在 `completion_tokens` 后增加 `reasoning_tokens`，在 `ttft` 后增加 `ttft_content`：

```python
class GenerateResponse(BaseModel):
    """LLM API 调用响应，包含文本和 token 用量。"""

    content: str
    reasoning_content: str = ""   # 推理过程（从 API reasoning_content 字段获取）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0     # 推理 token 数（OpenAI usage.completion_tokens_details）
    duration: float = 0.0
    tokens_per_second: float = 0.0
    ttft: float = 0.0             # 首 token 延迟（TTFT-R，含推理）
    ttft_content: float = 0.0     # 首 content token 延迟（TTFT-C，用户感知延迟）
    truncated: bool = False
    finish_reason: str = ""
```

- [ ] **Step 2: 给 ApiCallMetrics 增加 3 个字段**

在 `ApiCallMetrics` 类中，在 `completion_tokens` 后增加 `reasoning_tokens`，在 `duration` 前增加 `reasoning_content`，在 `tokens_per_second` 后增加 `ttft_content`：

```python
class ApiCallMetrics(BaseModel):
    """单次 API 调用的 token 指标。"""

    result_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_content: str = ""
    duration: float = 0.0
    tokens_per_second: float = 0.0
    ttft_content: float = 0.0
    created_at: datetime
```

- [ ] **Step 3: 验证 schema 加载正常**

Run: `.venv/bin/python -c "from benchmark.models.schemas import GenerateResponse, ApiCallMetrics; print('schemas OK')"`

Expected: `schemas OK`

- [ ] **Step 4: Commit**

```bash
git add benchmark/models/schemas.py
git commit -m "feat(schemas): add reasoning_content, reasoning_tokens, ttft_content fields"
```

---

### Task 2: config.py — 传递 thinking 配置

**Files:**
- Modify: `benchmark/config.py:49-99`

- [ ] **Step 1: 修改 get_model_config() 返回 thinking 配置**

在 `get_model_config()` 函数的返回字典中增加 `thinking` 字段。将第 93-99 行的 return 语句改为：

```python
    model_cfg = models[model_id] or {}
    # 读取全局默认值，默认为 131072
    defaults = cfg.get("defaults", {})
    default_max_tokens = defaults.get("max_tokens", 131072)
    return {
        "provider": provider_name,
        "api_key": provider_cfg["api_key"],
        "api_base": provider_cfg["api_base"],
        "max_tokens": model_cfg.get("max_tokens", default_max_tokens),
        "rate_limit": float(provider_cfg["rate_limit"]) if "rate_limit" in provider_cfg else None,
        "thinking": model_cfg.get("thinking", {}),
    }
```

改动说明：仅增加一行 `"thinking": model_cfg.get("thinking", {})`，其余不变。

- [ ] **Step 2: 验证配置加载**

Run: `.venv/bin/python -c "from benchmark.config import get_model_config; c = get_model_config('zai/glm-4.7'); print(c.get('thinking', 'MISSING'))"`

Expected: `{}`（glm-4.7 没有 thinking 配置，返回空 dict）

- [ ] **Step 3: Commit**

```bash
git add benchmark/config.py
git commit -m "feat(config): pass thinking config from models.yaml to adapter"
```

---

### Task 3: models.yaml — 模型级 thinking 配置

**Files:**
- Modify: `benchmark/configs/models.yaml`

- [ ] **Step 1: 为每个推理模型添加 thinking 配置块**

将 `zai/models` 下的 glm-5 和 glm-5.1 改为：

```yaml
      glm-5:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
      glm-5.1:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
```

将 `minimax/models` 下所有模型改为：

```yaml
      MiniMax-M2.7:
        thinking:
          enabled: true
          reasoning_field: reasoning_details
          request_params:
            reasoning_split: true
      MiniMax-M2.7-highspeed: {}
      MiniMax-M2.5:
        thinking:
          enabled: true
          reasoning_field: reasoning_details
          request_params:
            reasoning_split: true
      MiniMax-M2.5-highspeed: {}
```

将 `kimi/models` 下改为：

```yaml
      kimi-for-coding:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
```

将 `opencode-go-gk/models` 下改为：

```yaml
      glm-5:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
      kimi-k2.5:
        thinking:
          enabled: true
          reasoning_field: reasoning_content
```

将 `opencode-go-minimax/models` 下改为：

```yaml
      minimax-m2.7:
        thinking:
          enabled: true
          reasoning_field: reasoning_details
          request_params:
            reasoning_split: true
      minimax-m2.5:
        thinking:
          enabled: true
          reasoning_field: reasoning_details
          request_params:
            reasoning_split: true
```

glm-4.7、MiniMax-M2.7-highspeed、MiniMax-M2.5-highspeed 不配置 thinking（非推理模型）。

- [ ] **Step 2: 验证配置加载**

Run: `.venv/bin/python -c "from benchmark.config import get_model_config; c = get_model_config('kimi/kimi-for-coding'); print(c['thinking'])"`

Expected: `{'enabled': True, 'reasoning_field': 'reasoning_content'}`

- [ ] **Step 3: Commit**

```bash
git add benchmark/configs/models.yaml
git commit -m "feat(config): add per-model thinking config for reasoning models"
```

---

### Task 4: llm_adapter.py — 分离 reasoning/content 收集

**Files:**
- Modify: `benchmark/core/llm_adapter.py`

这是最核心的改动。同步 `generate()` 和异步 `agenerate()` 都需要修改。两者逻辑完全一致。

- [ ] **Step 1: 修改 generate() 的 payload 构造（第 134-141 行）**

在第 141 行 `"stream_options": {"include_usage": True},` 之后，增加 thinking 参数注入：

```python
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # 注入模型级 thinking 参数（如 MiniMax 的 reasoning_split）
        thinking_cfg = cfg.get("thinking", {})
        if thinking_cfg.get("enabled") and thinking_cfg.get("request_params"):
            payload.update(thinking_cfg["request_params"])
```

- [ ] **Step 2: 修改 generate() 的 SSE 解析（第 168-176 行）**

将 `content_parts` 初始化部分改为双列表：

```python
                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                usage: dict[str, Any] = {}
                t_start = time.monotonic()
                t_first_token: float | None = None
                t_first_content_token: float | None = None
                t_last_chunk = t_start
                t_last_activity = t_start
                got_done = False
                truncated = False
                content_filtered = False
                final_finish_reason = ""
                chunk_count = 0
                reasoning_field = thinking_cfg.get("reasoning_field", "reasoning_content")
```

- [ ] **Step 3: 修改 generate() 的 delta 处理（第 234-249 行）**

将原来的：

```python
                            delta = choices[0].get("delta", {})
                            delta_content = delta.get("content") or delta.get(
                                "reasoning_content"
                            )
                            if delta_content:
                                content_parts.append(delta_content)
                                t_last_chunk = time.monotonic()
                                if t_first_token is None:
                                    t_first_token = t_last_chunk
```

改为：

```python
                            delta = choices[0].get("delta", {})
                            delta_reasoning = delta.get(reasoning_field)
                            delta_content = delta.get("content")
                            if delta_reasoning:
                                reasoning_parts.append(delta_reasoning)
                                t_last_chunk = time.monotonic()
                                t_last_activity = t_last_chunk
                                if t_first_token is None:
                                    t_first_token = t_last_chunk
                            if delta_content:
                                content_parts.append(delta_content)
                                t_last_chunk = time.monotonic()
                                t_last_activity = t_last_chunk
                                if t_first_content_token is None:
                                    t_first_content_token = t_last_chunk
```

- [ ] **Step 4: 修改 generate() 的日志中 content 片段计数**

将所有 `len(content_parts)` 用于"进度日志"的地方改为同时报告 reasoning 和 content 数量。例如第 227-232 行的进度日志改为：

```python
                        if chunk_count % 50 == 0:
                            logger.debug(
                                f"[{model}] 流进度: chunk#{chunk_count}, "
                                f"reasoning 片段: {len(reasoning_parts)}, "
                                f"content 片段: {len(content_parts)}, "
                                f"已耗时 {(time.monotonic() - t_start):.1f}s"
                            )
```

其余日志中 `len(content_parts)` 改为 `content 长度={len(''.join(content_parts))}, reasoning 长度={len(''.join(reasoning_parts))}` 的形式。

- [ ] **Step 5: 修改 generate() 的 full_content 拼接和返回值（第 298-360 行）**

在 `full_content = "".join(content_parts)` 后面增加：

```python
                full_reasoning = "".join(reasoning_parts)
```

将返回值改为：

```python
                return GenerateResponse(
                    content=full_content,
                    reasoning_content=full_reasoning,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                    duration=duration,
                    tokens_per_second=tokens_per_second,
                    ttft=ttft,
                    ttft_content=t_first_content_token - t_start if t_first_content_token else 0.0,
                    truncated=truncated,
                    finish_reason=final_finish_reason,
                )
```

注意：`ttft` 保持为 TTFT-R（首 reasoning token 延迟）。`ttft_content` 是 TTFT-C（首 content token 延迟）。

- [ ] **Step 6: 对 agenerate() 的 _do_agenerate() 方法做完全相同的修改**

`_do_agenerate()` (第 457 行起) 与 `generate()` 逻辑完全一致，需要做相同的 5 处修改：

1. payload 注入 thinking 参数（在第 492 行后）
2. 初始化双列表 + reasoning_field 变量（第 520-530 行）
3. delta 分离处理（第 594-607 行）
4. 进度日志修改
5. 返回值增加 reasoning_content、reasoning_tokens、ttft_content

每处修改的代码与 generate() 对应位置完全相同，只是上下文不同（async with / async for vs with / for）。

- [ ] **Step 7: 验证 adapter 可正常实例化**

Run: `.venv/bin/python -c "from benchmark.core.llm_adapter import LLMEvalAdapter; llm = LLMEvalAdapter(model='kimi/kimi-for-coding'); print('adapter OK')"`

Expected: `adapter OK`

- [ ] **Step 8: Commit**

```bash
git add benchmark/core/llm_adapter.py
git commit -m "feat(adapter): separate reasoning_content/content collection with TTFT-R/C metrics"
```

---

### Task 5: database.py — api_call_metrics 增加列

**Files:**
- Modify: `benchmark/models/database.py`

- [ ] **Step 1: 修改 _init_db() 增加列检测和建表语句**

在 `_init_db()` 中，找到现有的列检测逻辑（第 61-66 行检查 model_think），在其后增加对 `api_call_metrics.reasoning_tokens` 列的检测：

```python
        # 检查 eval_results 是否缺少 model_think 列，若缺少则 drop 重建
        needs_rebuild = False
        try:
            cursor.execute("SELECT model_think FROM eval_results LIMIT 1")
        except sqlite3.OperationalError:
            needs_rebuild = True

        if needs_rebuild:
            cursor.execute("DROP TABLE IF EXISTS api_call_metrics")
            cursor.execute("DROP TABLE IF EXISTS eval_results")
            cursor.execute("DROP TABLE IF EXISTS eval_runs")

        # 检查 api_call_metrics 是否缺少新列，若缺少则 drop 重建该表
        metrics_needs_rebuild = False
        try:
            cursor.execute("SELECT reasoning_tokens FROM api_call_metrics LIMIT 1")
        except sqlite3.OperationalError:
            metrics_needs_rebuild = True

        if metrics_needs_rebuild:
            cursor.execute("DROP TABLE IF EXISTS api_call_metrics")
```

- [ ] **Step 2: 修改 api_call_metrics 建表语句**

将 `api_call_metrics` 的 CREATE TABLE 改为：

```sql
            CREATE TABLE IF NOT EXISTS api_call_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id TEXT UNIQUE NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_content TEXT DEFAULT '',
                duration REAL NOT NULL DEFAULT 0,
                tokens_per_second REAL NOT NULL DEFAULT 0,
                ttft_content REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (result_id) REFERENCES eval_results(result_id)
            )
```

新增 3 列：`reasoning_tokens`、`reasoning_content`、`ttft_content`。

- [ ] **Step 3: 修改 save_metrics() 方法**

将 `save_metrics()` 的 INSERT 语句改为：

```python
    def save_metrics(self, metrics: ApiCallMetrics) -> str:
        """保存 API 调用的 token 指标。"""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO api_call_metrics
               (result_id, prompt_tokens, completion_tokens,
                reasoning_tokens, reasoning_content,
                duration, tokens_per_second, ttft_content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                metrics.result_id,
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.reasoning_tokens,
                metrics.reasoning_content,
                metrics.duration,
                metrics.tokens_per_second,
                metrics.ttft_content,
                metrics.created_at.isoformat(),
            ),
        )
        conn.commit()
        return metrics.result_id
```

- [ ] **Step 4: 删除旧 DB 并验证建表**

```bash
rm -f benchmark/data/results.db
.venv/bin/python -c "
from benchmark.models.database import Database
import sqlite3
db = Database()
conn = sqlite3.connect('benchmark/data/results.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(api_call_metrics)').fetchall()]
print('columns:', cols)
assert 'reasoning_tokens' in cols
assert 'reasoning_content' in cols
assert 'ttft_content' in cols
print('OK')
db.close()
"
```

Expected: `columns: [...]` 包含 `reasoning_tokens`, `reasoning_content`, `ttft_content`，最后输出 `OK`

- [ ] **Step 5: Commit**

```bash
git add benchmark/models/database.py
git commit -m "feat(db): add reasoning_tokens, reasoning_content, ttft_content columns"
```

---

### Task 6: prompt_builder.py — 简化 JSON schema

**Files:**
- Modify: `benchmark/core/prompt_builder.py`

- [ ] **Step 1: 去掉 reasoning/explanation 字段**

将 `_REASONING_SCHEMA` 改为：

```python
_REASONING_SCHEMA = {
    "example": {"answer": "42"},
    "fields": {
        "answer": "（字符串）最终数值答案，纯数字，不含单位或其他文字",
    },
}
```

将 `_BACKEND_DEV_SCHEMA` 改为：

```python
_BACKEND_DEV_SCHEMA = {
    "example": {
        "code": "import ...\n\ndef task_func(...):\n    ...",
    },
    "fields": {
        "code": "（字符串）完整的、可直接执行的 Python 代码，不包含 markdown 标记",
    },
}
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/core/prompt_builder.py
git commit -m "refactor(prompt): simplify JSON schema, remove reasoning/explanation fields"
```

---

### Task 7: response_parser.py — 简化解析

**Files:**
- Modify: `benchmark/core/response_parser.py`

- [ ] **Step 1: 去掉 strip_think_tags 和相关正则，简化 parse_response**

删除 `_THINK_CLOSED_RE`、`_THINK_OPEN_RE` 两个正则和 `strip_think_tags()` 函数。

将 `parse_response()` 简化为：

```python
def parse_response(raw: str, dimension: str) -> ParsedResponse:
    """解析模型 content（已由 adapter 分离 reasoning_content），提取最终答案.

    Args:
        raw: 模型的 content 部分（不含 reasoning_content）
        dimension: 评测维度（"reasoning" 或 "backend-dev"）

    Returns:
        ParsedResponse，think 固定为空（推理内容由 adapter 层分离）
    """
    if not raw:
        return ParsedResponse(think="", answer="")

    # Step 1: 尝试 JSON 解析
    json_data = extract_json_object(raw)
    if json_data:
        answer = _extract_answer_from_json(json_data, dimension)
        return ParsedResponse(think="", answer=answer)

    # Step 2: JSON 解析失败的 fallback
    if dimension == "backend-dev":
        code = extract_python_code(raw)
        if code:
            return ParsedResponse(think="", answer=code)

    # 最终 fallback：原文整体作为 answer
    return ParsedResponse(think="", answer=raw)
```

`ParsedResponse` 保留（think 字段始终为空，保持接口兼容），`extract_json_object()`、`extract_python_code()`、`_extract_answer_from_json()` 保留不变。

- [ ] **Step 2: 验证简化后的 parser**

Run: `.venv/bin/python -c "
from benchmark.core.response_parser import parse_response

# JSON 格式
r = parse_response('{\"answer\": \"42\"}', 'reasoning')
assert r.answer == '42' and r.think == '', f'Failed: {r}'
print('JSON reasoning: OK')

r = parse_response('{\"code\": \"def hello(): pass\"}', 'backend-dev')
assert r.answer == 'def hello(): pass' and r.think == ''
print('JSON backend-dev: OK')

# python code block fallback
r = parse_response('Here is code:\n\`\`\`python\ndef foo():\n    pass\n\`\`\`', 'backend-dev')
assert 'def foo' in r.answer
print('Code block fallback: OK')

# 纯文本 fallback
r = parse_response('42', 'reasoning')
assert r.answer == '42'
print('Text fallback: OK')

print('All tests passed!')
"`

Expected: 4 行 OK + `All tests passed!`

- [ ] **Step 3: Commit**

```bash
git add benchmark/core/response_parser.py
git commit -m "refactor(parser): simplify by removing strip_think_tags, adapter handles separation"
```

---

### Task 8: cli.py — 使用 adapter 返回的 reasoning_content

**Files:**
- Modify: `benchmark/cli.py`

- [ ] **Step 1: 修改 _evaluate_task() 的 think/answer 获取逻辑**

将 `_evaluate_task()` 中第 127-128 行的解析逻辑：

```python
        # 解析模型输出：分离 think 和 answer
        parsed = parse_response(model_output, task.dimension)
```

改为：

```python
        # 推理内容直接从 API 层获取（adapter 已分离）
        think_content = gen_response.reasoning_content

        # 从 content 中解析最终答案
        parsed = parse_response(model_output, task.dimension)
```

同时将 `EvalResult` 构造中的 `model_think=parsed.think` 改为 `model_think=think_content`：

```python
        result = EvalResult(
            result_id=result_id,
            run_id=run_id,
            task_id=task.task_id,
            task_content=task.prompt,
            model_output=model_output,
            model_think=think_content,
            model_answer=parsed.answer,
            ...
        )
```

- [ ] **Step 2: 修改 _evaluate_task() 中 save_metrics 调用，传递 reasoning 相关字段**

将第 165-174 行的 `ApiCallMetrics` 构造改为：

```python
        db.save_metrics(
            ApiCallMetrics(
                result_id=result_id,
                prompt_tokens=gen_response.prompt_tokens,
                completion_tokens=gen_response.completion_tokens,
                reasoning_tokens=gen_response.reasoning_tokens,
                reasoning_content=gen_response.reasoning_content,
                duration=gen_response.duration,
                tokens_per_second=tps,
                ttft_content=gen_response.ttft_content,
                created_at=datetime.now(),
            )
        )
```

- [ ] **Step 3: 修改 console.print 输出增加 TTFT-C**

将第 179-185 行的状态输出改为：

```python
        console.print(
            f"  [{task_idx + 1}/{total}] {task.task_id} | "
            f"Score: {score_result.score:.0f} | {status_icon} | "
            f"Time: {execution_time:.1f}s | "
            f"TTFT-R: {gen_response.ttft:.2f}s | "
            f"TTFT-C: {gen_response.ttft_content:.2f}s | "
            f"Speed: {tps:.1f} tok/s"
        )
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/cli.py
git commit -m "feat(cli): use adapter-level reasoning_content, add TTFT-R/C output"
```

---

### Task 9: app.py — 从 metrics 表读取 reasoning_content

**Files:**
- Modify: `benchmark/visualization/app.py`

- [ ] **Step 1: 修改 get_results_df 查询增加 ttft_content 和 reasoning_tokens**

在 `get_results_df()` 的 SELECT 中，在 `m.tokens_per_second` 后增加：

```sql
        m.tokens_per_second,
        m.ttft_content,
        m.reasoning_tokens,
        m.prompt_tokens,
        m.completion_tokens,
```

- [ ] **Step 2: 修改 detail 区域，优先展示 metrics 中的 reasoning_content**

在 app.py 第 184-193 行的 "Thinking" 折叠区，将数据源从 `detail.get("model_think")` 改为优先从 metrics 读取：

```python
                # 展示思考过程（折叠），优先从 metrics 读取 API 原生 reasoning_content
                think_content = detail.get("model_think", "") or ""
                # 从 metrics 获取更完整的 reasoning_content
                if metrics_row and metrics_row["reasoning_content"]:
                    think_content = metrics_row["reasoning_content"]
                if think_content:
                    with st.expander("Thinking", expanded=False):
                        st.text_area(
                            "Thinking Process",
                            value=think_content,
                            height=200,
                            disabled=True,
                            label_visibility="collapsed",
                        )
```

- [ ] **Step 3: 在指标区域增加 TTFT-C 和 reasoning_tokens 展示**

在 metrics 展示区域（第 166-173 行），增加：

```python
                if metrics_row:
                    st.metric(
                        "Token Speed",
                        f"{metrics_row['tokens_per_second']:.1f} tok/s",
                    )
                    st.metric(
                        "Tokens",
                        f"{metrics_row['prompt_tokens']} in / {metrics_row['completion_tokens']} out",
                    )
                    if metrics_row.get("reasoning_tokens", 0) > 0:
                        st.metric(
                            "Reasoning Tokens",
                            f"{metrics_row['reasoning_tokens']}",
                        )
                    if metrics_row.get("ttft_content", 0) > 0:
                        st.metric(
                            "TTFT-C",
                            f"{metrics_row['ttft_content']:.2f}s",
                        )
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/visualization/app.py
git commit -m "feat(ui): display reasoning_content from metrics, add TTFT-C and reasoning tokens"
```

---

### Task 10: 端到端验证

**Files:** 无代码改动

- [ ] **Step 1: 删除旧 DB，确保从干净状态开始**

```bash
rm -f benchmark/data/results.db
```

- [ ] **Step 2: 运行 reasoning 维度评测（2 样本）**

```bash
.venv/bin/python -m benchmark evaluate --model kimi/kimi-for-coding --dimension reasoning --samples 2
```

验证：
- 控制台输出包含 TTFT-R 和 TTFT-C 两行指标
- 评测完成无报错

- [ ] **Step 3: 运行 backend-dev 维度评测（2 样本）**

```bash
.venv/bin/python -m benchmark evaluate --model kimi/kimi-for-coding --dimension backend-dev --samples 2
```

验证：同上

- [ ] **Step 4: 检查 DB 数据**

```bash
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('benchmark/data/results.db')

# 检查 model_think 是否有内容
rows = conn.execute('SELECT result_id, model_think FROM eval_results WHERE model_think != \"\"').fetchall()
print(f'results with think: {len(rows)}')

# 检查 reasoning_content 是否有内容
rows = conn.execute('SELECT result_id, reasoning_content FROM api_call_metrics WHERE reasoning_content != \"\"').fetchall()
print(f'metrics with reasoning: {len(rows)}')

# 检查 TTFT-C 是否非零
rows = conn.execute('SELECT result_id, ttft_content FROM api_call_metrics WHERE ttft_content > 0').fetchall()
print(f'metrics with TTFT-C: {len(rows)}')

conn.close()
"
```

Expected: 三行都有非零计数

- [ ] **Step 5: 启动前端验证**

```bash
.venv/bin/streamlit run benchmark/visualization/app.py
```

验证：
- 表格行点击可切换详情
- Thinking 折叠区有内容（从 metrics.reasoning_content 读取）
- 指标区域显示 Token Speed、Tokens、可能显示 Reasoning Tokens 和 TTFT-C
