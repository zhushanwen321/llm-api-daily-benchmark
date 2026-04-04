# Token 生成速度追踪 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 记录每次 API 调用的 token 用量和生成速度，持久化到独立数据库表，并在 UI 中展示。

**Architecture:** 在现有 `eval_results` 的基础上新增 `api_call_metrics` 表，通过 `result_id` 一对一关联。LLM Adapter 返回结构化响应携带 token 用量，CLI 层负责计算速度并写入数据库，UI 层从新表读取并展示。

**Tech Stack:** Python 3.13, Pydantic, SQLite, Streamlit, requests

---

### Task 1: 新增 GenerateResponse 和 ApiCallMetrics 数据模型

**Files:**
- Modify: `benchmark/models/schemas.py` (全文 59 行)
- Create: `tests/__init__.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: 在 schemas.py 顶部添加 GenerateResponse 模型**

在 `benchmark/models/schemas.py` 的 `ScoreResult` 类之后（第 29 行之后），添加：

```python
class GenerateResponse(BaseModel):
    """LLM API 调用响应，包含文本和 token 用量。"""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

在文件末尾（`EvalResult` 类之后）添加：

```python
class ApiCallMetrics(BaseModel):
    """单次 API 调用的 token 指标。"""

    result_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration: float = 0.0
    tokens_per_second: float = 0.0
    created_at: datetime
```

- [ ] **Step 2: 写测试验证模型创建**

创建 `tests/__init__.py`（空文件）和 `tests/test_schemas.py`：

```python
from datetime import datetime

from benchmark.models.schemas import ApiCallMetrics, GenerateResponse


def test_generate_response_defaults():
    resp = GenerateResponse(content="hello")
    assert resp.content == "hello"
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0


def test_generate_response_with_tokens():
    resp = GenerateResponse(content="hello", prompt_tokens=10, completion_tokens=5)
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5


def test_api_call_metrics():
    now = datetime.now()
    metrics = ApiCallMetrics(
        result_id="abc123",
        prompt_tokens=100,
        completion_tokens=50,
        duration=2.5,
        tokens_per_second=20.0,
        created_at=now,
    )
    assert metrics.result_id == "abc123"
    assert metrics.tokens_per_second == 20.0
```

- [ ] **Step 3: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_schemas.py -v`
Expected: 3 tests PASS

- [ ] **Step 4: 提交**

```bash
git add benchmark/models/schemas.py tests/__init__.py tests/test_schemas.py
git commit -m "feat(schemas): 新增 GenerateResponse 和 ApiCallMetrics 模型"
```

---

### Task 2: 修改 LLMEvalAdapter.generate() 返回 GenerateResponse

**Files:**
- Modify: `benchmark/core/llm_adapter.py` (全文 134 行)
- Create: `tests/test_llm_adapter.py`

- [ ] **Step 1: 修改 generate() 方法签名和返回值**

在 `benchmark/core/llm_adapter.py` 顶部（第 9 行 `import requests` 之后）添加 import：

```python
from benchmark.models.schemas import GenerateResponse
```

将 `generate()` 方法的返回类型从 `str` 改为 `GenerateResponse`。修改第 62-116 行的 `generate()` 方法，将原来的：

```python
    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
```

改为：

```python
    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> GenerateResponse:
```

将第 115-116 行：

```python
                data = resp.json()
                return data["choices"][0]["message"]["content"]
```

改为：

```python
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return GenerateResponse(
                    content=content,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
```

同时更新第 131-133 行的异常消息，将 `'model'` 位置不变。

- [ ] **Step 2: 写测试验证 generate 返回结构**

创建 `tests/test_llm_adapter.py`：

```python
from unittest.mock import MagicMock, patch

from benchmark.core.llm_adapter import LLMEvalAdapter


def _mock_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """构造模拟的 API 响应。"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return mock_resp


@patch("benchmark.core.llm_adapter.get_model_config")
@patch("benchmark.core.llm_adapter.requests.post")
def test_generate_returns_generate_response(mock_post, mock_config):
    mock_config.return_value = {
        "api_key": "test-key",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
    }
    mock_post.return_value = _mock_response("hello world", 20, 10)

    adapter = LLMEvalAdapter()
    result = adapter.generate("test prompt", "test/model")

    assert result.content == "hello world"
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 10


@patch("benchmark.core.llm_adapter.get_model_config")
@patch("benchmark.core.llm_adapter.requests.post")
def test_generate_handles_missing_usage(mock_post, mock_config):
    mock_config.return_value = {
        "api_key": "test-key",
        "api_base": "https://api.test.com/v1",
        "provider": "test",
        "max_tokens": 4096,
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "no usage"}}],
    }
    mock_post.return_value = mock_resp

    adapter = LLMEvalAdapter()
    result = adapter.generate("test prompt", "test/model")

    assert result.content == "no usage"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
```

- [ ] **Step 3: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_llm_adapter.py -v`
Expected: 2 tests PASS

- [ ] **Step 4: 提交**

```bash
git add benchmark/core/llm_adapter.py tests/test_llm_adapter.py
git commit -m "feat(llm_adapter): generate() 返回 GenerateResponse 含 token 用量"
```

---

### Task 3: 数据库层新增 api_call_metrics 表和操作方法

**Files:**
- Modify: `benchmark/models/database.py` (全文 195 行)
- Create: `tests/test_database.py`

- [ ] **Step 1: 在 _init_db 方法中添加建表语句**

在 `benchmark/models/database.py` 的 `_init_db()` 方法中，第 92 行 `conn.commit()` 之前，添加建表语句：

```python
        cursor.execute("""
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
        """)
```

- [ ] **Step 2: 添加 save_metrics 方法**

在 `database.py` 顶部 import 中添加（第 15 行已有 schemas import）：

```python
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun
```

在 `save_result()` 方法之后（约第 149 行之后）添加：

```python
    def save_metrics(self, metrics: ApiCallMetrics) -> str:
        """保存 API 调用的 token 指标。"""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO api_call_metrics
               (result_id, prompt_tokens, completion_tokens,
                duration, tokens_per_second, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                metrics.result_id,
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.duration,
                metrics.tokens_per_second,
                metrics.created_at.isoformat(),
            ),
        )
        conn.commit()
        return metrics.result_id
```

- [ ] **Step 3: 写测试验证数据库操作**

创建 `tests/test_database.py`：

```python
import tempfile
from datetime import datetime
from pathlib import Path

from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics


def _test_db() -> Database:
    """创建临时数据库。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = Database(db_path=Path(tmp.name))
    return db


def test_save_and_query_metrics():
    db = _test_db()
    now = datetime.now()
    metrics = ApiCallMetrics(
        result_id="r001",
        prompt_tokens=100,
        completion_tokens=50,
        duration=2.0,
        tokens_per_second=25.0,
        created_at=now,
    )
    db.save_metrics(metrics)

    conn = db._get_conn()
    row = conn.execute(
        "SELECT * FROM api_call_metrics WHERE result_id = ?", ("r001",)
    ).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM api_call_metrics WHERE result_id = ?", ("r001",)).description]
    result = dict(zip(cols, row))

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["duration"] == 2.0
    assert result["tokens_per_second"] == 25.0
    db.close()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_database.py -v`
Expected: 1 test PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/models/database.py tests/test_database.py
git commit -m "feat(database): 新增 api_call_metrics 表和 save_metrics 方法"
```

---

### Task 4: CLI 层集成 token 速度记录

**Files:**
- Modify: `benchmark/cli.py` (全文 205 行)

- [ ] **Step 1: 修改 cli.py 的 import 和 evaluate 逻辑**

在 `benchmark/cli.py` 第 21 行的 import 中，添加 `ApiCallMetrics`：

```python
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun
```

将第 106-123 行的评测循环核心逻辑：

```python
                start_time = datetime.now()
                model_output = llm.generate(task.prompt, model)
                execution_time = (datetime.now() - start_time).total_seconds()

                score_result = scorer.score(model_output, task.expected_output, task)

                result = EvalResult(
                    result_id=str(uuid.uuid4())[:12],
                    run_id=run_id,
                    task_id=task.task_id,
                    task_content=task.prompt,
                    model_output=model_output,
                    functional_score=score_result.score,
                    final_score=score_result.score,
                    passed=score_result.passed,
                    details=score_result.details,
                    execution_time=execution_time,
                    created_at=datetime.now(),
                )
                db.save_result(result)
```

改为：

```python
                start_time = datetime.now()
                gen_response = llm.generate(task.prompt, model)
                execution_time = (datetime.now() - start_time).total_seconds()
                model_output = gen_response.content

                score_result = scorer.score(model_output, task.expected_output, task)

                result_id = str(uuid.uuid4())[:12]
                result = EvalResult(
                    result_id=result_id,
                    run_id=run_id,
                    task_id=task.task_id,
                    task_content=task.prompt,
                    model_output=model_output,
                    functional_score=score_result.score,
                    final_score=score_result.score,
                    passed=score_result.passed,
                    details=score_result.details,
                    execution_time=execution_time,
                    created_at=datetime.now(),
                )
                db.save_result(result)

                # 计算 token 速度并记录
                tps = (
                    gen_response.completion_tokens / execution_time
                    if execution_time > 0
                    else 0.0
                )
                db.save_metrics(
                    ApiCallMetrics(
                        result_id=result_id,
                        prompt_tokens=gen_response.prompt_tokens,
                        completion_tokens=gen_response.completion_tokens,
                        duration=execution_time,
                        tokens_per_second=tps,
                        created_at=datetime.now(),
                    )
                )
```

同时更新 CLI 输出，将第 136-137 行：

```python
                console.print(
                    f"  [{i}/{len(tasks)}] {task.task_id} | "
                    f"Score: {score_result.score:.0f} | {status_icon} | "
                    f"Time: {execution_time:.1f}s"
                )
```

改为：

```python
                console.print(
                    f"  [{i}/{len(tasks)}] {task.task_id} | "
                    f"Score: {score_result.score:.0f} | {status_icon} | "
                    f"Time: {execution_time:.1f}s | "
                    f"Speed: {tps:.1f} tok/s"
                )
```

- [ ] **Step 2: 运行全部测试确认无破坏**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 3: 提交**

```bash
git add benchmark/cli.py
git commit -m "feat(cli): 集成 token 速度记录，CLI 输出增加速度信息"
```

---

### Task 5: UI 层展示 token 速度

**Files:**
- Modify: `benchmark/visualization/app.py` (全文 168 行)

- [ ] **Step 1: 修改 get_results_df 函数，关联查询 api_call_metrics**

将 `app.py` 中的 `get_results_df()` 函数（第 32-60 行）的 SQL 查询从：

```python
    query = """
        SELECT
            r.result_id,
            e.model,
            e.dimension,
            r.task_id,
            r.final_score,
            r.passed,
            r.execution_time,
            r.created_at
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        WHERE 1=1
    """
```

改为：

```python
    query = """
        SELECT
            r.result_id,
            e.model,
            e.dimension,
            r.task_id,
            r.final_score,
            r.passed,
            r.execution_time,
            m.tokens_per_second,
            m.prompt_tokens,
            m.completion_tokens,
            r.created_at
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        LEFT JOIN api_call_metrics m ON r.result_id = m.result_id
        WHERE 1=1
    """
```

- [ ] **Step 2: 修改表格展示列**

将 `main()` 函数中第 108-121 行的表格展示代码：

```python
    display_df = df.copy()
    display_df["passed"] = display_df["passed"].map(lambda x: "Yes" if x else "No")
    display_df["execution_time"] = (
        display_df["execution_time"].round(2).astype(str) + "s"
    )
    display_df.columns = [
        "ID",
        "Model",
        "Dimension",
        "Task",
        "Score",
        "Passed",
        "Time",
        "Date",
    ]
```

改为：

```python
    display_df = df.copy()
    display_df["passed"] = display_df["passed"].map(lambda x: "Yes" if x else "No")
    display_df["execution_time"] = (
        display_df["execution_time"].round(2).astype(str) + "s"
    )
    display_df["tokens_per_second"] = display_df["tokens_per_second"].apply(
        lambda x: f"{x:.1f} tok/s" if pd.notna(x) else "-"
    )
    display_df = display_df.drop(columns=["prompt_tokens", "completion_tokens"])
    display_df.columns = [
        "ID",
        "Model",
        "Dimension",
        "Task",
        "Score",
        "Passed",
        "Time",
        "Token Speed",
        "Date",
    ]
```

- [ ] **Step 3: 修改详情页，展示 token 指标**

在 `main()` 函数的详情页部分（约第 131-136 行），将：

```python
            with col1:
                st.metric("Score", f"{detail['final_score']:.1f}")
                st.metric("Passed", "Yes" if detail["passed"] else "No")
                st.metric("Execution Time", f"{detail['execution_time']:.2f}s")
```

改为：

```python
            with col1:
                st.metric("Score", f"{detail['final_score']:.1f}")
                st.metric("Passed", "Yes" if detail["passed"] else "No")
                st.metric("Execution Time", f"{detail['execution_time']:.2f}s")

                # 查询 token 指标
                metrics_row = conn.execute(
                    "SELECT * FROM api_call_metrics WHERE result_id = ?",
                    (selected_result,),
                ).fetchone()
                if metrics_row:
                    cols = [d[0] for d in conn.execute(
                        "SELECT * FROM api_call_metrics WHERE result_id = ?",
                        (selected_result,),
                    ).description]
                    metrics = dict(zip(cols, metrics_row))
                    st.metric(
                        "Token Speed",
                        f"{metrics['tokens_per_second']:.1f} tok/s",
                    )
                    st.metric(
                        "Tokens",
                        f"{metrics['prompt_tokens']} in / {metrics['completion_tokens']} out",
                    )
```

- [ ] **Step 4: 提交**

```bash
git add benchmark/visualization/app.py
git commit -m "feat(ui): 结果表格和详情页展示 token 速度指标"
```

---

### Task 6: 验证整体流程

- [ ] **Step 1: 运行全部测试**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 2: 验证 CLI 帮助正常**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m benchmark evaluate --help`
Expected: 正常显示帮助信息，无 import 错误

- [ ] **Step 3: 提交最终状态（如有未提交的修改）**

```bash
git status
# 确认干净
```
