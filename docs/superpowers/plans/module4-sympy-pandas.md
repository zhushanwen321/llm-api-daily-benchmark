# 模块 4+5：移除 sympy + pandas（Task 9-10）

### Task 9: 移除 sympy 依赖（math_scorer.py）

**Files:**
- Modify: `benchmark/scorers/math_scorer.py`

---

- [ ] **Step 1: 删除 `_try_sympy_match` 函数**

删除 `math_scorer.py` 中的 `_try_sympy_match` 函数（第 103-115 行）。

- [ ] **Step 2: 删除 `score()` 中的 sympy 调用**

删除 `score()` 方法中第 149-156 行的 sympy 分支：

```python
# 删除以下代码块
if _try_sympy_match(predicted, expected):
    return ScoreResult(
        score=100.0, passed=True,
        details={"predicted": predicted, "expected": expected, "method": "sympy"},
        reasoning=f"Correct (symbolic): {predicted} == {expected}",
    )
```

- [ ] **Step 3: 更新文档字符串**

模块 docstring 从 `"支持数值比较和 sympy 符号比较"` 改为 `"支持数值比较"`。

类 docstring 从 `"支持三种匹配模式"` 改为 `"支持两种匹配模式"`，删除第 3 条。

- [ ] **Step 4: 运行测试确认无回归**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_math_scorer.py -v`
Expected: 全部 8 个测试通过（无测试依赖 sympy）

- [ ] **Step 5: 提交**

```
git add benchmark/scorers/math_scorer.py
git commit -m "refactor(math_scorer): 移除 sympy fallback，仅保留字符串+数值匹配"
```

---

### Task 10: 移除 pandas 依赖（app.py）

**Files:**
- Modify: `benchmark/visualization/app.py`

---

- [ ] **Step 1: 移除 pandas import**

删除 `import pandas as pd`。

- [ ] **Step 2: 重写 `get_results_df` → `get_results`**

将函数签名从 `-> pd.DataFrame` 改为 `-> list[dict]`，函数体改为：

```python
def get_results(
    conn: sqlite3.Connection, model: str | None, dimension: str | None
) -> list[dict]:
    """查询结果并返回 list[dict]。"""
    query = """
        SELECT
            r.result_id, e.model, e.dimension, r.task_id,
            r.final_score, r.passed, r.execution_time,
            m.tokens_per_second, m.ttft_content, m.reasoning_tokens,
            m.prompt_tokens, m.completion_tokens, r.created_at
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        LEFT JOIN api_call_metrics m ON r.result_id = m.result_id
        WHERE 1=1
    """
    params: list[str] = []
    if model and model != "All":
        query += " AND e.model = ?"
        params.append(model)
    if dimension and dimension != "All":
        query += " AND e.dimension = ?"
        params.append(dimension)
    query += " ORDER BY r.created_at DESC"
    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]
```

- [ ] **Step 3: 更新 main() 中所有下游消费**

**3a.** `df = get_results_df(...)` → `results = get_results(...)`

**3b.** `if df.empty:` → `if not results:`

**3c.** `scores = df["final_score"].tolist()` → `scores = [row["final_score"] for row in results]`

**3d.** 结果表格展示部分，替换 DataFrame 操作为 list[dict] 构建：

```python
_DROP_COLUMNS = {"prompt_tokens", "completion_tokens", "ttft_content", "reasoning_tokens"}
_COLUMN_RENAME = {
    "result_id": "ID", "model": "Model", "dimension": "Dimension",
    "task_id": "Task", "final_score": "Score", "passed": "Passed",
    "execution_time": "Time", "tokens_per_second": "Token Speed",
    "created_at": "Date",
}
display_data = []
for row in results:
    display_row = {}
    for key, value in row.items():
        if key in _DROP_COLUMNS:
            continue
        new_key = _COLUMN_RENAME.get(key, key)
        if key == "passed":
            display_row[new_key] = "Yes" if value else "No"
        elif key == "execution_time":
            display_row[new_key] = f"{value:.2f}s" if value is not None else "-"
        elif key == "tokens_per_second":
            display_row[new_key] = f"{value:.1f} tok/s" if value is not None else "-"
        else:
            display_row[new_key] = value
    display_data.append(display_row)

st.dataframe(display_data, use_container_width=True)
```

关键替换：`pd.notna(x)` → `x is not None`

**3e.** `df["result_id"].tolist()` → `[row["result_id"] for row in results]`（出现在 tab3 中）

- [ ] **Step 4: 搜索确认无遗漏**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && grep -n "pd\.\|DataFrame\|df\[\|df\." benchmark/visualization/app.py`
Expected: 无输出

- [ ] **Step 5: 运行测试**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/ -v -k "app"`
Expected: 相关测试通过

- [ ] **Step 6: 提交**

```
git add benchmark/visualization/app.py
git commit -m "refactor(app): 移除 pandas，改用 sqlite3 原生查询 + list[dict]"
```
