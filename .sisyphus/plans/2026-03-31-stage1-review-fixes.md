# Stage 1 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Stage 1 代码审查发现的 2 个 Critical + 6 个 Important 问题。

**Architecture:** 最小化改动，每个修复独立成 task。不引入新依赖，不改变公共接口。Database 连接管理改为上下文管理器模式。

**Tech Stack:** Python 3.11+, sqlite3, pydantic, click

---

## File Structure

```
修改文件:
├── benchmark/cli.py                      # C1, C2, I2, I3 — 4 个修复
├── benchmark/models/database.py          # I1 — 连接管理
├── benchmark/scorers/exact_match_scorer.py # I4 — 正则修复
├── benchmark/core/llm_adapter.py         # I5 — 配置缓存
```

不需要创建新文件，所有修复都是对现有文件的改动。

---

## TODOs

- [ ] 1. Fix C1: `System.Exit` -> `SystemExit` (cli.py)

  **What to do**:
  - 修复 `benchmark/cli.py` 第 52 行和第 63 行的 `raise System.Exit(1)` 为 `raise SystemExit(1)`

  **Must NOT do**:
  - 不要改变其他逻辑

  **Step-by-step instructions**:

  - [ ] **Step 1: 修改 cli.py 第 52 行**

  将：
  ```python
      raise System.Exit(1)
  ```
  改为：
  ```python
      raise SystemExit(1)
  ```

  - [ ] **Step 2: 修改 cli.py 第 63 行**

  将：
  ```python
      raise System.Exit(1)
  ```
  改为：
  ```python
      raise SystemExit(1)
  ```

  - [ ] **Step 3: 验证**

  ```bash
  python -c "import ast; ast.parse(open('benchmark/cli.py').read()); print('Syntax OK')"
  ```

  Expected: `Syntax OK`

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/cli.py
  git commit -m "fix(benchmark): correct System.Exit to SystemExit in CLI"
  ```

---

- [ ] 2. Fix C2: `dataset` 字段赋值错误 (cli.py)

  **What to do**:
  - 修复 `benchmark/cli.py` 第 75 行 `dataset=adapter.get_dimension()` 语义错误
  - `dataset` 字段应存储数据集名称（如 `"gsm8k"` / `"bigcodebench"`），而非维度名称（如 `"reasoning"` / `"backend-dev"`）
  - 适配器本身在 `load()` 方法中已经为每个 `TaskDefinition` 设置了正确的 `dataset` 字段（GSM8K 设置 `"gsm8k"`，BigCodeBench 设置 `"bigcodebench"`）
  - 需要从已加载的 tasks 中提取数据集名称；如果没有 tasks，回退到从 DIMENSION_REGISTRY 映射中获取

  **Must NOT do**:
  - 不要修改适配器类
  - 不要修改 schemas

  **Step-by-step instructions**:

  - [ ] **Step 1: 添加 DATASET_REGISTRY 映射**

  在 `benchmark/cli.py` 中 `DIMENSION_REGISTRY` 定义之后，添加数据集名称映射：

  ```python
  DATASET_REGISTRY: dict[str, str] = {
      "reasoning": "gsm8k",
      "backend-dev": "bigcodebench",
  }
  ```

  - [ ] **Step 2: 修改 EvalRun 的 dataset 赋值**

  将 cli.py 中创建 EvalRun 的代码（约第 71-78 行）：
  ```python
  run = EvalRun(
      run_id=run_id,
      model=model,
      dimension=dimension,
      dataset=adapter.get_dimension(),
      started_at=datetime.now(),
      status="running",
  )
  ```
  改为：
  ```python
  run = EvalRun(
      run_id=run_id,
      model=model,
      dimension=dimension,
      dataset=DATASET_REGISTRY[dimension],
      started_at=datetime.now(),
      status="running",
  )
  ```

  - [ ] **Step 3: 验证**

  ```bash
  python -c "
  from benchmark.cli import DATASET_REGISTRY
  assert DATASET_REGISTRY['reasoning'] == 'gsm8k'
  assert DATASET_REGISTRY['backend-dev'] == 'bigcodebench'
  print('DATASET_REGISTRY OK')
  "
  ```

  Expected: `DATASET_REGISTRY OK`

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/cli.py
  git commit -m "fix(benchmark): use dataset name instead of dimension name in EvalRun"
  ```

---

- [ ] 3. Fix I1: Database 连接管理优化 (database.py)

  **What to do**:
  - 将 `Database` 类改为上下文管理器模式，`__init__` 时建立连接，`close()` / `__exit__` 时关闭
  - 内部方法复用同一个连接，避免频繁开关
  - 保持公共接口不变（`create_run`, `finish_run`, `save_result`, `get_results`, `get_result_detail`）

  **Must NOT do**:
  - 不要引入连接池
  - 不要修改方法签名

  **Step-by-step instructions**:

  - [ ] **Step 1: 重写 database.py**

  完整替换 `benchmark/models/database.py`：

  ```python
  """SQLite 数据库操作。

  负责评测结果的持久化存储。使用 sqlite3 标准库，不依赖 ORM。
  支持上下文管理器，在 __init__ 时建立连接，close() 时关闭。
  """

  from __future__ import annotations

  import json
  import sqlite3
  from datetime import datetime
  from pathlib import Path
  from typing import Optional

  from benchmark.models.schemas import EvalResult, EvalRun


  class Database:
      """SQLite 数据库操作类。

      支持上下文管理器：
          with Database() as db:
              db.create_run(run)
              db.save_result(result)
      也可以直接使用：
          db = Database()
          db.create_run(run)
          db.close()
      """

      def __init__(self, db_path: str | Path = "benchmark/data/results.db") -> None:
          self.db_path = Path(db_path)
          self.db_path.parent.mkdir(parents=True, exist_ok=True)
          self._conn: sqlite3.Connection | None = None
          self._init_db()

      def _get_conn(self) -> sqlite3.Connection:
          """获取连接。单次 Database 实例生命周期内复用同一连接."""
          if self._conn is None:
              self._conn = sqlite3.connect(str(self.db_path))
          return self._conn

      def close(self) -> None:
          """关闭数据库连接."""
          if self._conn is not None:
              self._conn.close()
              self._conn = None

      def __enter__(self) -> Database:
          return self

      def __exit__(self, *args: object) -> None:
          self.close()

      def _init_db(self) -> None:
          """初始化数据库表（如不存在则创建）。"""
          conn = self._get_conn()
          cursor = conn.cursor()

          cursor.execute("""
              CREATE TABLE IF NOT EXISTS eval_runs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT UNIQUE NOT NULL,
                  model TEXT NOT NULL,
                  dimension TEXT NOT NULL,
                  dataset TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  status TEXT NOT NULL DEFAULT 'running',
                  config_snapshot TEXT
              )
          """)

          cursor.execute("""
              CREATE TABLE IF NOT EXISTS eval_results (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  result_id TEXT UNIQUE NOT NULL,
                  run_id TEXT NOT NULL,
                  task_id TEXT NOT NULL,
                  task_content TEXT,
                  model_output TEXT,
                  functional_score REAL NOT NULL DEFAULT 0,
                  quality_score REAL NOT NULL DEFAULT 0,
                  final_score REAL NOT NULL DEFAULT 0,
                  passed INTEGER NOT NULL DEFAULT 0,
                  details TEXT,
                  execution_time REAL NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
              )
          """)

          conn.commit()

      def create_run(self, run: EvalRun) -> str:
          """创建评测运行记录。"""
          conn = self._get_conn()
          conn.execute(
              """INSERT INTO eval_runs
                 (run_id, model, dimension, dataset, started_at, status, config_snapshot)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (
                  run.run_id,
                  run.model,
                  run.dimension,
                  run.dataset,
                  run.started_at.isoformat(),
                  run.status,
                  getattr(run, "config_snapshot", None),
              ),
          )
          conn.commit()
          return run.run_id

      def finish_run(self, run_id: str, status: str = "completed") -> None:
          """标记运行记录为已完成。"""
          conn = self._get_conn()
          conn.execute(
              "UPDATE eval_runs SET finished_at = ?, status = ? WHERE run_id = ?",
              (datetime.now().isoformat(), status, run_id),
          )
          conn.commit()

      def save_result(self, result: EvalResult) -> str:
          """保存单题评测结果。"""
          conn = self._get_conn()
          conn.execute(
              """INSERT INTO eval_results
                 (result_id, run_id, task_id, task_content, model_output,
                  functional_score, quality_score, final_score, passed,
                  details, execution_time, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (
                  result.result_id,
                  result.run_id,
                  result.task_id,
                  result.task_content,
                  result.model_output,
                  result.functional_score,
                  result.quality_score,
                  result.final_score,
                  int(result.passed),
                  json.dumps(result.details, ensure_ascii=False),
                  result.execution_time,
                  result.created_at.isoformat(),
              ),
          )
          conn.commit()
          return result.result_id

      def get_results(
          self,
          model: Optional[str] = None,
          dimension: Optional[str] = None,
      ) -> list[dict]:
          """查询评测结果。"""
          conn = self._get_conn()
          query = """
              SELECT r.result_id, e.model, e.dimension,
                     r.task_id, r.final_score, r.passed,
                     r.execution_time, r.created_at
              FROM eval_results r
              JOIN eval_runs e ON r.run_id = e.run_id
              WHERE 1=1
          """
          params: list = []
          if model:
              query += " AND e.model = ?"
              params.append(model)
          if dimension:
              query += " AND e.dimension = ?"
              params.append(dimension)
          query += " ORDER BY r.created_at DESC"

          cursor = conn.execute(query, params)
          columns = [desc[0] for desc in cursor.description]
          rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
          return rows

      def get_result_detail(self, result_id: str) -> Optional[dict]:
          """获取单条结果的完整详情。"""
          conn = self._get_conn()
          cursor = conn.execute(
              """SELECT r.*, e.model, e.dimension
                 FROM eval_results r
                 JOIN eval_runs e ON r.run_id = e.run_id
                 WHERE r.result_id = ?""",
              (result_id,),
          )
          columns = [desc[0] for desc in cursor.description]
          row = cursor.fetchone()
          if row is None:
              return None
          return dict(zip(columns, row))
  ```

  - [ ] **Step 2: 修改 cli.py 使用上下文管理器**

  将 `benchmark/cli.py` 的 `evaluate` 函数中 db 使用改为上下文管理器：

  将：
  ```python
      db = Database()
  ```
  改为：
  ```python
      db = Database()
      try:
  ```
  并且在 evaluate 函数结尾（`db.finish_run` 和 `console.print` 之后）添加 `finally`：

  ```python
      finally:
          db.close()
  ```

  注意：由于后续 Task 5（I3）会重构整个 evaluate 函数的异常处理，这里可以只做简单的 try/finally 包裹。完整改法在 Task 5 中。

  - [ ] **Step 3: 验证 CRUD 操作**

  ```bash
  python -c "
  from benchmark.models.database import Database
  from benchmark.models.schemas import EvalRun, EvalResult
  from datetime import datetime
  import os

  with Database('benchmark/data/test_fix.db') as db:
      db.create_run(EvalRun(run_id='fix-1', model='test', dimension='reasoning', dataset='gsm8k', started_at=datetime.now(), status='running'))
      db.save_result(EvalResult(result_id='fr-1', run_id='fix-1', task_id='t1', task_content='test', model_output='42', functional_score=100, final_score=100, passed=True, execution_time=0.5, created_at=datetime.now()))
      db.finish_run('fix-1')
      rows = db.get_results()
      assert len(rows) == 1
      assert rows[0]['model'] == 'test'
      print('Database context manager OK')

  os.remove('benchmark/data/test_fix.db')
  "
  ```

  Expected: `Database context manager OK`

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/models/database.py benchmark/cli.py
  git commit -m "refactor(benchmark): Database connection reuse with context manager support"
  ```

---

- [ ] 4. Fix I2: UUID 截断过短 (cli.py)

  **What to do**:
  - 将 UUID 截断从 `[:8]`（32 bit）改为 `[:12]`（48 bit，约 281 万亿种可能）
  - 12 个十六进制字符足够避免碰撞，同时保持 ID 可读

  **Step-by-step instructions**:

  - [ ] **Step 1: 修改 run_id 截断**

  将 `benchmark/cli.py` 第 70 行：
  ```python
  run_id = str(uuid.uuid4())[:8]
  ```
  改为：
  ```python
  run_id = str(uuid.uuid4())[:12]
  ```

  - [ ] **Step 2: 修改 result_id 截断**

  将 `benchmark/cli.py` 第 93 行：
  ```python
  result_id = str(uuid.uuid4())[:8],
  ```
  改为：
  ```python
  result_id = str(uuid.uuid4())[:12],
  ```

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/cli.py
  git commit -m "fix(benchmark): extend UUID truncation from 8 to 12 chars"
  ```

---

- [ ] 5. Fix I3: evaluate 命令异常处理 (cli.py)

  **What to do**:
  - 在 evaluate 函数的评测循环外包裹 try/except，异常时调用 `db.finish_run(run_id, "failed")`
  - 确保 LLM 调用失败或评分异常时，eval_runs 记录不会永远停留在 running 状态
  - 将 console.print 错误信息，然后 re-raise 让 click 处理退出码

  **Step-by-step instructions**:

  - [ ] **Step 1: 重构 evaluate 函数**

  将 `benchmark/cli.py` 中 evaluate 函数的主体（从 `run_id = ...` 到函数末尾）改为：

  ```python
      run_id = str(uuid.uuid4())[:12]
      run = EvalRun(
          run_id=run_id,
          model=model,
          dimension=dimension,
          dataset=DATASET_REGISTRY[dimension],
          started_at=datetime.now(),
          status="running",
      )
      db = Database()
      try:
          db.create_run(run)

          total_score = 0.0
          passed_count = 0
          with Progress() as progress:
              task_progress = progress.add_task("Evaluating", total=len(tasks))
              for i, task in enumerate(tasks, 1):
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

                  total_score += score_result.score
                  if score_result.passed:
                      passed_count += 1

                  status_icon = (
                      "[green]PASS[/green]" if score_result.passed else "[red]FAIL[/red]"
                  )
                  console.print(
                      f"  [{i}/{len(tasks)}] {task.task_id} | "
                      f"Score: {score_result.score:.0f} | {status_icon} | "
                      f"Time: {execution_time:.1f}s"
                  )
                  progress.advance(task_progress)

          db.finish_run(run_id, "completed")

          avg_score = total_score / len(tasks) if tasks else 0
          console.print(
              f"\n[bold]Evaluation complete:[/bold] run_id={run_id}\n"
              f"  Average Score: [bold]{avg_score:.1f}[/bold]\n"
              f"  Passed: {passed_count}/{len(tasks)}"
          )
      except Exception:
          console.print("[red]Evaluation failed![/red]")
          try:
              db.finish_run(run_id, "failed")
          except Exception:
              pass
          raise
      finally:
          db.close()
  ```

  - [ ] **Step 2: 验证语法**

  ```bash
  python -c "import ast; ast.parse(open('benchmark/cli.py').read()); print('Syntax OK')"
  ```

  Expected: `Syntax OK`

  - [ ] **Step 3: 验证 CLI help 仍正常**

  ```bash
  python -m benchmark evaluate --help
  ```

  Expected: 显示 --model, --dimension, --samples 参数帮助

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/cli.py
  git commit -m "fix(benchmark): mark eval_runs as failed on exception in evaluate command"
  ```

---

- [ ] 6. Fix I4: ExactMatchScorer 数值比较 (exact_match_scorer.py)

  **What to do**:
  - 修复正则表达式 `-?\d+\.?\d*` 会匹配单独 `-` 的问题，改为 `-?\d+(?:\.\d+)?`
  - 将字符串精确匹配改为数值比较：将 predicted 和 expected 都转为 float 后比较，避免 `"42"` vs `"42.00"` 误判
  - 比较时使用 `math.isclose` 容差比较（容忍浮点精度差异）

  **Step-by-step instructions**:

  - [ ] **Step 1: 重写 exact_match_scorer.py**

  完整替换 `benchmark/scorers/exact_match_scorer.py`：

  ```python
  """精确匹配评分器。从模型输出中提取数字，与期望答案数值比较."""

  from __future__ import annotations

  import math
  import re

  from benchmark.models.schemas import ScoreResult, TaskDefinition
  from benchmark.scorers.base import BaseScorer


  class ExactMatchScorer(BaseScorer):
      """精确匹配评分器，用于 reasoning 维度（GSM8K）。

      从模型输出中提取最后一个数字作为预测答案，
      与 expected_output 中的期望答案进行数值比较。
      使用 math.isclose 容忍浮点精度差异。
      匹配成功 score=100，失败 score=0。
      """

      # 匹配整数或小数，不匹配单独的减号
      _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

      def score(
          self,
          model_output: str,
          expected: str,
          task: TaskDefinition,  # noqa: ARG002 — 基类接口要求
      ) -> ScoreResult:
          numbers = self._NUMBER_RE.findall(model_output)
          if not numbers:
              return ScoreResult(
                  score=0,
                  passed=False,
                  details={
                      "error": "No number found in output",
                      "raw_output": model_output[:200],
                  },
                  reasoning="Model output contains no numeric answer",
              )

          predicted_str = numbers[-1]
          expected_str = expected.strip()

          # 先尝试字符串精确匹配（快速路径）
          if predicted_str == expected_str:
              return ScoreResult(
                  score=100.0,
                  passed=True,
                  details={"predicted": predicted_str, "expected": expected_str},
                  reasoning=f"Correct: predicted={predicted_str}",
              )

          # 字符串不匹配时，尝试数值比较（处理 "42" vs "42.00" 等）
          try:
              predicted_val = float(predicted_str)
              expected_val = float(expected_str)
              passed = math.isclose(predicted_val, expected_val, rel_tol=1e-9)
          except ValueError:
              passed = False

          score = 100.0 if passed else 0.0
          return ScoreResult(
              score=score,
              passed=passed,
              details={"predicted": predicted_str, "expected": expected_str},
              reasoning=(
                  f"Correct: predicted={predicted_str}"
                  if passed
                  else f"Incorrect: predicted={predicted_str}, expected={expected_str}"
              ),
          )

      def get_metric_name(self) -> str:
          return "exact_match"
  ```

  - [ ] **Step 2: 验证评分器**

  ```bash
  python -c "
  from benchmark.scorers.exact_match_scorer import ExactMatchScorer
  from benchmark.models.schemas import TaskDefinition

  s = ExactMatchScorer()
  task = TaskDefinition(task_id='t1', dimension='reasoning', dataset='gsm8k', prompt='q', expected_output='5')

  # 精确匹配
  r1 = s.score('The answer is 5', '5', task)
  assert r1.passed, f'Exact match failed: {r1}'

  # 数值等价（42 vs 42.00）
  r2 = s.score('The answer is 42', '42.00', task)
  assert r2.passed, f'Numeric equivalence failed: {r2}'

  # 错误答案
  r3 = s.score('The answer is 6', '5', task)
  assert not r3.passed

  # 无数字
  r4 = s.score('No numbers here!', '5', task)
  assert not r4.passed

  # 负数
  r5 = s.score('Result: -3.14', '-3.14', task)
  assert r5.passed, f'Negative number failed: {r5}'

  print('All scorer tests passed')
  "
  ```

  Expected: `All scorer tests passed`

  - [ ] **Step 3: Commit**

  ```bash
  git add benchmark/scorers/exact_match_scorer.py
  git commit -m "fix(benchmark): numeric comparison in ExactMatchScorer — handle float format and regex edge cases"
  ```

---

- [ ] 7. Fix I5: LLM Adapter 配置缓存 (llm_adapter.py)

  **What to do**:
  - 在 `LLMEvalAdapter.__init__` 中接受 `model` 参数，初始化时加载一次配置
  - `generate()` 方法使用缓存的配置，不再每次调用都读文件
  - 保持向后兼容：`model` 参数可选，如果未传则 `generate()` 时再加载（lazy load）

  **Step-by-step instructions**:

  - [ ] **Step 1: 重写 llm_adapter.py**

  完整替换 `benchmark/core/llm_adapter.py`：

  ```python
  """LLM API 调用适配器.支持 OpenAI 兼容接口（GLM、GPT 等）."""

  from __future__ import annotations

  import time
  from typing import Any

  import requests

  from benchmark.config import get_model_config


  class LLMEvalAdapter:
      """LLM 调用适配器.

      从 models.yaml 加载配置，调用 OpenAI 兼容的 /chat/completions API.
      支持重试（最多 max_retries 次，指数退避）。
      如果初始化时传入 model，配置只加载一次；否则在 generate() 时按需加载。
      """

      def __init__(
          self,
          model: str | None = None,
          max_retries: int = 3,
          timeout: int = 300,
      ) -> None:
          self.max_retries = max_retries
          self.timeout = timeout
          self._model_cache: dict[str, dict[str, Any]] = {}
          # 初始化时预加载模型配置（如果指定了 model）
          if model:
              self._model_cache[model] = get_model_config(model)

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
              model: 模型名称（需在 models.yaml 中配置）.
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

          url = f"{api_base}/chat/completions"
          headers = {
              "Authorization": f"Bearer {api_key}",
              "Content-Type": "application/json",
          }
          payload: dict[str, Any] = {
              "model": model,
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
                      wait = 2**attempt  # 1s, 2s, 4s
                      time.sleep(wait)

          raise ConnectionError(
              f"Failed after {self.max_retries} retries for model '{model}': {last_error}"
          ) from last_error
  ```

  - [ ] **Step 2: 更新 cli.py 中 LLMEvalAdapter 的初始化**

  将 `benchmark/cli.py` evaluate 函数中：
  ```python
  llm = LLMEvalAdapter()
  ```
  改为：
  ```python
  llm = LLMEvalAdapter(model=model)
  ```

  - [ ] **Step 3: 验证配置缓存**

  ```bash
  python -c "
  from benchmark.core.llm_adapter import LLMEvalAdapter
  a = LLMEvalAdapter()
  # 不传 model，lazy load
  assert a._model_cache == {}
  print('LLMEvalAdapter lazy init OK')
  "
  ```

  Expected: `LLMEvalAdapter lazy init OK`

  - [ ] **Step 4: Commit**

  ```bash
  git add benchmark/core/llm_adapter.py benchmark/cli.py
  git commit -m "perf(benchmark): cache model config in LLMEvalAdapter to avoid repeated file reads"
  ```

---

- [ ] 8. Merge commit — 整合所有修复

  **What to do**:
  - 确认所有修复已正确应用
  - 运行完整的语法和导入检查

  **Step-by-step instructions**:

  - [ ] **Step 1: 全量语法检查**

  ```bash
  python -c "
  import ast
  import pathlib
  for f in pathlib.Path('benchmark').rglob('*.py'):
      ast.parse(f.read_text())
  print('All Python files parse OK')
  "
  ```

  Expected: `All Python files parse OK`

  - [ ] **Step 2: 导入检查**

  ```bash
  python -c "
  from benchmark.cli import cli
  from benchmark.models.database import Database
  from benchmark.models.schemas import EvalRun, EvalResult
  from benchmark.scorers.exact_match_scorer import ExactMatchScorer
  from benchmark.scorers.execution_scorer import ExecutionScorer
  from benchmark.core.llm_adapter import LLMEvalAdapter
  from benchmark.config import load_config, get_model_config
  print('All imports OK')
  "
  ```

  Expected: `All imports OK`

  - [ ] **Step 3: CLI help 验证**

  ```bash
  python -m benchmark --help
  ```

  Expected: 显示 evaluate, list-datasets, export 命令

  - [ ] **Step 4: Database 上下文管理器验证**

  ```bash
  python -c "
  from benchmark.models.database import Database
  from benchmark.models.schemas import EvalRun, EvalResult
  from datetime import datetime
  import os

  with Database('benchmark/data/test_final.db') as db:
      db.create_run(EvalRun(run_id='final-1', model='test', dimension='reasoning', dataset='gsm8k', started_at=datetime.now(), status='running'))
      db.save_result(EvalResult(result_id='fr-1', run_id='final-1', task_id='t1', task_content='q', model_output='5', functional_score=100, final_score=100, passed=True, execution_time=0.5, created_at=datetime.now()))
      db.finish_run('final-1')
      rows = db.get_results()
      assert len(rows) == 1
      assert rows[0]['dataset'] == 'gsm8k'

  os.remove('benchmark/data/test_final.db')
  print('Final integration check OK')
  "
  ```

  Expected: `Final integration check OK`

  - [ ] **Step 5: Commit（如有遗漏修复）**

  ```bash
  git add -A
  git status
  # 如果有未提交的变更才 commit
  ```
