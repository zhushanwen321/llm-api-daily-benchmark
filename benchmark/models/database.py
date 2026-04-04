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

from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun


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
                model_think TEXT DEFAULT '',
                model_answer TEXT DEFAULT '',
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

        cursor.execute("""
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
                model_think, model_answer,
                functional_score, quality_score, final_score, passed,
                details, execution_time, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.result_id,
                result.run_id,
                result.task_id,
                result.task_content,
                result.model_output,
                result.model_think,
                result.model_answer,
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
