"""SQLite 数据库操作。

负责评测结果的持久化存储。使用 sqlite3 标准库，不依赖 ORM。
支持上下文管理器，在 __init__ 时建立连接，close() 时关闭。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun

logger = logging.getLogger(__name__)


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
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取连接。单次 Database 实例生命周期内复用同一连接.

        使用 timeout=30 让 SQLite 在锁冲突时等待最多 30 秒，
        避免多线程并发写入时的 "database is locked" 错误。
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.commit()
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
        with self._lock:
            conn = self._get_conn()
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
            cursor = conn.cursor()

            needs_rebuild = False
            try:
                cursor.execute("SELECT model_think FROM eval_results LIMIT 1")
            except sqlite3.OperationalError:
                needs_rebuild = True

            if not needs_rebuild:
                try:
                    cursor.execute("SELECT expected_output FROM eval_results LIMIT 1")
                except sqlite3.OperationalError:
                    needs_rebuild = True

            if needs_rebuild:
                cursor.execute("DROP TABLE IF EXISTS api_call_metrics")
                cursor.execute("DROP TABLE IF EXISTS eval_results")
                cursor.execute("DROP TABLE IF EXISTS eval_runs")

            metrics_needs_rebuild = False
            try:
                cursor.execute("SELECT reasoning_tokens FROM api_call_metrics LIMIT 1")
            except sqlite3.OperationalError:
                metrics_needs_rebuild = True

            if metrics_needs_rebuild:
                cursor.execute("DROP TABLE IF EXISTS api_call_metrics")

            try:
                cursor.execute("SELECT ttft FROM api_call_metrics LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    cursor.execute(
                        "ALTER TABLE api_call_metrics ADD COLUMN ttft REAL NOT NULL DEFAULT 0"
                    )
                except sqlite3.OperationalError:
                    pass

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
                    expected_output TEXT DEFAULT '',
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
                    ttft REAL NOT NULL DEFAULT 0,
                    ttft_content REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (result_id) REFERENCES eval_results(result_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_signals (
                    signal_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL REFERENCES eval_results(result_id),
                    format_compliance REAL DEFAULT 0,
                    repetition_ratio REAL DEFAULT 0,
                    garbled_text_ratio REAL DEFAULT 0,
                    refusal_detected INTEGER DEFAULT 0,
                    language_consistency REAL DEFAULT 1.0,
                    output_length_zscore REAL DEFAULT 0,
                    thinking_ratio REAL DEFAULT 0,
                    empty_reasoning INTEGER DEFAULT 0,
                    truncated INTEGER DEFAULT 0,
                    token_efficiency_zscore REAL DEFAULT 0,
                    tps_zscore REAL DEFAULT 0,
                    ttft_zscore REAL DEFAULT 0,
                    answer_entropy REAL DEFAULT 0,
                    raw_output_length INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_qs_result_id ON quality_signals(result_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_qs_created_at ON quality_signals(created_at)"
            )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stability_reports (
                    report_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    anomalies TEXT NOT NULL DEFAULT '[]',
                    change_points TEXT NOT NULL DEFAULT '[]',
                    stat_tests TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sr_model ON stability_reports(model)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sr_run_id ON stability_reports(run_id)"
            )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster_reports (
                    report_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    n_clusters INTEGER NOT NULL,
                    n_noise INTEGER NOT NULL DEFAULT 0,
                    clusters TEXT NOT NULL DEFAULT '[]',
                    suspected_changes TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_cr_model ON cluster_reports(model)"
            )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timing_phases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    phase_name TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    wait_time REAL NOT NULL,
                    active_time REAL NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tp_model ON timing_phases(model)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tp_run_id ON timing_phases(run_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tp_result_id ON timing_phases(result_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tp_phase_name ON timing_phases(phase_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tp_created_at ON timing_phases(created_at)"
            )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_scoring_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    dimension TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    expected_output TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    model_answer TEXT NOT NULL,
                    reasoning_content TEXT DEFAULT '',
                    test_cases TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    scoring_dimensions TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    score_result TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    processing_started_at TEXT,
                    processing_finished_at TEXT,
                    result_id TEXT NOT NULL,
                    run_id TEXT NOT NULL
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_pst_status_created ON pending_scoring_tasks(status, created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_pst_result_id ON pending_scoring_tasks(result_id)"
            )

            conn.commit()

    def create_run(self, run: EvalRun) -> str:
        """创建评测运行记录。"""
        with self._lock:
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
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE eval_runs SET finished_at = ?, status = ? WHERE run_id = ?",
                (datetime.now().isoformat(), status, run_id),
            )
            conn.commit()

    def save_result(self, result: EvalResult) -> str:
        """保存单题评测结果。"""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO eval_results
                   (result_id, run_id, task_id, task_content, model_output,
                    model_think, model_answer, expected_output,
                    functional_score, quality_score, final_score, passed,
                    details, execution_time, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.result_id,
                    result.run_id,
                    result.task_id,
                    result.task_content,
                    result.model_output,
                    result.model_think,
                    result.model_answer,
                    result.expected_output,
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
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO api_call_metrics
                   (result_id, prompt_tokens, completion_tokens,
                    reasoning_tokens, reasoning_content,
                    duration, tokens_per_second, ttft, ttft_content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    metrics.result_id,
                    metrics.prompt_tokens,
                    metrics.completion_tokens,
                    metrics.reasoning_tokens,
                    metrics.reasoning_content,
                    metrics.duration,
                    metrics.tokens_per_second,
                    metrics.ttft,
                    metrics.ttft_content,
                    metrics.created_at.isoformat(),
                ),
            )
            conn.commit()
            return metrics.result_id

    async def asave_result(self, result: EvalResult) -> str:
        """异步保存单题评测结果。"""
        return await asyncio.to_thread(self.save_result, result)

    async def asave_metrics(self, metrics: ApiCallMetrics) -> str:
        """异步保存 API 调用指标。"""
        return await asyncio.to_thread(self.save_metrics, metrics)

    def get_results(
        self,
        model: Optional[str] = None,
        dimension: Optional[str] = None,
        run_id: Optional[str] = None,
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
        if run_id:
            query += " AND r.run_id = ?"
            params.append(run_id)
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

    # ── quality_signals ──

    def _save_quality_signals(self, signals: dict) -> str:
        """同步写入一条 quality_signals 记录。"""
        with self._lock:
            conn = self._get_conn()
            signal_id = str(uuid.uuid4())[:12]
            conn.execute(
                """INSERT INTO quality_signals
                   (signal_id, result_id, format_compliance, repetition_ratio,
                    garbled_text_ratio, refusal_detected, language_consistency,
                    output_length_zscore, thinking_ratio, empty_reasoning,
                    truncated, token_efficiency_zscore, tps_zscore,
                    ttft_zscore, answer_entropy, raw_output_length)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_id,
                    signals["result_id"],
                    signals.get("format_compliance", 0),
                    signals.get("repetition_ratio", 0),
                    signals.get("garbled_text_ratio", 0),
                    int(signals.get("refusal_detected", 0)),
                    signals.get("language_consistency", 1.0),
                    signals.get("output_length_zscore", 0),
                    signals.get("thinking_ratio", 0),
                    int(signals.get("empty_reasoning", 0)),
                    int(signals.get("truncated", 0)),
                    signals.get("token_efficiency_zscore", 0),
                    signals.get("tps_zscore", 0),
                    signals.get("ttft_zscore", 0),
                    signals.get("answer_entropy", 0),
                    signals.get("raw_output_length", 0),
                ),
            )
            conn.commit()
            return signal_id

    async def asave_quality_signals(self, signals: dict) -> str:
        """异步写入 quality_signals 记录。"""
        return await asyncio.to_thread(self._save_quality_signals, signals)

    # ── stability_reports ──

    def _save_stability_report(self, report: object) -> str:
        """同步写入 stability_reports 记录。report 需要有 StabilityReport 的字段。"""
        from benchmark.analysis.models import StabilityReport

        assert isinstance(report, StabilityReport)
        with self._lock:
            conn = self._get_conn()
            report_id = str(uuid.uuid4())[:12]
            anomalies = json.dumps(
                [
                    {
                        "signal_name": a.signal_name,
                        "current_value": a.current_value,
                        "baseline_mean": a.baseline_mean,
                        "baseline_std": a.baseline_std,
                        "z_score": a.z_score,
                    }
                    for a in report.anomalies
                ],
                ensure_ascii=False,
            )
            change_points = json.dumps(
                [
                    {
                        "signal_name": cp.signal_name,
                        "detected_at": cp.detected_at.isoformat(),
                        "direction": cp.direction,
                        "magnitude": cp.magnitude,
                    }
                    for cp in report.change_points
                ],
                ensure_ascii=False,
            )
            conn.execute(
                """INSERT INTO stability_reports
                   (report_id, model, run_id, overall_status,
                    anomalies, change_points, stat_tests, summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_id,
                    report.model,
                    report.run_id,
                    report.overall_status,
                    anomalies,
                    change_points,
                    json.dumps(report.stat_tests, ensure_ascii=False),
                    report.summary,
                ),
            )
            conn.commit()
            return report_id

    async def asave_stability_report(self, report: object) -> str:
        """异步写入 stability_reports 记录。"""
        return await asyncio.to_thread(self._save_stability_report, report)

    # ── 查询方法 ──

    def _get_quality_signals_for_run(self, run_id: str) -> list[dict]:
        """同步：通过 eval_results JOIN 获取某个 run 的所有质量信号。"""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT qs.*
               FROM quality_signals qs
               JOIN eval_results er ON qs.result_id = er.result_id
               WHERE er.run_id = ?""",
            (run_id,),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    async def aget_quality_signals_for_run(self, run_id: str) -> list[dict]:
        """异步：获取某个 run 的所有质量信号。"""
        return await asyncio.to_thread(self._get_quality_signals_for_run, run_id)

    def _get_quality_signals_history(self, model: str, days: int = 7) -> list[dict]:
        """同步：获取某模型最近 N 天的质量信号。"""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT qs.*
               FROM quality_signals qs
               JOIN eval_results er ON qs.result_id = er.result_id
               JOIN eval_runs ev ON er.run_id = ev.run_id
               WHERE ev.model = ?
                 AND qs.created_at >= datetime('now', ?)
               ORDER BY qs.created_at DESC""",
            (model, f"-{days} days"),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    async def aget_quality_signals_history(
        self, model: str, days: int = 7
    ) -> list[dict]:
        """异步：获取某模型最近 N 天的质量信号。"""
        return await asyncio.to_thread(self._get_quality_signals_history, model, days)

    def _get_stability_reports(self, model: str | None = None) -> list[dict]:
        """同步：查询稳定性报告。"""
        conn = self._get_conn()
        if model:
            cursor = conn.execute(
                """SELECT * FROM stability_reports
                   WHERE model = ?
                   ORDER BY created_at DESC""",
                (model,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM stability_reports ORDER BY created_at DESC"
            )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    async def aget_stability_reports(self, model: str | None = None) -> list[dict]:
        """异步：查询稳定性报告。"""
        return await asyncio.to_thread(self._get_stability_reports, model)

    # ── cluster_reports ──

    def _save_cluster_report(self, report: object) -> str:
        """同步写入 cluster_reports 记录。"""
        from benchmark.analysis.models import ClusterReport

        assert isinstance(report, ClusterReport)
        with self._lock:
            conn = self._get_conn()
            report_id = str(uuid.uuid4())[:12]
            clusters = json.dumps(
                [
                    {
                        "cluster_id": c.cluster_id,
                        "size": c.size,
                        "time_range": c.time_range,
                        "centroid": c.centroid,
                        "avg_score": c.avg_score,
                    }
                    for c in report.clusters
                ],
                ensure_ascii=False,
            )
            conn.execute(
                """INSERT INTO cluster_reports
                   (report_id, model, n_clusters, n_noise,
                    clusters, suspected_changes, summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_id,
                    report.model,
                    report.n_clusters,
                    report.n_noise,
                    clusters,
                    json.dumps(report.suspected_changes, ensure_ascii=False),
                    report.summary,
                ),
            )
            conn.commit()
            return report_id

    async def asave_cluster_report(self, report: object) -> str:
        """异步写入 cluster_reports 记录。"""
        return await asyncio.to_thread(self._save_cluster_report, report)

    def _get_cluster_reports(self, model: str | None = None) -> list[dict]:
        """同步：查询聚类报告。"""
        conn = self._get_conn()
        if model:
            cursor = conn.execute(
                """SELECT * FROM cluster_reports
                   WHERE model = ?
                   ORDER BY created_at DESC""",
                (model,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM cluster_reports ORDER BY created_at DESC"
            )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    async def aget_cluster_reports(self, model: str | None = None) -> list[dict]:
        """异步：查询聚类报告。"""
        return await asyncio.to_thread(self._get_cluster_reports, model)

    # ── timing_phases ──

    def get_timing_phases(
        self,
        model: Optional[str] = None,
        run_id: Optional[str] = None,
        result_id: Optional[str] = None,
        phase_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """查询耗时阶段数据。

        Args:
            model: 按模型名称筛选
            run_id: 按运行 ID 筛选
            result_id: 按结果 ID 筛选
            phase_name: 按阶段名称筛选
            start_date: 筛选此日期之后的记录
            end_date: 筛选此日期之前的记录
            limit: 返回记录数上限

        Returns:
            包含 timing_phases 数据的 DataFrame，metadata 列已解析为 dict
        """
        conn = self._get_conn()
        query = "SELECT * FROM timing_phases WHERE 1=1"
        params: list = []

        if model:
            query += " AND model = ?"
            params.append(model)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if result_id:
            query += " AND result_id = ?"
            params.append(result_id)
        if phase_name:
            query += " AND phase_name = ?"
            params.append(phase_name)
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)

        # 解析 metadata JSON 字段
        if not df.empty and "metadata" in df.columns:

            def safe_json_loads(x):
                try:
                    return json.loads(x) if x else {}
                except json.JSONDecodeError:
                    logger.warning("Failed to parse metadata JSON: %s", x)
                    return {}

            df["metadata"] = df["metadata"].apply(safe_json_loads)

        return df

    def get_timing_summaries(
        self,
        model: Optional[str] = None,
        run_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """查询耗时阶段汇总统计。

        按 phase_name 分组，统计 duration/wait_time/active_time 的均值、样本数。

        Args:
            model: 按模型名称筛选
            run_id: 按运行 ID 筛选
            start_date: 筛选此日期之后的记录
            end_date: 筛选此日期之前的记录

        Returns:
            包含各 phase_name 统计信息的 DataFrame
        """
        conn = self._get_conn()
        query = """SELECT
                phase_name,
                COUNT(*) as count,
                AVG(duration) as avg_duration,
                AVG(wait_time) as avg_wait_time,
                AVG(active_time) as avg_active_time,
                MIN(duration) as min_duration,
                MAX(duration) as max_duration,
                SUM(duration) as total_duration
            FROM timing_phases WHERE 1=1"""
        params: list = []

        if model:
            query += " AND model = ?"
            params.append(model)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())

        query += " GROUP BY phase_name ORDER BY phase_name"

        df = pd.read_sql_query(query, conn, params=params)
        return df

    # ── pending_scoring_tasks ──

    def create_scoring_task(
        self,
        result_id: str,
        run_id: str,
        task_id: str,
        dimension: str,
        dataset: str,
        prompt: str,
        expected_output: str,
        model_output: str,
        model_answer: str,
        reasoning_content: str = "",
        test_cases: list[str] | None = None,
        metadata: dict | None = None,
        scoring_dimensions: list[str] | None = None,
    ) -> int:
        """创建待评分任务，返回任务ID。"""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()
            test_cases = test_cases if test_cases is not None else []
            metadata = metadata if metadata is not None else {}
            scoring_dimensions = (
                scoring_dimensions if scoring_dimensions is not None else []
            )

            cursor = conn.execute(
                """INSERT INTO pending_scoring_tasks
                   (result_id, run_id, task_id, dimension, dataset, prompt,
                    expected_output, model_output, model_answer, reasoning_content,
                    test_cases, metadata, scoring_dimensions, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_id,
                    run_id,
                    task_id,
                    dimension,
                    dataset,
                    prompt,
                    expected_output,
                    model_output,
                    model_answer,
                    reasoning_content,
                    json.dumps(test_cases, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(scoring_dimensions, ensure_ascii=False),
                    "pending",
                    now,
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def fetch_pending_scoring_tasks(self, limit: int = 10) -> list[dict]:
        """拉取待处理任务并原子锁定（设为processing）。"""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            cursor = conn.execute(
                "SELECT id FROM pending_scoring_tasks WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
                (limit,),
            )
            ids = [row[0] for row in cursor.fetchall()]

            if not ids:
                return []

            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE pending_scoring_tasks SET status = 'processing', processing_started_at = ?, updated_at = ? WHERE id IN ({placeholders})",
                [now, now] + ids,
            )
            conn.commit()

            cursor = conn.execute(
                f"SELECT * FROM pending_scoring_tasks WHERE id IN ({placeholders})",
                ids,
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def complete_scoring_task(self, task_id: int, score_result: dict) -> None:
        """标记任务完成，保存评分结果。"""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()
            conn.execute(
                """UPDATE pending_scoring_tasks
                   SET status = 'completed',
                       score_result = ?,
                       processing_finished_at = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (json.dumps(score_result, ensure_ascii=False), now, now, task_id),
            )
            conn.commit()

    def fail_scoring_task(self, task_id: int, error_message: str) -> None:
        """标记任务失败。"""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()
            conn.execute(
                """UPDATE pending_scoring_tasks
                   SET status = 'failed',
                       error_message = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (error_message, now, task_id),
            )
            conn.commit()

    def retry_scoring_task(self, task_id: int) -> None:
        """将任务重置为pending以便重试。"""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()
            conn.execute(
                """UPDATE pending_scoring_tasks
                   SET status = 'pending',
                       retry_count = retry_count + 1,
                       processing_started_at = NULL,
                       updated_at = ?
                   WHERE id = ?""",
                (now, task_id),
            )
            conn.commit()

    def get_pending_task_count(self) -> int:
        """获取待处理任务数量。"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM pending_scoring_tasks WHERE status = 'pending'"
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    async def acreate_scoring_task(
        self,
        result_id: str,
        run_id: str,
        task_id: str,
        dimension: str,
        dataset: str,
        prompt: str,
        expected_output: str,
        model_output: str,
        model_answer: str,
        reasoning_content: str = "",
        test_cases: list[str] | None = None,
        metadata: dict | None = None,
        scoring_dimensions: list[str] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.create_scoring_task,
            result_id,
            run_id,
            task_id,
            dimension,
            dataset,
            prompt,
            expected_output,
            model_output,
            model_answer,
            reasoning_content,
            test_cases,
            metadata,
            scoring_dimensions,
        )

    async def afetch_pending_scoring_tasks(self, limit: int = 10) -> list[dict]:
        return await asyncio.to_thread(self.fetch_pending_scoring_tasks, limit)

    async def acomplete_scoring_task(self, task_id: int, score_result: dict) -> None:
        await asyncio.to_thread(self.complete_scoring_task, task_id, score_result)

    async def afail_scoring_task(self, task_id: int, error_message: str) -> None:
        await asyncio.to_thread(self.fail_scoring_task, task_id, error_message)

    async def aretry_scoring_task(self, task_id: int) -> None:
        await asyncio.to_thread(self.retry_scoring_task, task_id)

    async def aget_pending_task_count(self) -> int:
        return await asyncio.to_thread(self.get_pending_task_count)
