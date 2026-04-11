from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun

DATA_DIR = "data"


class Repository(ABC):
    """存储操作的抽象接口。

    覆盖 Database 类的全部公开方法，为不同存储后端（SQLite / 文件系统 / 远程）
    提供统一契约。
    """

    # ── 路径辅助方法 ──

    @staticmethod
    def get_run_dir(run_id: str) -> Path:
        """返回某个 run 的目录路径。"""
        return Path(DATA_DIR) / "runs" / run_id

    @staticmethod
    def get_question_dir(run_id: str, task_id: str) -> Path:
        """返回某个 run 下某道题目的目录路径。"""
        return Repository.get_run_dir(run_id) / "questions" / task_id

    @staticmethod
    def get_scoring_path(run_id: str, task_id: str) -> Path:
        """返回评分结果文件路径。"""
        return Repository.get_question_dir(run_id, task_id) / "scoring.json"

    @staticmethod
    def get_answer_path(run_id: str, task_id: str) -> Path:
        """返回答案文件路径。"""
        return Repository.get_question_dir(run_id, task_id) / "answer.json"

    @staticmethod
    def get_metrics_path(run_id: str, task_id: str) -> Path:
        """返回指标文件路径。"""
        return Repository.get_question_dir(run_id, task_id) / "metrics.json"

    # ── Run 生命周期 ──

    @abstractmethod
    def create_run(self, run: EvalRun) -> str:
        """创建评测运行记录，返回 run_id。"""

    @abstractmethod
    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """标记运行记录为已完成。"""

    @abstractmethod
    def get_active_runs(self) -> list[EvalRun]:
        """获取所有 status 为 running 的运行记录。"""

    # ── Result ──

    @abstractmethod
    def save_answer(self, result: EvalResult) -> str:
        """保存单题评测结果（答案），返回 result_id。"""

    @abstractmethod
    def save_scoring(self, result: EvalResult) -> str:
        """保存单题评分结果，返回 result_id。"""

    @abstractmethod
    def update_status(self, run_id: str, status: str) -> None:
        """更新运行状态。"""

    # ── Metrics ──

    @abstractmethod
    def save_metrics(self, metrics: ApiCallMetrics) -> str:
        """保存 API 调用指标，返回 result_id。"""

    # ── 查询 ──

    @abstractmethod
    def get_results(
        self,
        model: Optional[str] = None,
        dimension: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """查询评测结果列表。"""

    @abstractmethod
    def get_result_detail(self, result_id: str) -> Optional[dict[str, Any]]:
        """获取单条结果的完整详情。"""

    # ── Index ──

    @abstractmethod
    def build_index(self) -> dict[str, Any]:
        """构建/刷新数据索引，返回索引摘要。"""

    # ── Analysis ──

    @abstractmethod
    def save_analysis(self, report: object) -> str:
        """保存分析报告（stability / cluster），返回 report_id。"""

    @abstractmethod
    def save_cluster_report(self, report: object) -> str:
        """保存聚类报告，返回 report_id。"""

    # ── Quality Signals ──

    @abstractmethod
    def save_quality_signals(self, signals: dict[str, Any]) -> str:
        """保存质量信号记录，返回 signal_id。"""

    @abstractmethod
    def get_quality_signals_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """获取某个 run 的所有质量信号。"""

    @abstractmethod
    def get_quality_signals_history(
        self, model: str, days: int = 7
    ) -> list[dict[str, Any]]:
        """获取某模型最近 N 天的质量信号。"""

    # ── Stability Reports ──

    @abstractmethod
    def get_stability_reports(
        self, model: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """查询稳定性报告。"""

    # ── Cluster Reports ──

    @abstractmethod
    def get_cluster_reports(self, model: Optional[str] = None) -> list[dict[str, Any]]:
        """查询聚类报告。"""

    # ── Timing ──

    @abstractmethod
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
        """查询耗时阶段数据。"""

    @abstractmethod
    def get_timing_summaries(
        self,
        model: Optional[str] = None,
        run_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """查询耗时阶段汇总统计。"""

    # ── Pending Scoring Tasks ──

    @abstractmethod
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
        metadata: dict[str, Any] | None = None,
        scoring_dimensions: list[str] | None = None,
    ) -> int:
        """创建待评分任务，返回任务ID。"""

    @abstractmethod
    def fetch_pending_scoring_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        """拉取待处理任务并原子锁定。"""

    @abstractmethod
    def complete_scoring_task(self, task_id: int, score_result: dict[str, Any]) -> None:
        """标记任务完成，保存评分结果。"""

    @abstractmethod
    def fail_scoring_task(self, task_id: int, error_message: str) -> None:
        """标记任务失败。"""

    @abstractmethod
    def retry_scoring_task(self, task_id: int) -> None:
        """将任务重置为 pending 以便重试。"""

    @abstractmethod
    def get_pending_task_count(self) -> int:
        """获取待处理任务数量。"""
