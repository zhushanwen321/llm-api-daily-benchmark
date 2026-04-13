"""FileRepository - 基于文件系统的 Repository 实现。

组合所有 handler 提供统一的存储接口。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmark.core.tz import now

import pandas as pd

from benchmark.analysis.models import ClusterReport, StabilityReport
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun
from benchmark.repository.handlers import (
    AnalysisHandler,
    AnswerHandler,
    ClusterHandler,
    ExecutionLogHandler,
    MetadataHandler,
    ScoringHandler,
    StatusHandler,
    TimingHandler,
)
from benchmark.repository.index_builder import IndexBuilder
from benchmark.repository.interface import Repository


class FileRepository(Repository):
    """基于文件系统的 Repository 实现。

    内部组合使用所有 handler 来管理不同维度的数据存储：
    - StatusHandler: status.json 管理
    - MetadataHandler: metadata.jsonl 管理
    - AnswerHandler: answer.jsonl 管理
    - ScoringHandler: scoring.jsonl 管理
    - TimingHandler: timing.jsonl 管理
    - ExecutionLogHandler: execution.log 管理
    - AnalysisHandler: analysis.jsonl 管理
    - ClusterHandler: cluster_reports.jsonl 管理
    """

    def __init__(
        self,
        data_root: Path | str | None = None,
        execution_id: str | None = None,
    ) -> None:
        root_value = data_root or os.getenv("DATA_ROOT", "data")
        self._root = Path(root_value)
        self._root.mkdir(parents=True, exist_ok=True)

        # 生成或接收 execution_id，用于区分不同的 CLI 执行批次
        self._execution_id = execution_id or self._generate_execution_id()
        self._exec_dir = self._root / self._execution_id
        # exec_dir 延迟创建：仅在首次写入时 mkdir，避免只读场景产生空目录
        self._exec_dir_created = False

        # handlers 的 root 指向 execution 目录，自动按批次隔离
        self._status = StatusHandler(self._exec_dir)
        self._metadata = MetadataHandler(self._exec_dir)
        self._answer = AnswerHandler(self._exec_dir)
        self._scoring = ScoringHandler(self._exec_dir)
        self._timing = TimingHandler(self._exec_dir)
        self._execution_log = ExecutionLogHandler(self._exec_dir)
        self._analysis = AnalysisHandler(self._exec_dir)
        self._cluster = ClusterHandler(self._root)

        # IndexBuilder 的 root 仍是 data/，扫描所有 execution 目录
        self._index_builder = IndexBuilder(self._root)

        # scoring task ID → (benchmark_id, question_id) 缓存
        self._task_location_cache: dict[int, tuple[str, str]] = {}

    def _ensure_exec_dir(self) -> None:
        if not self._exec_dir_created:
            self._exec_dir.mkdir(parents=True, exist_ok=True)
            self._exec_dir_created = True

    # ── Run 生命周期 ──

    @property
    def data_root(self) -> Path:
        return self._root

    def create_benchmark_run(
        self,
        model: str,
        dimension: str,
        dataset: str,
        questions: list[str],
    ) -> str:
        """创建评测运行记录，返回 benchmark_id。

        Args:
            model: 模型名称
            dimension: 评测维度
            dataset: 数据集名称
            questions: 问题 ID 列表

        Returns:
            benchmark_id: 生成的运行 ID
        """
        benchmark_id = self._generate_benchmark_id(model, dimension)
        total_questions = len(questions)
        self._ensure_exec_dir()

        # 创建 status.json
        self._status.create(
            benchmark_id=benchmark_id,
            model=model,
            dimension=dimension,
            total_questions=total_questions,
        )

        # 写入 metadata.jsonl
        self._metadata.write(
            benchmark_id=benchmark_id,
            model=model,
            dimension=dimension,
            dataset=dataset,
        )

        return benchmark_id

    def save_question_result(
        self,
        benchmark_id: str,
        question_id: str,
        answer_data: dict[str, Any],
        api_metrics: dict[str, Any] | None = None,
    ) -> str:
        """保存单题答案结果。

        Args:
            benchmark_id: 评测运行 ID
            question_id: 问题 ID
            answer_data: 答案数据字典
            api_metrics: API 调用指标（可选）

        Returns:
            result_id: 结果记录 ID
        """
        now_ts = now()

        result = EvalResult(
            result_id=answer_data.get("result_id", self._generate_id()),
            run_id=benchmark_id,
            task_id=question_id,
            task_content=answer_data.get("task_content", ""),
            model_output=answer_data.get("model_output", ""),
            model_think=answer_data.get("model_think", ""),
            model_answer=answer_data.get("model_answer", ""),
            expected_output=answer_data.get("expected_output", ""),
            functional_score=answer_data.get("functional_score", 0.0),
            quality_score=answer_data.get("quality_score", 0.0),
            final_score=answer_data.get("final_score", 0.0),
            passed=answer_data.get("passed", False),
            details=answer_data.get("details", {}),
            execution_time=answer_data.get("execution_time", 0.0),
            created_at=now_ts,
        )

        metrics = None
        if api_metrics:
            metrics = ApiCallMetrics(
                result_id=result.result_id,
                prompt_tokens=api_metrics.get("prompt_tokens", 0),
                completion_tokens=api_metrics.get("completion_tokens", 0),
                reasoning_tokens=api_metrics.get("reasoning_tokens", 0),
                reasoning_content=api_metrics.get("reasoning_content", ""),
                duration=api_metrics.get("duration", 0.0),
                tokens_per_second=api_metrics.get("tokens_per_second", 0.0),
                ttft=api_metrics.get("ttft", 0.0),
                ttft_content=api_metrics.get("ttft_content", 0.0),
                created_at=now_ts,
            )

        # 保存答案
        result_id = self._answer.save_answer(result, metrics)

        # 更新 status
        self._status.increment_answered(benchmark_id)

        return result_id

    def save_question_scoring(
        self,
        benchmark_id: str,
        question_id: str,
        scoring_data: dict[str, Any],
        quality_signals: dict[str, Any] | None = None,
    ) -> None:
        """保存单题评分结果。

        Args:
            benchmark_id: 评测运行 ID
            question_id: 问题 ID
            scoring_data: 评分数据字典
            quality_signals: 质量信号（可选）
        """
        # 合并 quality_signals 到 scoring_data
        data = dict(scoring_data)
        if quality_signals:
            data["quality_signals"] = quality_signals

        # 确保必要字段存在
        if "quality_signals" not in data:
            data["quality_signals"] = {}
        if "scoring_status" not in data:
            data["scoring_status"] = "completed"

        # 保存评分
        self._scoring.save_scoring(benchmark_id, question_id, data)

        # 更新 status
        self._status.increment_scored(benchmark_id)

    def save_timing(
        self,
        benchmark_id: str,
        question_id: str,
        timing_data: list[dict[str, Any]],
    ) -> None:
        """保存耗时数据。

        Args:
            benchmark_id: 评测运行 ID
            question_id: 问题 ID
            timing_data: 耗时阶段数据列表
        """
        self._timing.save_timing(benchmark_id, question_id, timing_data)

    def save_analysis_data(
        self,
        benchmark_id: str,
        analysis_data: dict[str, Any],
    ) -> str:
        """保存分析报告。

        Args:
            benchmark_id: 评测运行 ID
            analysis_data: 分析报告数据

        Returns:
            report_id: 报告 ID
        """
        return self._analysis.save_analysis(benchmark_id, analysis_data)

    def is_run_completed(self, benchmark_id: str) -> bool:
        """检查运行是否已完成。

        Args:
            benchmark_id: 评测运行 ID

        Returns:
            是否已完成
        """
        return self._status.is_completed(benchmark_id)

    def get_active_benchmark_runs(self) -> list[EvalRun]:
        """获取所有未完成的运行记录。"""
        active_runs: list[EvalRun] = []

        for run_dir in self._iter_run_dirs():
            try:
                status = self._status.get(run_dir.name)
                if status.get("status") not in ("completed", "failed"):
                    metadata = self._metadata.read(run_dir.name)
                    dataset = metadata.get("dataset", "")

                    active_runs.append(
                        EvalRun(
                            run_id=run_dir.name,
                            model=status.get("model", ""),
                            dimension=status.get("dimension", ""),
                            dataset=dataset,
                            started_at=datetime.fromisoformat(
                                status.get("created_at", "")
                            ),
                            status=status.get("status", "running"),
                        )
                    )
            except FileNotFoundError:
                continue

        return active_runs

    def build_index(self) -> dict[str, Any]:
        """构建/刷新数据索引。

        Returns:
            索引摘要信息
        """
        self._index_builder.build()

        index_path = self._root / "index.jsonl"
        row_count = len(self._read_jsonl(index_path))

        return {
            "index_file": str(index_path),
            "row_count": row_count,
            "data_root": str(self._root),
        }

    # ── Repository 接口实现 ──

    def create_run(self, run: EvalRun) -> str:
        """创建评测运行记录（Repository 接口实现）。"""
        return self.create_benchmark_run(
            model=run.model,
            dimension=run.dimension,
            dataset=run.dataset,
            questions=[],  # total_questions=0，需显式 finish_run 标记完成
        )

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """标记运行记录为已完成或失败。"""
        if status == "failed":
            self._status.set_failed(run_id)
        else:
            self._status.set_completed(run_id)

    def get_active_runs(self) -> list[EvalRun]:
        """获取所有 status 为 running 的运行记录（Repository 接口实现）。"""
        return self.get_active_benchmark_runs()

    def save_answer(self, result: EvalResult) -> str:
        """保存单题评测结果（答案）。"""
        answer_data = {
            "result_id": result.result_id,
            "task_content": result.task_content,
            "model_output": result.model_output,
            "model_think": result.model_think,
            "model_answer": result.model_answer,
            "expected_output": result.expected_output,
            "functional_score": result.functional_score,
            "quality_score": result.quality_score,
            "final_score": result.final_score,
            "passed": result.passed,
            "details": result.details,
            "execution_time": result.execution_time,
        }

        return self.save_question_result(
            benchmark_id=result.run_id,
            question_id=result.task_id,
            answer_data=answer_data,
            api_metrics=None,
        )

    def save_scoring(self, result: EvalResult) -> str:
        """保存单题评分结果。"""
        scoring_data = {
            "task_id": result.task_id,
            "functional_score": result.functional_score,
            "quality_score": result.quality_score,
            "final_score": result.final_score,
            "passed": result.passed,
            "details": result.details,
            "reasoning": "",
        }

        self.save_question_scoring(
            benchmark_id=result.run_id,
            question_id=result.task_id,
            scoring_data=scoring_data,
            quality_signals={},
        )

        return result.result_id

    def update_status(self, run_id: str, status: str) -> None:
        """更新运行状态（通过 finish_run 实现）。"""
        if status in ("completed", "failed"):
            self.finish_run(run_id, status)

    def save_metrics(
        self,
        metrics: ApiCallMetrics,
        run_id: str = "",
        task_id: str = "",
    ) -> str:
        """保存 API 调用指标。

        注意：metrics 通常在 save_answer 时一并保存，
        此方法用于单独保存 metrics 的场景。

        Args:
            metrics: API 调用指标
            run_id: 评测运行 ID（FileRepository 需要此参数确定写入路径）
            task_id: 问题 ID（FileRepository 需要此参数确定写入路径）
        """
        if not run_id or not task_id:
            # 无法确定写入路径，跳过——metrics 可能已通过 save_answer 一并保存
            return metrics.result_id

        result = EvalResult(
            result_id=metrics.result_id,
            run_id=run_id,
            task_id=task_id,
            task_content="",
            model_output="",
            model_answer="",
            expected_output="",
            functional_score=0.0,
            quality_score=0.0,
            final_score=0.0,
            passed=False,
            execution_time=metrics.duration,
            created_at=metrics.created_at,
        )

        self._answer.save_answer(result, metrics)
        return metrics.result_id

    def get_results(
        self,
        model: Optional[str] = None,
        dimension: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """查询评测结果列表。"""
        results: list[dict[str, Any]] = []

        run_dirs = (
            [d for d in self._iter_run_dirs() if d.name == run_id]
            if run_id
            else self._iter_run_dirs()
        )

        for run_dir in run_dirs:
            benchmark_id = run_dir.name
            try:
                status = self._read_json(run_dir / "status.json")

                if model and status.get("model") != model:
                    continue
                if dimension and status.get("dimension") != dimension:
                    continue

                answers = self._read_answers_from_dir(run_dir)

                for ans in answers:
                    answer_data = ans.get("answer", {})
                    result = {
                        "result_id": answer_data.get("result_id"),
                        "run_id": benchmark_id,
                        "task_id": answer_data.get("task_id"),
                        "model": status.get("model"),
                        "dimension": status.get("dimension"),
                        "model_output": answer_data.get("model_output"),
                        "model_answer": answer_data.get("model_answer"),
                        "final_score": answer_data.get("final_score"),
                        "passed": answer_data.get("passed"),
                        "execution_time": answer_data.get("execution_time"),
                        "created_at": status.get("created_at"),
                        "status": status.get("status"),
                    }
                    results.append(result)

            except FileNotFoundError:
                continue

        return results

    def get_result_detail(self, result_id: str) -> Optional[dict[str, Any]]:
        """获取单条结果的完整详情。"""
        for run_dir in self._iter_run_dirs():
            benchmark_id = run_dir.name
            answers = self._read_answers_from_dir(run_dir)

            for ans in answers:
                answer_data = ans.get("answer", {})
                if answer_data.get("result_id") == result_id:
                    try:
                        status = self._read_json(run_dir / "status.json")
                        model = status.get("model", "")
                        dimension = status.get("dimension", "")
                    except FileNotFoundError:
                        model = ""
                        dimension = ""

                    # 从 metadata.jsonl 读取 dataset 信息
                    try:
                        meta_path = run_dir / "metadata.jsonl"
                        meta_records = self._read_jsonl(meta_path)
                        meta = meta_records[-1] if meta_records else {}
                    except (FileNotFoundError, IndexError):
                        meta = {}

                    return {
                        "result_id": result_id,
                        "run_id": benchmark_id,
                        "task_id": answer_data.get("task_id"),
                        "task_content": answer_data.get("task_content"),
                        "model": model,
                        "dimension": dimension,
                        "dataset": meta.get("dataset", ""),
                        "metadata": {
                            "dataset": meta.get("dataset", ""),
                            "dimension": dimension,
                        },
                        "model_output": answer_data.get("model_output"),
                        "model_think": answer_data.get("model_think"),
                        "model_answer": answer_data.get("model_answer"),
                        "expected_output": answer_data.get("expected_output"),
                        "functional_score": answer_data.get("functional_score"),
                        "quality_score": answer_data.get("quality_score"),
                        "final_score": answer_data.get("final_score"),
                        "passed": answer_data.get("passed"),
                        "details": answer_data.get("details"),
                        "execution_time": answer_data.get("execution_time"),
                        "api_metrics": ans.get("api_metrics"),
                    }

        return None

    def get_runs(
        self,
        model: Optional[str] = None,
        dimension: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """获取运行列表（使用 IndexBuilder）。"""
        self._index_builder.build()

        index_path = self._root / "index.jsonl"
        all_rows = self._read_jsonl(index_path)

        rows: list[dict[str, Any]] = []
        for row in all_rows:
            if model and row.get("model") != model:
                continue
            if dimension and row.get("dimension") != dimension:
                continue
            if status_filter and row.get("status") != status_filter:
                continue

            rows.append(row)

        return rows

    def get_trend_data(
        self,
        model: str,
        dimension: Optional[str] = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """获取趋势数据。"""
        # 获取该模型的所有 runs
        runs = self.get_runs(model=model, dimension=dimension)

        # 过滤最近 N 天的数据
        cutoff = now().timestamp() - days * 24 * 3600

        trends: list[dict[str, Any]] = []
        for run in runs:
            created_at = run.get("created_at", "")
            if not created_at:
                continue

            try:
                run_time = datetime.fromisoformat(created_at).timestamp()
                if run_time < cutoff:
                    continue
            except ValueError:
                continue

            trends.append(
                {
                    "benchmark_id": run.get("benchmark_id"),
                    "model": run.get("model"),
                    "dimension": run.get("dimension"),
                    "avg_score": run.get("avg_score"),
                    "created_at": created_at,
                    "status": run.get("status"),
                }
            )

        # 按时间排序
        trends.sort(key=lambda x: x.get("created_at", ""))

        return trends

    def save_analysis(self, report: object) -> str:
        """保存分析报告（Repository 接口实现）。"""
        if isinstance(report, StabilityReport):
            analysis_data = {
                "report_type": "stability",
                "model": getattr(report, "model", ""),
                "benchmark_id": getattr(report, "run_id", ""),
                "overall_status": getattr(report, "overall_status", "stable"),
                "anomalies": [
                    {
                        "signal_name": a.signal_name,
                        "current_value": a.current_value,
                        "baseline_mean": a.baseline_mean,
                        "baseline_std": a.baseline_std,
                        "z_score": a.z_score,
                    }
                    for a in getattr(report, "anomalies", [])
                ],
                "change_points": [
                    {
                        "signal_name": c.signal_name,
                        "detected_at": c.detected_at.isoformat(),
                        "direction": c.direction,
                        "magnitude": c.magnitude,
                    }
                    for c in getattr(report, "change_points", [])
                ],
                "stat_tests": getattr(report, "stat_tests", []),
                "summary": getattr(report, "summary", ""),
                "created_at": getattr(report, "created_at", now()).isoformat(),
            }
            return self.save_analysis_data(
                benchmark_id=analysis_data["benchmark_id"],
                analysis_data=analysis_data,
            )
        elif isinstance(report, dict):
            bid = report.get("benchmark_id", "")
            return self.save_analysis_data(bid, report)
        else:
            raise ValueError(f"Unsupported report type: {type(report)}")

    def save_cluster_report(self, report: object) -> str:
        """保存聚类报告。"""
        if isinstance(report, ClusterReport):
            self._cluster.save_report(report)
            return f"cluster_{report.model}_{int(now().timestamp())}"
        else:
            raise ValueError(f"Unsupported report type: {type(report)}")

    def save_quality_signals(self, signals: dict[str, Any]) -> str:
        """保存质量信号记录。"""
        # 质量信号通常在 scoring 时保存
        benchmark_id = signals.get("benchmark_id", "")
        question_id = signals.get("question_id", "")

        if benchmark_id and question_id:
            # 更新对应的 scoring 记录
            scoring_data = {
                "task_id": question_id,
                "functional_score": signals.get("functional_score", 0.0),
                "quality_score": signals.get("quality_score", 0.0),
                "final_score": signals.get("final_score", 0.0),
                "passed": signals.get("passed", False),
                "quality_signals": signals.get("signals", {}),
                "scoring_status": "completed",
            }
            self._scoring.save_scoring(benchmark_id, question_id, scoring_data)

        return signals.get("signal_id", self._generate_id())

    def get_quality_signals_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """获取某个 run 的所有质量信号。"""
        signals: list[dict[str, Any]] = []

        bench_dir = self._exec_dir / run_id
        if not bench_dir.exists():
            return signals

        for question_dir in bench_dir.iterdir():
            if not question_dir.is_dir():
                continue

            scoring_path = question_dir / "scoring.jsonl"
            for record in self._read_jsonl(scoring_path):
                quality_signals = record.get("quality_signals", {})
                if quality_signals:
                    signals.append(
                        {
                            "benchmark_id": run_id,
                            "question_id": question_dir.name,
                            "quality_signals": quality_signals,
                            "final_score": record.get("final_score"),
                            "scoring_status": record.get("scoring_status"),
                        }
                    )

        return signals

    def get_quality_signals_history(
        self, model: str, days: int = 7
    ) -> list[dict[str, Any]]:
        """获取某模型最近 N 天的质量信号。"""
        # 获取该模型的所有 runs
        runs = self.get_runs(model=model)

        # 过滤最近 N 天
        cutoff = now().timestamp() - days * 24 * 3600

        history: list[dict[str, Any]] = []
        for run in runs:
            created_at = run.get("created_at", "")
            if not created_at:
                continue

            try:
                run_time = datetime.fromisoformat(created_at).timestamp()
                if run_time < cutoff:
                    continue
            except ValueError:
                continue

            benchmark_id = run.get("benchmark_id", "")
            signals = self.get_quality_signals_for_run(benchmark_id)

            for sig in signals:
                sig["run_created_at"] = created_at
                history.append(sig)

        return history

    def get_stability_reports(
        self, model: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """查询稳定性报告。"""
        reports: list[dict[str, Any]] = []

        for run_dir in self._iter_run_dirs():
            analysis_list = self._analysis.get_analysis(run_dir.name)
            for analysis in analysis_list:
                if analysis.get("report_type") == "stability":
                    if model and analysis.get("model") != model:
                        continue
                    reports.append(analysis)

        return reports

    def get_cluster_reports(self, model: Optional[str] = None) -> list[dict[str, Any]]:
        """查询聚类报告。"""
        cluster_path = self._root / "cluster_reports.jsonl"
        all_records = self._read_jsonl(cluster_path)

        reports: list[dict[str, Any]] = []
        for record in all_records:
            if model and record.get("model") != model:
                continue
            reports.append(record)

        return reports

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
        records: list[dict[str, Any]] = []

        run_dirs = (
            [d for d in self._iter_run_dirs() if d.name == run_id]
            if run_id
            else self._iter_run_dirs()
        )

        for run_dir in run_dirs:
            bid = run_dir.name
            try:
                status = self._status.get(bid)

                if model and status.get("model") != model:
                    continue

                timing_records = self._timing.get_timing_by_run(bid)

                for record in timing_records:
                    if phase_name and record.get("phase_name") != phase_name:
                        continue
                    if result_id and record.get("result_id") != result_id:
                        continue

                    if start_date or end_date:
                        created_at = record.get("created_at", "")
                        if created_at:
                            try:
                                record_time = datetime.fromisoformat(created_at)
                                if start_date and record_time < start_date:
                                    continue
                                if end_date and record_time > end_date:
                                    continue
                            except ValueError:
                                pass

                    records.append(record)

                    if len(records) >= limit:
                        break

            except FileNotFoundError:
                continue

            if len(records) >= limit:
                break

        return pd.DataFrame(records)

    def get_timing_summaries(
        self,
        model: Optional[str] = None,
        run_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """查询耗时阶段汇总统计。"""
        df = self.get_timing_phases(
            model=model,
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            limit=100000,  # 大量数据用于汇总
        )

        if df.empty:
            return pd.DataFrame()

        # 按 run_id 和 phase_name 汇总
        summary = (
            df.groupby(["run_id", "phase_name"])
            .agg(
                {
                    "duration": ["mean", "min", "max", "sum", "count"],
                    "wait_time": ["mean", "min", "max"],
                    "active_time": ["mean", "min", "max"],
                }
            )
            .reset_index()
        )

        return summary

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
        """创建待评分任务。"""
        # 在文件系统实现中，创建 scoring.jsonl 记录并标记为 pending
        scoring_data = {
            "task_id": task_id,
            "functional_score": 0.0,
            "quality_score": 0.0,
            "final_score": 0.0,
            "passed": False,
            "details": {
                "prompt": prompt,
                "expected_output": expected_output,
                "model_output": model_output,
                "model_answer": model_answer,
                "reasoning_content": reasoning_content,
                "test_cases": test_cases or [],
                "metadata": metadata or {},
                "scoring_dimensions": scoring_dimensions or [],
            },
            "scoring_status": "pending",
        }

        self._scoring.save_scoring(run_id, task_id, scoring_data)

        task_hash = self._generate_scoring_task_id(run_id, task_id)
        self._task_location_cache[task_hash] = (run_id, task_id)
        return task_hash

    def fetch_pending_scoring_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        """拉取待处理任务并原子锁定。"""
        tasks: list[dict[str, Any]] = []

        for run_dir in self._iter_run_dirs():
            benchmark_id = run_dir.name
            pending = self._scoring.find_pending(benchmark_id)

            for task in pending:
                if len(tasks) >= limit:
                    break

                task_key = f"{benchmark_id}:{task['task_id']}"
                task["task_id_hash"] = self._generate_scoring_task_id(
                    benchmark_id, task["task_id"]
                )
                self._task_location_cache[task["task_id_hash"]] = (
                    benchmark_id,
                    task["question_id"],
                )
                tasks.append(task)

            if len(tasks) >= limit:
                break

        return tasks

    def complete_scoring_task(self, task_id: int, score_result: dict[str, Any]) -> None:
        """标记任务完成，保存评分结果。"""
        cached = self._task_location_cache.get(task_id)
        if cached:
            benchmark_id, question_id = cached
            scoring_data = dict(score_result)
            scoring_data["task_id"] = question_id
            self.save_question_scoring(
                benchmark_id=benchmark_id,
                question_id=question_id,
                scoring_data=scoring_data,
                quality_signals=score_result.get("quality_signals", {}),
            )
            self._task_location_cache.pop(task_id, None)
            return

        for run_dir in self._iter_run_dirs():
            benchmark_id = run_dir.name
            pending = self._scoring.find_pending(benchmark_id)

            for task in pending:
                expected_hash = self._generate_scoring_task_id(
                    benchmark_id, task["task_id"]
                )
                if expected_hash == task_id:
                    scoring_data = dict(score_result)
                    scoring_data["task_id"] = task["task_id"]
                    self.save_question_scoring(
                        benchmark_id=benchmark_id,
                        question_id=task["question_id"],
                        scoring_data=scoring_data,
                        quality_signals=score_result.get("quality_signals", {}),
                    )
                    self._task_location_cache[expected_hash] = (
                        benchmark_id,
                        task["question_id"],
                    )
                    return

    def fail_scoring_task(self, task_id: int, error_message: str) -> None:
        """标记任务失败。"""
        cached = self._task_location_cache.get(task_id)
        if cached:
            benchmark_id, question_id = cached
            scoring_data = {
                "task_id": question_id,
                "functional_score": 0.0,
                "quality_score": 0.0,
                "final_score": 0.0,
                "passed": False,
                "details": {"error": error_message},
                "scoring_status": "failed",
            }
            self._scoring.save_scoring(benchmark_id, question_id, scoring_data)
            self._task_location_cache.pop(task_id, None)
            return

        for run_dir in self._iter_run_dirs():
            benchmark_id = run_dir.name
            pending = self._scoring.find_pending(benchmark_id)

            for task in pending:
                expected_hash = self._generate_scoring_task_id(
                    benchmark_id, task["task_id"]
                )
                if expected_hash == task_id:
                    scoring_data = {
                        "task_id": task["task_id"],
                        "functional_score": 0.0,
                        "quality_score": 0.0,
                        "final_score": 0.0,
                        "passed": False,
                        "details": {"error": error_message},
                        "scoring_status": "failed",
                    }
                    self._scoring.save_scoring(
                        benchmark_id, task["question_id"], scoring_data
                    )
                    return

    def retry_scoring_task(self, task_id: int) -> None:
        """将任务重置为 pending 以便重试。"""
        cached = self._task_location_cache.get(task_id)
        if cached:
            benchmark_id, question_id = cached
            self._scoring.mark_pending(benchmark_id, question_id)
            return

        for run_dir in self._iter_run_dirs():
            benchmark_id = run_dir.name

            for question_dir in run_dir.iterdir():
                if not question_dir.is_dir():
                    continue

                scoring_path = question_dir / "scoring.jsonl"
                if not scoring_path.exists():
                    continue

                for record in self._read_jsonl(scoring_path):
                    task_id_from_record = record.get("task_id", "")
                    expected_hash = self._generate_scoring_task_id(
                        benchmark_id, task_id_from_record
                    )
                    if expected_hash == task_id:
                        self._scoring.mark_pending(benchmark_id, question_dir.name)
                        self._task_location_cache[expected_hash] = (
                            benchmark_id,
                            question_dir.name,
                        )
                        return

    def get_pending_task_count(self) -> int:
        """获取待处理任务数量。"""
        count = 0

        for run_dir in self._iter_run_dirs():
            pending = self._scoring.find_pending(run_dir.name)
            count += len(pending)

        return count

    # ── 辅助方法 ──

    def _iter_run_dirs(self) -> list[Path]:
        """扫描 data/{execution_id}/{run_id}/ 的所有 run 目录（跨执行）。"""
        run_dirs: list[Path] = []
        for exec_entry in sorted(self._root.iterdir()):
            if not exec_entry.is_dir():
                continue
            for run_entry in sorted(exec_entry.iterdir()):
                if not run_entry.is_dir():
                    continue
                if (run_entry / "status.json").exists():
                    run_dirs.append(run_entry)
        return run_dirs

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_answers_from_dir(run_dir: Path) -> list[dict[str, Any]]:
        if not run_dir.is_dir():
            return []
        answers: list[dict[str, Any]] = []
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            answer_path = task_dir / "answer.jsonl"
            if not answer_path.exists():
                continue
            for line in answer_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped:
                    answers.append(json.loads(stripped))
        return answers

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """流式读取 JSONL 文件，返回解析后的记录列表。"""
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    @staticmethod
    def _generate_execution_id() -> str:
        timestamp = now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"bench_{timestamp}_{unique}"

    @staticmethod
    def _generate_benchmark_id(model: str, dimension: str) -> str:
        """生成 benchmark_id。"""
        timestamp = now().strftime("%Y%m%d_%H%M%S")
        model_slug = model.replace("/", "_").replace(".", "_")
        unique = uuid.uuid4().hex[:8]
        return f"{model_slug}_{dimension}_{timestamp}_{unique}"

    @staticmethod
    def _generate_id() -> str:
        """生成唯一 ID。"""
        return uuid.uuid4().hex[:16]

    @staticmethod
    def _generate_scoring_task_id(run_id: str, task_id: str) -> int:
        """基于 run_id + task_id 生成确定性 task hash（跨进程稳定）。"""
        digest = hashlib.sha256(f"{run_id}:{task_id}".encode()).hexdigest()
        return int(digest[:8], 16)

    # ── 日志方法 ──

    def append_execution_log(
        self,
        benchmark_id: str,
        question_id: str,
        message: str,
        level: str = "INFO",
    ) -> None:
        """追加执行日志。"""
        self._execution_log.append_log(benchmark_id, question_id, message, level)

    def read_execution_log(self, benchmark_id: str, question_id: str) -> str:
        """读取执行日志。"""
        return self._execution_log.read_log(benchmark_id, question_id)

    async def asave_result(self, result: EvalResult) -> str:
        return await asyncio.to_thread(self.save_answer, result)

    async def asave_metrics(
        self,
        metrics: ApiCallMetrics,
        run_id: str = "",
        task_id: str = "",
    ) -> str:
        return await asyncio.to_thread(self.save_metrics, metrics, run_id, task_id)

    async def aget_quality_signals_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.get_quality_signals_for_run, run_id)

    async def aget_quality_signals_history(
        self, model: str, days: int = 7
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.get_quality_signals_history, model, days)

    async def asave_quality_signals(self, signals: dict[str, Any]) -> str:
        return await asyncio.to_thread(self.save_quality_signals, signals)

    async def asave_stability_report(self, report: StabilityReport) -> str:
        return await asyncio.to_thread(self.save_analysis, report)

    async def asave_cluster_report(self, report: ClusterReport) -> str:
        return await asyncio.to_thread(self.save_cluster_report, report)

    async def aget_results(
        self,
        model: Optional[str] = None,
        dimension: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.get_results, model=model, dimension=dimension, run_id=run_id
        )
