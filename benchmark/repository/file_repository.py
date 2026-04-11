"""FileRepository - 基于文件系统的 Repository 实现。

组合所有 handler 提供统一的存储接口。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from benchmark.analysis.models import ClusterReport
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

    def __init__(self, data_root: Path | str = "data") -> None:
        self._root = Path(data_root)
        self._root.mkdir(parents=True, exist_ok=True)

        # 初始化所有 handlers
        self._status = StatusHandler(self._root)
        self._metadata = MetadataHandler(self._root)
        self._answer = AnswerHandler(self._root)
        self._scoring = ScoringHandler(self._root)
        self._timing = TimingHandler(self._root)
        self._execution_log = ExecutionLogHandler(self._root)
        self._analysis = AnalysisHandler(self._root)
        self._cluster = ClusterHandler(self._root)

        # 初始化 IndexBuilder
        self._index_builder = IndexBuilder(self._root)

    # ── Run 生命周期 ──

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
        # 构造 EvalResult
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
            created_at=datetime.now(timezone.utc),
        )

        # 构造 ApiCallMetrics（如果提供）
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
                created_at=datetime.now(timezone.utc),
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
        """获取所有未完成的运行记录。

        Returns:
            未完成的 EvalRun 列表
        """
        active_runs: list[EvalRun] = []

        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            status_path = entry / "status.json"
            if not status_path.exists():
                continue

            try:
                status = self._status.get(entry.name)
                if status.get("status") not in ("completed", "failed"):
                    # 读取 metadata 获取 dataset
                    metadata = self._metadata.read(entry.name)
                    dataset = metadata.get("dataset", "")

                    active_runs.append(
                        EvalRun(
                            run_id=entry.name,
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

        # 返回索引摘要
        index_path = self._root / "index.jsonl"
        row_count = 0
        if index_path.exists():
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row_count += 1

        return {
            "index_file": str(index_path),
            "row_count": row_count,
            "data_root": str(self._root),
        }

    # ── Repository 接口实现 ──

    def create_run(self, run: EvalRun) -> str:
        """创建评测运行记录（Repository 接口实现）。"""
        # 构造 questions 列表（从 task_id 推断，这里假设有一个 task）
        questions = ["task_1"]  # 默认至少有一个任务

        return self.create_benchmark_run(
            model=run.model,
            dimension=run.dimension,
            dataset=run.dataset,
            questions=questions,
        )

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """标记运行记录为已完成或失败。"""
        if status == "failed":
            self._status.set_failed(run_id)
        # completed 状态由 save_question_scoring 在 scored == total 时自动设置

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

    def save_metrics(self, metrics: ApiCallMetrics) -> str:
        """保存 API 调用指标。

        注意：metrics 通常在 save_answer 时一并保存，
        此方法用于单独保存 metrics 的场景。
        """
        # 构造一个空的 result 来保存 metrics
        result = EvalResult(
            result_id=metrics.result_id,
            run_id="",  # 需要通过其他方式获取
            task_id="",
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

        # 确定要查询的 benchmark_ids
        benchmark_ids: list[str] = []
        if run_id:
            benchmark_ids = [run_id]
        else:
            # 从 index 或扫描目录获取
            for entry in sorted(self._root.iterdir()):
                if entry.is_dir() and (entry / "status.json").exists():
                    benchmark_ids.append(entry.name)

        for bid in benchmark_ids:
            try:
                status = self._status.get(bid)

                # 过滤条件
                if model and status.get("model") != model:
                    continue
                if dimension and status.get("dimension") != dimension:
                    continue

                # 获取所有答案
                answers = self._answer.get_answers_by_run(bid)

                for ans in answers:
                    answer_data = ans.get("answer", {})
                    result = {
                        "result_id": answer_data.get("result_id"),
                        "run_id": bid,
                        "task_id": answer_data.get("task_id"),
                        "model": status.get("model"),
                        "dimension": status.get("dimension"),
                        "model_output": answer_data.get("model_output"),
                        "model_answer": answer_data.get("model_answer"),
                        "final_score": answer_data.get("final_score"),
                        "passed": answer_data.get("passed"),
                        "execution_time": answer_data.get("execution_time"),
                    }
                    results.append(result)

            except FileNotFoundError:
                continue

        return results

    def get_result_detail(self, result_id: str) -> Optional[dict[str, Any]]:
        """获取单条结果的完整详情。"""
        # 需要遍历所有 runs 找到对应的 result
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            benchmark_id = entry.name
            answers = self._answer.get_answers_by_run(benchmark_id)

            for ans in answers:
                answer_data = ans.get("answer", {})
                if answer_data.get("result_id") == result_id:
                    # 找到匹配的 result，构造详情
                    try:
                        status = self._status.get(benchmark_id)
                        model = status.get("model", "")
                        dimension = status.get("dimension", "")
                    except FileNotFoundError:
                        model = ""
                        dimension = ""

                    return {
                        "result_id": result_id,
                        "run_id": benchmark_id,
                        "task_id": answer_data.get("task_id"),
                        "model": model,
                        "dimension": dimension,
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

        rows: list[dict[str, Any]] = []
        index_path = self._root / "index.jsonl"

        if not index_path.exists():
            return rows

        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                # 应用过滤条件
                if model and row.get("model") != model:
                    continue
                if dimension and row.get("dimension") != dimension:
                    continue
                if status_filter and row.get("status") != status_filter:
                    continue

                rows.append(row)
            except Exception:
                continue

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
        cutoff = datetime.now(timezone.utc).timestamp() - days * 24 * 3600

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
        if hasattr(report, "model") and hasattr(report, "run_id"):
            # 假设是 StabilityReport 类型
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
                "created_at": getattr(
                    report, "created_at", datetime.now(timezone.utc)
                ).isoformat(),
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
            return f"cluster_{report.model}_{int(datetime.now().timestamp())}"
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

        bench_dir = self._root / run_id
        if not bench_dir.exists():
            return signals

        for question_dir in bench_dir.iterdir():
            if not question_dir.is_dir():
                continue

            scoring_path = question_dir / "scoring.jsonl"
            if not scoring_path.exists():
                continue

            for line in scoring_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue

                try:
                    record = __import__("json").loads(line)
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
                except Exception:
                    continue

        return signals

    def get_quality_signals_history(
        self, model: str, days: int = 7
    ) -> list[dict[str, Any]]:
        """获取某模型最近 N 天的质量信号。"""
        # 获取该模型的所有 runs
        runs = self.get_runs(model=model)

        # 过滤最近 N 天
        cutoff = datetime.now(timezone.utc).timestamp() - days * 24 * 3600

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

        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            analysis_list = self._analysis.get_analysis(entry.name)
            for analysis in analysis_list:
                if analysis.get("report_type") == "stability":
                    if model and analysis.get("model") != model:
                        continue
                    reports.append(analysis)

        return reports

    def get_cluster_reports(self, model: Optional[str] = None) -> list[dict[str, Any]]:
        """查询聚类报告。"""
        cluster_path = self._root / "cluster_reports.jsonl"
        if not cluster_path.exists():
            return []

        reports: list[dict[str, Any]] = []
        for line in cluster_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                record = __import__("json").loads(line)
                if model and record.get("model") != model:
                    continue
                reports.append(record)
            except Exception:
                continue

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

        # 确定要查询的 runs
        benchmark_ids: list[str] = []
        if run_id:
            benchmark_ids = [run_id]
        else:
            for entry in sorted(self._root.iterdir()):
                if entry.is_dir() and (entry / "status.json").exists():
                    benchmark_ids.append(entry.name)

        for bid in benchmark_ids:
            try:
                status = self._status.get(bid)

                # 过滤 model
                if model and status.get("model") != model:
                    continue

                # 获取该 run 的所有 timing 数据
                timing_records = self._timing.get_timing_by_run(bid)

                for record in timing_records:
                    # 过滤条件
                    if phase_name and record.get("phase_name") != phase_name:
                        continue
                    if result_id and record.get("result_id") != result_id:
                        continue

                    # 日期过滤
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

        # 返回一个基于 run_id 和 task_id 的哈希 ID（用于后续查找）
        return hash(f"{run_id}:{task_id}") % 2147483647

    def fetch_pending_scoring_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        """拉取待处理任务并原子锁定。"""
        tasks: list[dict[str, Any]] = []

        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            benchmark_id = entry.name
            pending = self._scoring.find_pending(benchmark_id)

            for task in pending:
                if len(tasks) >= limit:
                    break

                task["task_id_hash"] = hash(task["task_id"]) % 2147483647
                tasks.append(task)

            if len(tasks) >= limit:
                break

        return tasks

    def complete_scoring_task(self, task_id: int, score_result: dict[str, Any]) -> None:
        """标记任务完成，保存评分结果。"""
        # 找到对应的任务并更新
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            benchmark_id = entry.name
            pending = self._scoring.find_pending(benchmark_id)

            for task in pending:
                # 使用与 create_scoring_task 相同的哈希计算方式
                expected_hash = hash(f"{benchmark_id}:{task['task_id']}") % 2147483647
                if expected_hash == task_id:
                    # 更新评分数据（确保包含 task_id）
                    scoring_data = dict(score_result)
                    scoring_data["task_id"] = task["task_id"]
                    self.save_question_scoring(
                        benchmark_id=benchmark_id,
                        question_id=task["question_id"],
                        scoring_data=scoring_data,
                        quality_signals=score_result.get("quality_signals", {}),
                    )
                    return

    def fail_scoring_task(self, task_id: int, error_message: str) -> None:
        """标记任务失败。"""
        # 类似 complete_scoring_task，但标记为 failed
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            benchmark_id = entry.name
            pending = self._scoring.find_pending(benchmark_id)

            for task in pending:
                expected_hash = hash(f"{benchmark_id}:{task['task_id']}") % 2147483647
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
        # 遍历所有 scoring 文件找对应的 task
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            benchmark_id = entry.name

            for question_dir in entry.iterdir():
                if not question_dir.is_dir():
                    continue

                scoring_path = question_dir / "scoring.jsonl"
                if not scoring_path.exists():
                    continue

                # 检查是否是目标 task
                for line in scoring_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = __import__("json").loads(line)
                        task_id_from_record = record.get("task_id", "")
                        expected_hash = (
                            hash(f"{benchmark_id}:{task_id_from_record}") % 2147483647
                        )
                        if expected_hash == task_id:
                            # 标记为 pending
                            self._scoring.mark_pending(benchmark_id, question_dir.name)
                            return
                    except Exception:
                        continue

    def get_pending_task_count(self) -> int:
        """获取待处理任务数量。"""
        count = 0

        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue

            pending = self._scoring.find_pending(entry.name)
            count += len(pending)

        return count

    # ── 辅助方法 ──

    @staticmethod
    def _generate_benchmark_id(model: str, dimension: str) -> str:
        """生成 benchmark_id。"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_slug = model.replace("/", "_").replace(".", "_")
        unique = uuid.uuid4().hex[:8]
        return f"{model_slug}_{dimension}_{timestamp}_{unique}"

    @staticmethod
    def _generate_id() -> str:
        """生成唯一 ID。"""
        return uuid.uuid4().hex[:16]

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
