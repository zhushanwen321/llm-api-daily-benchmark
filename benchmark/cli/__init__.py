"""CLI 命令入口。支持 evaluate / list-datasets / export 命令."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console

from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
from benchmark.adapters.frontcode_adapter import FrontCodeAdapter
from benchmark.adapters.probe_adapter import ProbeAdapter
from benchmark.adapters.math_adapter import MATHAdapter
from benchmark.core.evaluator import SingleTurnEvaluator
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.logging_config import setup_logging

from benchmark.core.timing_tracker import (
    TimingTracker,
    get_timing_collector,
    start_timing_collection,
    stop_timing_collection,
)
from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun
from benchmark.scorers.backend import create_backend_composite
from benchmark.scorers.composite import CompositeScorer
from benchmark.scorers.frontend import create_frontend_composite
from benchmark.scorers.probe_scorer import ProbeScorer
from benchmark.scorers.reasoning import create_reasoning_composite

console = Console(force_terminal=True)
logger = logging.getLogger(__name__)

DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (MATHAdapter, create_reasoning_composite, SingleTurnEvaluator),
    "backend-dev": (BigCodeBenchAdapter, create_backend_composite, SingleTurnEvaluator),
    "frontend-dev": (FrontCodeAdapter, create_frontend_composite, SingleTurnEvaluator),
    "probe": (ProbeAdapter, lambda: [(1.0, ProbeScorer())], SingleTurnEvaluator),
}

DATASET_REGISTRY: dict[str, str] = {
    "reasoning": "math",
    "backend-dev": "bigcodebench",
    "frontend-dev": "frontcode",
    "probe": "probe",
}

_THINKING_SYSTEM_MESSAGE = (
    "你是一个高效助手。根据任务难度自适应调节思考深度：\n"
    "- 简单任务（如选择题、事实查询）：直接回答，简短推理即可\n"
    "- 中等任务（如数学计算、代码编写）：适当推理，重点关注核心逻辑\n"
    "- 复杂任务（如系统设计、多步证明）：审慎推理，但避免重复验证已知结论\n"
    "如果已经找到答案，立即停止推理并给出最终结果。"
)


async def _score_with_timing(scorer: Any, ctx: Any, timing: TimingTracker) -> Any:
    """带耗时追踪的评分调用。"""
    score_result = await scorer.ascore(ctx)
    return score_result


def _get_scoring_dimensions(task: Any) -> list[str]:
    """根据任务维度获取需要计算的评分维度列表。"""
    # 基础维度：所有任务都需要答案正确性评分
    dimensions = ["answer_correctness"]

    # reasoning 维度的额外评分
    if task.dimension == "reasoning":
        weighted_scoring = os.getenv("WEIGHTED_SCORING", "false").lower() == "true"
        if weighted_scoring:
            dimensions.extend(
                [
                    "reasoning_completeness",
                    "method_elegance",
                    "difficulty_adaptation",
                ]
            )

    return dimensions


def _setup_proxy() -> None:
    """从 .env 加载代理配置，用于 HuggingFace 数据集下载."""
    load_dotenv()
    from pathlib import Path

    dataset_flag = Path("benchmark/datasets/.download-complete")
    if dataset_flag.exists():
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        logger.info("检测到 .download-complete 标志，启用离线模式")
    else:
        logger.debug("未检测到 .download-complete 标志，使用网络下载数据集")

    proxy = os.getenv("HF_PROXY")
    if proxy:
        os.environ.setdefault("http_proxy", proxy)
        os.environ.setdefault("https_proxy", proxy)
        os.environ.setdefault("all_proxy", proxy)


@click.group()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="启用 debug 模式，输出详细日志到控制台和 logs/ 目录",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """LLM Benchmark 评测工具."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    setup_logging(debug=debug)


@cli.command()
@click.option(
    "--model", required=True, help="模型标识，格式: provider/model（如 glm/glm-4.7）"
)
@click.option(
    "--dimension",
    required=True,
    type=click.Choice(
        [
            "reasoning",
            "backend-dev",
            "frontend-dev",
            "probe",
            "all",
        ]
    ),
    help="评测维度",
)
@click.option("--samples", default=15, help="评测题目数量")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="启用 debug 模式（也可放在 'benchmark' 命令后）",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    model: str,
    dimension: str,
    samples: int,
    debug: bool,
) -> None:
    # 如果子命令没有指定 debug，使用父命令的设置
    if not debug:
        debug = ctx.obj.get("debug", False)
    _setup_proxy()

    models = [m.strip() for m in model.split(",") if m.strip()]

    if dimension == "all":
        dimensions = list(DIMENSION_REGISTRY.keys())
    else:
        dimensions = [dimension]

    db_path = "benchmark/data/results.db"
    timing_data_dir = "benchmark/data"

    async def _run_evaluation_with_timing():
        import sys

        print("[BENCHMARK] Starting evaluation...", flush=True, file=sys.stderr)
        await start_timing_collection(timing_data_dir)
        print("[BENCHMARK] Running multi-evaluation...", flush=True, file=sys.stderr)
        try:
            await _run_multi_evaluation(models, dimensions, samples, debug)
            print("[BENCHMARK] Evaluation completed", flush=True, file=sys.stderr)
        finally:
            await stop_timing_collection()

    asyncio.run(_run_evaluation_with_timing())


async def _evaluate_task(
    task_idx: int,
    task: Any,
    model: str,
    llm: LLMEvalAdapter,
    scorer: Any,
    evaluator: Any,
    db: Database,
    run_id: str,
    total: int,
    debug: bool,
    system_message: str | None = None,
    dimension: str = "",
) -> dict[str, Any]:
    """单个 task 的异步评测协程，支持耗时追踪。"""
    timing = TimingTracker()
    result_id = str(uuid.uuid4())[:12]

    async def _do_evaluate() -> dict[str, Any]:
        """实际评测逻辑（不含 semaphore 等待）。"""
        nonlocal result_id
        try:
            logger.info(
                f"[TASK] 开始 | task_id={task.task_id} | model={model} | dimension={dimension}"
            )

            # Phase 1: LLM request
            logger.info(f"[TASK] LLM请求开始 | task_id={task.task_id}")
            timing.start_phase("llm_request")
            ctx = await evaluator.evaluate(
                task, model, llm, system_message=system_message
            )
            timing.end_phase("llm_request")
            gm = ctx.gen_metrics or {}
            api_duration = gm.get("duration", 0.0)
            logger.info(
                f"[TASK] LLM请求完成 | task_id={task.task_id} | "
                f"duration={api_duration:.2f}s | ttft={gm.get('ttft', 0.0):.2f}s | "
                f"tokens={gm.get('prompt_tokens', 0)}/{gm.get('completion_tokens', 0)} | "
                f"speed={gm.get('tokens_per_second', 0.0):.1f}tok/s"
            )

            # Phase 2: Score calculation
            logger.info(f"[TASK] 评分开始 | task_id={task.task_id}")
            timing.start_phase("score_calculation")

            # 进程内协程评分（Worker 已移除）
            score_result = await _score_with_timing(scorer, ctx, timing)
            functional_score = score_result.score
            quality_score = 0.0
            final_score = score_result.score
            passed = score_result.passed
            score_details = score_result.details
            score_reasoning = getattr(score_result, "reasoning", "")

            timing.end_phase("score_calculation")
            logger.info(
                f"[TASK] 评分完成 | task_id={task.task_id} | "
                f"score={final_score:.1f} | passed={passed}"
            )

            # Phase 3: DB write
            logger.info(
                f"[TASK] 数据保存开始 | task_id={task.task_id} | result_id={result_id}"
            )
            timing.start_phase("db_write")
            result = EvalResult(
                result_id=result_id,
                run_id=run_id,
                task_id=task.task_id,
                task_content=task.prompt,
                model_output=ctx.raw_output,
                model_think=ctx.reasoning_content,
                model_answer=ctx.model_answer,
                expected_output=task.expected_output,
                functional_score=functional_score,
                final_score=final_score,
                passed=passed,
                details=score_details,
                execution_time=api_duration,
                created_at=datetime.now(),
            )
            await db.asave_result(result)

            tps = gm.get("tokens_per_second", 0.0)
            await db.asave_metrics(
                ApiCallMetrics(
                    result_id=result_id,
                    prompt_tokens=gm.get("prompt_tokens", 0),
                    completion_tokens=gm.get("completion_tokens", 0),
                    reasoning_tokens=gm.get("reasoning_tokens", 0),
                    reasoning_content=ctx.reasoning_content,
                    duration=api_duration,
                    tokens_per_second=tps,
                    ttft=gm.get("ttft", 0.0),
                    ttft_content=gm.get("ttft_content", 0.0),
                    created_at=datetime.now(),
                )
            )
            timing.end_phase("db_write")
            logger.info(
                f"[TASK] 数据保存完成 | task_id={task.task_id} | result_id={result_id}"
            )

            # Phase 4: Quality signals
            logger.info(f"[TASK] 质量信号采集开始 | task_id={task.task_id}")
            timing.start_phase("quality_signals")
            try:
                from benchmark.analysis.quality_signals import QualitySignalCollector

                qsc = QualitySignalCollector(db=db, model=model)
                await qsc.collect_and_save(
                    result_id=result_id,
                    raw_output=ctx.raw_output,
                    reasoning_content=ctx.reasoning_content,
                    gen_metrics=gm,
                    finish_reason=gm.get("finish_reason", ""),
                    task=task,
                    dimension=dimension,
                )
                logger.info(f"[TASK] 质量信号采集完成 | task_id={task.task_id}")
            except Exception as exc:
                logger.warning(
                    f"[TASK] 质量信号采集失败 | task_id={task.task_id}: {exc}"
                )
            timing.end_phase("quality_signals")

            effective_total = timing.get_total_duration()
            if effective_total > 10.0:
                logger.info(
                    f"GANTT | {model} | {task.task_id} | "
                    f"llm={api_duration:.1f}s | "
                    f"total={effective_total:.1f}s"
                )

            status_icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(
                f"  [{task_idx + 1}/{total}] {task.task_id} | "
                f"Score: {final_score:.0f} | {status_icon} | "
                f"Time: {effective_total:.1f}s | "
                f"TTFT-R: {gm.get('ttft', 0.0):.2f}s | "
                f"TTFT-C: {gm.get('ttft_content', 0.0):.2f}s | "
                f"Speed: {tps:.1f} tok/s"
            )

            logger.info(
                f"[TASK] 完成 | task_id={task.task_id} | "
                f"score={final_score:.1f} | passed={passed} | "
                f"total_time={effective_total:.2f}s"
            )

            return {
                "score": final_score,
                "passed": passed,
                "task_id": task.task_id,
            }
        except Exception as exc:
            logger.error(
                f"[TASK] 失败 | task_id={getattr(task, 'task_id', task_idx)}: {type(exc).__name__}: {exc}"
            )
            status_msg = f"[red]ERROR: {type(exc).__name__}: {exc}[/red]"
            console.print(
                f"  [{task_idx + 1}/{total}] {getattr(task, 'task_id', '?')} | {status_msg}"
            )
            return {
                "error": exc,
                "task_id": getattr(task, "task_id", str(task_idx)),
                "passed": False,
                "score": 0.0,
            }

    result = await _do_evaluate()

    # 异步收集耗时数据
    try:
        timing_collector = get_timing_collector()
        timing_collector.collect(timing, result_id, model, task.task_id, run_id=run_id)
    except RuntimeError:
        logger.debug("Timing collector not initialized, skipping timing collection")

    # 打印甘特图日志
    gantt_data = timing.to_gantt_data()
    if gantt_data:
        logger.info(
            f"TIMING_GANTT | {model} | task={task.task_id} | "
            f"total={timing.get_total_duration():.3f}s | "
            f"wait={timing.get_wait_duration():.3f}s | "
            f"active={timing.get_active_duration():.3f}s | "
            f"phases={','.join(p['phase'] for p in gantt_data)}"
        )

    return result


async def _run_evaluation(
    model: str,
    dimension: str,
    samples: int,
    debug: bool,
    db: Database | None = None,
) -> None:
    """异步评测主流程，使用 asyncio.gather 并发执行所有 task。"""
    adapter_cls, scorer_factory, evaluator_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    evaluator = evaluator_cls()
    llm = LLMEvalAdapter(model=model)
    scorer = CompositeScorer(scorer_factory())

    logger.debug(
        f"加载适配器: {adapter_cls.__name__}, 评分器: {type(scorer).__name__}, 编排器: {evaluator_cls.__name__}"
    )

    tasks = adapter.load()[:samples]
    if not tasks:
        console.print("[red]No tasks loaded.[/red]")
        raise SystemExit(1)

    logger.debug(f"已加载 {len(tasks)} 个任务")

    console.print(
        f"[bold green]Starting evaluation:[/bold green] "
        f"{dimension} with {len(tasks)} tasks, model={model}"
    )

    run_id = str(uuid.uuid4())[:12]
    run = EvalRun(
        run_id=run_id,
        model=model,
        dimension=dimension,
        dataset=DATASET_REGISTRY[dimension],
        started_at=datetime.now(),
        status="running",
    )
    own_db = db is None
    if own_db:
        db = Database()
    try:
        await asyncio.to_thread(db.create_run, run)

        coros = [
            _evaluate_task(
                i,
                task,
                model,
                llm,
                scorer,
                evaluator,
                db,
                run_id,
                len(tasks),
                debug,
                system_message=_THINKING_SYSTEM_MESSAGE,
                dimension=dimension,
            )
            for i, task in enumerate(tasks)
        ]

        total_score = 0.0
        passed_count = 0
        failed_count = 0
        completed_count = 0
        logger.info(
            f"[EVAL] 开始评测 | run_id={run_id} | model={model} | dimension={dimension} | tasks={len(tasks)}"
        )
        for coro in asyncio.as_completed(coros):
            r = await coro
            completed_count += 1
            if isinstance(r, Exception):
                failed_count += 1
                logger.error(
                    f"[EVAL] 任务失败 ({completed_count}/{len(tasks)}): {type(r).__name__}: {r}"
                )
                continue
            if r.get("error"):
                failed_count += 1
                logger.error(
                    f"[EVAL] 任务失败 ({completed_count}/{len(tasks)}): {r.get('task_id', '?')}"
                )
            else:
                status = "PASS" if r["passed"] else "FAIL"
                logger.info(
                    f"[EVAL] 任务完成 ({completed_count}/{len(tasks)}) | task_id={r.get('task_id', '?')} | score={r['score']:.1f} | status={status}"
                )
            total_score += r["score"]
            if r["passed"]:
                passed_count += 1
        logger.info(
            f"[EVAL] 评测完成 | run_id={run_id} | completed={completed_count}/{len(tasks)} | passed={passed_count} | failed={failed_count}"
        )

        try:
            from benchmark.analysis.stability_analyzer import StabilityAnalyzer

            analyzer = StabilityAnalyzer(db=db)
            report = await analyzer.run(model=model, run_id=run_id, dimension=dimension)
            status_color = {
                "stable": "green",
                "suspicious": "yellow",
                "degraded": "red",
            }.get(report.overall_status, "white")
            console.print(
                f"  Stability: [{status_color}]{report.overall_status}[/{status_color}] "
                f"{report.summary}"
            )
        except Exception as exc:
            logger.warning(f"稳定性分析失败: {exc}")

        if dimension == "probe":
            try:
                from benchmark.analysis.fingerprint import FingerprintManager

                fm = FingerprintManager()
                results = await asyncio.to_thread(
                    db.get_results,
                    model=model,
                    dimension="probe",
                    run_id=run_id,
                )
                signals = await db.aget_quality_signals_for_run(run_id)
                scores = [float(r["final_score"]) for r in results]

                fp = fm.generate_fingerprint_sync(
                    model=model,
                    scores=scores,
                    quality_signals=signals,
                    run_id=run_id,
                )
                comparison = fm.compare_with_baseline(model=model)

                status_color = "green" if comparison["status"] == "match" else "red"
                console.print(
                    f"  Fingerprint: [{status_color}]{comparison['status']}"
                    f"[/{status_color}] "
                    f"similarity={comparison.get('similarity', 'N/A')}"
                )
            except Exception as exc:
                logger.warning(f"指纹分析失败: {exc}")

            try:
                from benchmark.analysis.cluster_analyzer import (
                    FingerprintClusterAnalyzer,
                )

                cluster_analyzer = FingerprintClusterAnalyzer()
                cluster_report = cluster_analyzer.analyze(model)

                if cluster_report.n_clusters > 0:
                    c_color = "red" if cluster_report.n_clusters > 1 else "green"
                    console.print(
                        f"  Cluster: [{c_color}]{cluster_report.n_clusters} clusters"
                        f"[/{c_color}] {cluster_report.summary}"
                    )
                    if cluster_report.suspected_changes:
                        for change in cluster_report.suspected_changes:
                            console.print(
                                f"    Change at {change['at']}: "
                                f"cluster {change['from_cluster']} → {change['to_cluster']} "
                                f"(similarity={change['cosine_similarity']:.4f})"
                            )
                    await db.asave_cluster_report(cluster_report)
            except Exception as exc:
                logger.warning(f"聚类分析失败: {exc}")

        if failed_count == 0:
            await asyncio.to_thread(db.finish_run, run_id, "completed")
        else:
            await asyncio.to_thread(
                db.finish_run,
                run_id,
                "partial" if failed_count < len(tasks) else "failed",
            )

        avg_score = total_score / len(tasks) if tasks else 0
        summary = (
            f"\n[bold]Evaluation complete:[/bold] run_id={run_id}\n"
            f"  Average Score: [bold]{avg_score:.1f}[/bold]\n"
            f"  Passed: {passed_count}/{len(tasks)}"
        )
        if failed_count > 0:
            summary += f"\n  Failed: [red]{failed_count}/{len(tasks)}[/red]"
        console.print(summary)
    except Exception:
        console.print("[red]Evaluation failed![/red]")
        try:
            await asyncio.to_thread(db.finish_run, run_id, "failed")
        except Exception:
            pass
        raise
    finally:
        if own_db:
            db.close()
        await llm.close()


def _group_by_provider(
    models: list[str], dimensions: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """按 provider 分组 (model, dimension) 对。"""
    groups: dict[str, list[tuple[str, str]]] = {}
    for model in models:
        provider = model.split("/", 1)[0]
        for dim in dimensions:
            groups.setdefault(provider, []).append((model, dim))
    return groups


async def _run_provider_group(
    tasks: list[tuple[str, str]], samples: int, debug: bool, db: Database
) -> None:
    """同一 provider 内的 evaluation run 并发执行。"""
    if not tasks:
        return

    coros = [_run_evaluation(model, dim, samples, debug, db=db) for model, dim in tasks]
    results = await asyncio.gather(*coros, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            model, dim = tasks[i]
            logger.error(f"Task failed for {model}/{dim}: {result}")
            console.print(f"[red]Error: Evaluation failed for {model}/{dim}[/red]")


def _get_provider_concurrency(model: str) -> int:
    """获取 provider 的最大并发数。"""
    try:
        from benchmark.config import get_model_config

        cfg = get_model_config(model)
        return cfg.get("max_concurrency", cfg.get("rate_limit", 2))
    except Exception:
        return 2


async def _run_multi_evaluation(
    models: list[str], dimensions: list[str], samples: int, debug: bool
) -> None:
    """多模型 x 多维度评测。共享单一 Database 实例以避免 SQLite 写锁竞争。

    注意：维度之间串行执行，避免过多并发导致 SQLite 锁竞争。
    每个维度内部的任务仍然并发执行。
    """
    db = Database()
    try:
        groups = _group_by_provider(models, dimensions)
        # 串行执行每个 provider 组的维度，避免 SQLite 锁竞争
        for provider, tasks in groups.items():
            logger.info(f"[EVAL] 开始评测 provider: {provider} | 维度数: {len(tasks)}")
            await _run_provider_group(tasks, samples, debug, db=db)
            logger.info(f"[EVAL] 完成评测 provider: {provider}")
    finally:
        db.close()


@cli.command("list-datasets")
def list_datasets() -> None:
    """列出可用数据集."""
    console.print("[bold]Available datasets:[/bold]")
    console.print("  [cyan]reasoning:[/cyan]           MATH (Level 3-5, 15 tasks)")
    console.print("  [cyan]backend-dev:[/cyan]        BigCodeBench-Hard (15 tasks)")
    console.print("  [cyan]frontend-dev:[/cyan]       FrontCode (自建前端评测)")


@cli.command()
@click.option(
    "--output-dir",
    default="benchmark/datasets",
    help="缓存输出目录",
)
def download(output_dir: str) -> None:
    """预下载数据集并缓存为 JSON 格式，用于离线评测."""
    _setup_proxy()

    from datasets import load_dataset as hf_load_dataset

    datasets_spec = {
        "reasoning": [
            ("nlile/hendrycks-MATH-benchmark", "test", None, "math"),
        ],
        "backend-dev": [
            ("bigcode/bigcodebench-hard", "v0.1.0_hf", None, "bigcodebench"),
        ],
    }

    total = sum(len(v) for v in datasets_spec.values())
    done = 0
    for dimension, specs in datasets_spec.items():
        for repo, split, config_name, subdir in specs:
            done += 1
            console.print(
                f"[{done}/{total}] Downloading [cyan]{repo}[/cyan] "
                f"(split={split}, config={config_name or 'default'}) ..."
            )
            cache_dir = os.path.join(output_dir, subdir)

            safe_repo = repo.replace("/", "--")
            parts = [cache_dir, safe_repo]
            if config_name:
                parts.append(config_name.replace("/", "--"))
            parts.append(f"{split}.json")
            cache_file = os.path.join(*parts)

            if os.path.exists(cache_file):
                console.print(f"  [green]Cache exists, skipping.[/green]")
                continue

            # 用 datasets 库下载（支持 HF Hub，不依赖 datasets-server API）
            ds = hf_load_dataset(repo, name=config_name, split=split)
            rows = [dict(row) for row in ds]

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

            console.print(f"  [green]Cached {len(rows)} rows -> {cache_file}[/green]")

    flag_path = os.path.join(output_dir, ".download-complete")
    Path(flag_path).touch()
    console.print(
        f"\n[bold green]All datasets downloaded. Flag: {flag_path}[/bold green]"
    )


@cli.command()
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="导出格式",
)
@click.option("--output", default="results.json", help="输出文件路径")
@click.option("--model", default=None, help="按模型过滤")
@click.option("--dimension", default=None, help="按维度过滤")
def export(fmt: str, output: str, model: str | None, dimension: str | None) -> None:
    """导出评测结果."""
    db = Database()
    results = db.get_results(model=model, dimension=dimension)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    output_path = Path(output)

    if fmt == "json":
        data = [dict(row) for row in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[green]Exported {len(data)} results to {output_path}[/green]")

    elif fmt == "csv":
        if results:
            keys = results[0].keys()
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            console.print(
                f"[green]Exported {len(results)} results to {output_path}[/green]"
            )


@cli.command()
@click.option("--models", default=None, help="逗号分隔的模型列表")
@click.option("--dimensions", default=None, help="逗号分隔的维度列表")
@click.option(
    "--date-range", default=None, help="日期范围，格式: 2026-04-01,2026-04-30"
)
@click.option("--output", default="report.html", help="输出文件路径")
def report(
    models: str | None, dimensions: str | None, date_range: str | None, output: str
) -> None:
    """生成 HTML 评测报告."""
    from benchmark.core.reporter import generate_html_report

    model_list = models.split(",") if models else None
    dim_list = dimensions.split(",") if dimensions else None
    dr: tuple[str, str] | None = None
    if date_range:
        parts = date_range.split(",")
        if len(parts) == 2:
            dr = (parts[0], parts[1])
        else:
            console.print("[red]日期范围格式错误，应为: YYYY-MM-DD,YYYY-MM-DD[/red]")
            raise SystemExit(1)

    try:
        path = generate_html_report(
            models=model_list,
            dimensions=dim_list,
            date_range=dr,
            output_path=output,
        )
        console.print(f"[green]Report generated: {path}[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


@cli.group()
def scheduler() -> None:
    """定时调度器管理。"""


@scheduler.command()
def start() -> None:
    """启动定时调度器。"""
    from benchmark.core.scheduler import BenchmarkScheduler

    sched = BenchmarkScheduler()
    sched.start()
    if sched.enabled:
        console.print("[green]调度器已启动[/green]")
        console.print(f"  Cron: {sched.cron}")
        console.print(f"  Models: {sched.models}")
        console.print(f"  Dimensions: {sched.dimensions}")
        console.print(f"  Samples: {sched.samples}")
    else:
        console.print("[yellow]调度器未启用 (设置 SCHEDULER_ENABLED=true)[/yellow]")


@scheduler.command()
def stop() -> None:
    """停止定时调度器。"""
    from benchmark.core.scheduler import BenchmarkScheduler

    sched = BenchmarkScheduler()
    sched.stop()
    console.print("[green]调度器已停止[/green]")


@scheduler.command()
def status() -> None:
    """查看调度器状态。"""
    from benchmark.core.scheduler import BenchmarkScheduler

    sched = BenchmarkScheduler()
    console.print(f"  Enabled: {sched.enabled}")
    console.print(f"  Cron: {sched.cron}")
    console.print(f"  Models: {sched.models}")
    console.print(f"  Dimensions: {sched.dimensions}")
    console.print(f"  Samples: {sched.samples}")


@cli.command("analyze")
@click.option("--model", default=None, help="要分析的模型（不指定则分析全部）")
@click.option("--classify", is_flag=True, default=False, help="同时运行跨模型分类")
def analyze(model: str | None, classify: bool) -> None:
    """聚类分析：检测模型身份变化。"""
    from benchmark.analysis.cluster_analyzer import (
        FingerprintClusterAnalyzer,
        ModelClassifier,
    )

    _setup_proxy()
    analyzer = FingerprintClusterAnalyzer()

    if model:
        models = [model]
    else:
        fp_dir = Path("fingerprint_db")
        if not fp_dir.exists():
            console.print("[yellow]No fingerprint data found.[/yellow]")
            return
        models = sorted(
            d.name.replace("__", "/")
            for d in fp_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    for m in models:
        console.print(f"\n[bold]Model: {m}[/bold]")
        report = analyzer.analyze(m)

        if report.n_clusters == 0:
            console.print(f"  {report.summary}")
            continue

        c_color = "red" if report.n_clusters > 1 else "green"
        console.print(
            f"  Clusters: [{c_color}]{report.n_clusters}[/{c_color}]  "
            f"Noise: {report.n_noise}  {report.summary}"
        )
        for c in report.clusters:
            console.print(
                f"    Cluster {c.cluster_id}: {c.size} samples, "
                f"avg_score={c.avg_score:.1f}, "
                f"{c.time_range[0][:16]} ~ {c.time_range[1][:16]}"
            )
        if report.suspected_changes:
            console.print("  [bold red]Suspected model changes:[/bold red]")
            for change in report.suspected_changes:
                console.print(
                    f"    {change['at'][:19]}: "
                    f"cluster {change['from_cluster']} → {change['to_cluster']} "
                    f"(similarity={change['cosine_similarity']:.4f})"
                )

    if classify and len(models) >= 2:
        console.print("\n[bold]Cross-model classification:[/bold]")
        clf = ModelClassifier()
        train_report = clf.train()
        if train_report["status"] != "trained":
            console.print(f"  {train_report.get('message', 'Training failed')}")
            return

        console.print(f"  Trained on {train_report['total_samples']} samples")
        cv = clf.cross_validate()
        console.print(f"  LOO accuracy: {cv['accuracy']:.1%}")
        for name, acc in cv.get("per_model", {}).items():
            console.print(f"    {name}: {acc:.1%}")
