from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from benchmark.cli.registry import (
    DATASET_REGISTRY,
    DIMENSION_REGISTRY,
    THINKING_SYSTEM_MESSAGE,
)
from benchmark.cli.utils import get_provider_concurrency, setup_proxy
from benchmark.core.evaluator import SingleTurnEvaluator
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.logging_config import setup_logging
from benchmark.core.timing_tracker import (
    TimingTracker,
    get_timing_collector,
    start_timing_collection,
    stop_timing_collection,
)
from benchmark.repository import FileRepository
from benchmark.scorers.composite import CompositeScorer

console = Console(force_terminal=True)
logger = logging.getLogger(__name__)


@click.command()
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
    if not debug:
        debug = ctx.obj.get("debug", False)
    setup_proxy()

    models = [m.strip() for m in model.split(",") if m.strip()]

    if dimension == "all":
        dimensions = list(DIMENSION_REGISTRY.keys())
    else:
        dimensions = [dimension]

    async def _run_evaluation_with_timing():
        import sys

        print("[BENCHMARK] Starting evaluation...", flush=True, file=sys.stderr)
        await start_timing_collection("benchmark/data")
        print("[BENCHMARK] Running multi-evaluation...", flush=True, file=sys.stderr)
        try:
            from benchmark.cli.runner import run_multi_evaluation

            await run_multi_evaluation(models, dimensions, samples, debug)
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
    repo: FileRepository,
    run_id: str,
    total: int,
    debug: bool,
    system_message: str | None = None,
    dimension: str = "",
) -> dict[str, Any]:
    timing = TimingTracker()
    result_id = str(uuid.uuid4())[:12]

    async def _do_evaluate() -> dict[str, Any]:
        nonlocal result_id
        try:
            logger.info(
                f"[TASK] 开始 | task_id={task.task_id} | model={model} | dimension={dimension}"
            )

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

            logger.info(f"[TASK] 评分开始 | task_id={task.task_id}")
            timing.start_phase("score_calculation")
            score_result = await scorer.ascore(ctx)
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

            logger.info(
                f"[TASK] 数据保存开始 | task_id={task.task_id} | result_id={result_id}"
            )
            timing.start_phase("db_write")

            tps = gm.get("tokens_per_second", 0.0)
            metrics_data = {
                "prompt_tokens": gm.get("prompt_tokens", 0),
                "completion_tokens": gm.get("completion_tokens", 0),
                "reasoning_tokens": gm.get("reasoning_tokens", 0),
                "reasoning_content": ctx.reasoning_content,
                "duration": api_duration,
                "tokens_per_second": tps,
                "ttft": gm.get("ttft", 0.0),
                "ttft_content": gm.get("ttft_content", 0.0),
            }

            answer_data = {
                "result_id": result_id,
                "task_content": task.prompt,
                "model_output": ctx.raw_output,
                "model_think": ctx.reasoning_content,
                "model_answer": ctx.model_answer,
                "expected_output": task.expected_output,
                "functional_score": functional_score,
                "quality_score": quality_score,
                "final_score": final_score,
                "passed": passed,
                "details": score_details,
                "execution_time": api_duration,
            }

            await asyncio.to_thread(
                repo.save_question_result,
                run_id,
                task.task_id,
                answer_data,
                metrics_data,
            )

            scoring_data = {
                "task_id": task.task_id,
                "functional_score": functional_score,
                "quality_score": quality_score,
                "final_score": final_score,
                "passed": passed,
                "details": score_details,
                "reasoning": score_reasoning,
                "scoring_status": "completed",
            }
            await asyncio.to_thread(
                repo.save_question_scoring,
                run_id,
                task.task_id,
                scoring_data,
            )

            timing.end_phase("db_write")
            logger.info(
                f"[TASK] 数据保存完成 | task_id={task.task_id} | result_id={result_id}"
            )

            # 质量信号采集
            logger.info(f"[TASK] 质量信号采集开始 | task_id={task.task_id}")
            timing.start_phase("quality_signals")
            try:
                from benchmark.analysis.quality_signals import QualitySignalCollector

                qsc = QualitySignalCollector(repo=repo, model=model)
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

    try:
        timing_collector = get_timing_collector()
        timing_collector.collect(timing, result_id, model, task.task_id, run_id=run_id)
    except RuntimeError:
        logger.debug("Timing collector not initialized, skipping timing collection")

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


async def run_evaluation(
    model: str,
    dimension: str,
    samples: int,
    debug: bool,
    repo: FileRepository | None = None,
) -> None:
    adapter_cls, scorer_factory, evaluator_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    evaluator = evaluator_cls()
    llm = LLMEvalAdapter(model=model)
    _scorers_list = scorer_factory()
    if len(_scorers_list) == 1 and abs(_scorers_list[0][0] - 1.0) < 1e-9:
        # 单 scorer 权重为 1.0 时直接透传，避免 CompositeScorer 覆盖 passed 判断
        scorer = _scorers_list[0][1]
    else:
        scorer = CompositeScorer(_scorers_list)

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

    own_repo = repo is None
    if own_repo:
        repo = FileRepository()
    run_id: str = ""
    try:
        question_ids = [t.task_id for t in tasks]
        run_id = await asyncio.to_thread(
            repo.create_benchmark_run,
            model=model,
            dimension=dimension,
            dataset=DATASET_REGISTRY[dimension],
            questions=question_ids,
        )

        coros = [
            _evaluate_task(
                i,
                task,
                model,
                llm,
                scorer,
                evaluator,
                repo,
                run_id,
                len(tasks),
                debug,
                system_message=THINKING_SYSTEM_MESSAGE,
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

            analyzer = StabilityAnalyzer(repo=repo)
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
                results = await repo.aget_results(
                    model=model,
                    dimension="probe",
                    run_id=run_id,
                )
                signals = await repo.aget_quality_signals_for_run(run_id)
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
                    await repo.asave_cluster_report(cluster_report)
            except Exception as exc:
                logger.warning(f"聚类分析失败: {exc}")

        if failed_count == 0:
            await asyncio.to_thread(repo.finish_run, run_id, "completed")
        elif failed_count < len(tasks):
            await asyncio.to_thread(repo.finish_run, run_id, "completed")
        else:
            await asyncio.to_thread(repo.finish_run, run_id, "failed")

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
            if run_id:
                await asyncio.to_thread(repo.finish_run, run_id, "failed")
        except Exception:
            pass
        raise
    finally:
        if own_repo:
            pass
        await llm.close()
