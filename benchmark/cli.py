"""CLI 命令入口。支持 evaluate / list-datasets / export 命令."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
from benchmark.adapters.gsm8k_adapter import GSM8KAdapter
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.logging_config import setup_logging
from benchmark.core.response_parser import parse_response
from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun
from benchmark.scorers.execution_scorer import ExecutionScorer
from benchmark.scorers.exact_match_scorer import ExactMatchScorer

console = Console()
logger = logging.getLogger(__name__)

DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (GSM8KAdapter, ExactMatchScorer),
    "backend-dev": (BigCodeBenchAdapter, ExecutionScorer),
}

DATASET_REGISTRY: dict[str, str] = {
    "reasoning": "gsm8k",
    "backend-dev": "bigcodebench",
}


def _setup_proxy() -> None:
    """从 .env 加载代理配置，用于 HuggingFace 数据集下载."""
    load_dotenv()
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
    type=click.Choice(["reasoning", "backend-dev"]),
    help="评测维度",
)
@click.option("--samples", default=5, help="评测题目数量")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="启用 debug 模式（也可放在 'benchmark' 命令后）",
)
@click.pass_context
def evaluate(
    ctx: click.Context, model: str, dimension: str, samples: int, debug: bool
) -> None:
    # 如果子命令没有指定 debug，使用父命令的设置
    if not debug:
        debug = ctx.obj.get("debug", False)
    _setup_proxy()
    if dimension not in DIMENSION_REGISTRY:
        console.print(f"[red]Unknown dimension: {dimension}[/red]")
        raise SystemExit(1)

    asyncio.run(_run_evaluation(model, dimension, samples, debug))


async def _evaluate_task(
    task_idx: int,
    task: Any,
    model: str,
    llm: LLMEvalAdapter,
    scorer: Any,
    db: Database,
    run_id: str,
    total: int,
    debug: bool,
) -> dict[str, Any]:
    """单个 task 的异步评测协程。返回包含结果的字典，失败时返回 error 字段。"""
    try:
        logger.debug(f"处理任务 {task_idx + 1}/{total}: {task.task_id}")
        start_time = datetime.now()
        gen_response = await llm.agenerate(task.prompt, model=model)
        execution_time = (datetime.now() - start_time).total_seconds()
        model_output = gen_response.content

        logger.debug(
            f"任务 {task.task_id} 生成完成，输出长度: {len(model_output)} 字符"
        )
        if debug:
            logger.debug(f"模型输出:\n{model_output[:500]}...")

        # 推理内容直接从 API 层获取（adapter 已分离）
        think_content = gen_response.reasoning_content

        # 从 content 中解析最终答案
        parsed = parse_response(model_output, task.dimension)
        logger.debug(
            f"任务 {task.task_id} 解析完成: think_len={len(think_content)}, answer_len={len(parsed.answer)}"
        )

        # 使用解析后的 answer 进行评分
        score_result = scorer.score(parsed.answer, task.expected_output, task)
        logger.debug(
            f"任务 {task.task_id} 评分结果: score={score_result.score}, passed={score_result.passed}"
        )

        result_id = str(uuid.uuid4())[:12]
        result = EvalResult(
            result_id=result_id,
            run_id=run_id,
            task_id=task.task_id,
            task_content=task.prompt,
            model_output=model_output,
            model_think=think_content,
            model_answer=parsed.answer,
            functional_score=score_result.score,
            final_score=score_result.score,
            passed=score_result.passed,
            details=score_result.details,
            execution_time=execution_time,
            created_at=datetime.now(),
        )
        db.save_result(result)

        tps = (
            gen_response.tokens_per_second
            if gen_response.tokens_per_second > 0
            else (
                gen_response.completion_tokens / execution_time
                if execution_time > 0
                else 0.0
            )
        )
        db.save_metrics(
            ApiCallMetrics(
                result_id=result_id,
                prompt_tokens=gen_response.prompt_tokens,
                completion_tokens=gen_response.completion_tokens,
                reasoning_tokens=gen_response.reasoning_tokens,
                reasoning_content=gen_response.reasoning_content,
                duration=gen_response.duration,
                tokens_per_second=tps,
                ttft_content=gen_response.ttft_content,
                created_at=datetime.now(),
            )
        )

        status_icon = (
            "[green]PASS[/green]" if score_result.passed else "[red]FAIL[/red]"
        )
        console.print(
            f"  [{task_idx + 1}/{total}] {task.task_id} | "
            f"Score: {score_result.score:.0f} | {status_icon} | "
            f"Time: {execution_time:.1f}s | "
            f"TTFT-R: {gen_response.ttft:.2f}s | "
            f"TTFT-C: {gen_response.ttft_content:.2f}s | "
            f"Speed: {tps:.1f} tok/s"
        )

        return {
            "score": score_result.score,
            "passed": score_result.passed,
            "task_id": task.task_id,
        }
    except Exception as exc:
        logger.error(f"任务 {task.task_id if hasattr(task, 'task_id') else task_idx} 失败: {exc}")
        status_msg = f"[red]ERROR: {type(exc).__name__}: {exc}[/red]"
        console.print(f"  [{task_idx + 1}/{total}] {getattr(task, 'task_id', '?')} | {status_msg}")
        return {
            "error": exc,
            "task_id": getattr(task, "task_id", str(task_idx)),
            "passed": False,
            "score": 0.0,
        }


async def _run_evaluation(
    model: str, dimension: str, samples: int, debug: bool
) -> None:
    """异步评测主流程，使用 asyncio.gather 并发执行所有 task。"""
    adapter_cls, scorer_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    scorer = scorer_cls()
    llm = LLMEvalAdapter(model=model)

    logger.debug(f"加载适配器: {adapter_cls.__name__}, 评分器: {scorer_cls.__name__}")

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
    db = Database()
    try:
        db.create_run(run)

        coros = [
            _evaluate_task(i, task, model, llm, scorer, db, run_id, len(tasks), debug)
            for i, task in enumerate(tasks)
        ]

        total_score = 0.0
        passed_count = 0
        failed_count = 0
        with Progress() as progress:
            task_progress = progress.add_task("Evaluating", total=len(tasks))
            for coro in asyncio.as_completed(coros):
                r = await coro
                progress.advance(task_progress)
                if isinstance(r, Exception):
                    failed_count += 1
                    continue
                if r.get("error"):
                    failed_count += 1
                total_score += r["score"]
                if r["passed"]:
                    passed_count += 1

        if failed_count == 0:
            db.finish_run(run_id, "completed")
        else:
            db.finish_run(run_id, "partial" if failed_count < len(tasks) else "failed")

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
            db.finish_run(run_id, "failed")
        except Exception:
            pass
        raise
    finally:
        db.close()


@cli.command("list-datasets")
def list_datasets() -> None:
    """列出可用数据集."""
    console.print("[bold]Available datasets:[/bold]")
    console.print("  [cyan]reasoning:[/cyan]     GSM8K (hardest 5 tasks by step count)")
    console.print("  [cyan]backend-dev:[/cyan]  BigCodeBench-Hard (5 tasks)")


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
