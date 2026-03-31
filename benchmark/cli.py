"""CLI 命令入口。支持 evaluate / list-datasets / export 命令."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
from benchmark.adapters.gsm8k_adapter import GSM8KAdapter
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.models.database import Database
from benchmark.models.schemas import EvalResult, EvalRun
from benchmark.scorers.execution_scorer import ExecutionScorer
from benchmark.scorers.exact_match_scorer import ExactMatchScorer

console = Console()

DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (GSM8KAdapter, ExactMatchScorer),
    "backend-dev": (BigCodeBenchAdapter, ExecutionScorer),
}

DATASET_REGISTRY: dict[str, str] = {
    "reasoning": "gsm8k",
    "backend-dev": "bigcodebench",
}


@click.group()
def cli():
    """LLM Benchmark 评测工具."""
    pass


@cli.command()
@click.option(
    "--model", required=True, help="模型名称（需在 configs/models.yaml 中配置）"
)
@click.option(
    "--dimension",
    required=True,
    type=click.Choice(["reasoning", "backend-dev"]),
    help="评测维度",
)
@click.option("--samples", default=5, help="评测题目数量")
def evaluate(model: str, dimension: str, samples: int) -> None:
    """运行评测。调用 LLM 生成答案，评分并保存结果."""
    if dimension not in DIMENSION_REGISTRY:
        console.print(f"[red]Unknown dimension: {dimension}[/red]")
        raise SystemExit(1)

    adapter_cls, scorer_cls = DIMENSION_REGISTRY[dimension]
    adapter = adapter_cls()
    scorer = scorer_cls()
    llm = LLMEvalAdapter()
    db = Database()

    tasks = adapter.load()[:samples]
    if not tasks:
        console.print("[red]No tasks loaded.[/red]")
        raise SystemExit(1)

    console.print(
        f"[bold green]Starting evaluation:[/bold green] "
        f"{dimension} with {len(tasks)} tasks, model={model}"
    )

    run_id = str(uuid.uuid4())[:8]
    run = EvalRun(
        run_id=run_id,
        model=model,
        dimension=dimension,
        dataset=DATASET_REGISTRY[dimension],
        started_at=datetime.now(),
        status="running",
    )
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
                result_id=str(uuid.uuid4())[:8],
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
