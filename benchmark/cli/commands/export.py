from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import click
from rich.console import Console

from benchmark.repository import FileRepository

console = Console(force_terminal=True)


@click.command()
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
    async def _do_export() -> None:
        repo = FileRepository()
        results = await repo.aget_results(model=model, dimension=dimension)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        output_path = Path(output)

        if fmt == "json":
            data = [dict(row) for row in results]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            console.print(
                f"[green]Exported {len(data)} results to {output_path}[/green]"
            )

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

    asyncio.run(_do_export())
