from __future__ import annotations

import click
from rich.console import Console

from benchmark.core.reporter import generate_html_report

console = Console(force_terminal=True)


@click.command()
@click.option("--models", default=None, help="逗号分隔的模型列表")
@click.option("--dimensions", default=None, help="逗号分隔的维度列表")
@click.option(
    "--date-range", default=None, help="日期范围，格式: 2026-04-01,2026-04-30"
)
@click.option("--output", default="report.html", help="输出文件路径")
def report(
    models: str | None, dimensions: str | None, date_range: str | None, output: str
) -> None:
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
