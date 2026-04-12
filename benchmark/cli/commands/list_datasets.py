import click
from rich.console import Console

console = Console(force_terminal=True)


@click.command("list-datasets")
def list_datasets() -> None:
    console.print("[bold]Available datasets:[/bold]")
    console.print("  [cyan]reasoning:[/cyan]           MATH (Level 3-5, 15 tasks)")
    console.print("  [cyan]backend-dev:[/cyan]        BigCodeBench-Hard (15 tasks)")
    console.print("  [cyan]frontend-dev:[/cyan]       FrontCode (自建前端评测)")
