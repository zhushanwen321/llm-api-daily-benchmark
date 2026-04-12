from __future__ import annotations

import click
from rich.console import Console

console = Console(force_terminal=True)


@click.group()
def scheduler() -> None:
    pass


@scheduler.command()
def start() -> None:
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
    from benchmark.core.scheduler import BenchmarkScheduler

    sched = BenchmarkScheduler()
    sched.stop()
    console.print("[green]调度器已停止[/green]")


@scheduler.command()
def status() -> None:
    from benchmark.core.scheduler import BenchmarkScheduler

    sched = BenchmarkScheduler()
    console.print(f"  Enabled: {sched.enabled}")
    console.print(f"  Cron: {sched.cron}")
    console.print(f"  Models: {sched.models}")
    console.print(f"  Dimensions: {sched.dimensions}")
    console.print(f"  Samples: {sched.samples}")
