"""多模型 x 多维度评测调度。"""

from __future__ import annotations

import logging
from typing import Any

from benchmark.cli.registry import DIMENSION_REGISTRY
from benchmark.repository import FileRepository

logger = logging.getLogger(__name__)


def group_by_provider(
    models: list[str], dimensions: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """按 provider 分组 (model, dimension) 对。"""
    groups: dict[str, list[tuple[str, str]]] = {}
    for model in models:
        provider = model.split("/", 1)[0]
        for dim in dimensions:
            groups.setdefault(provider, []).append((model, dim))
    return groups


async def run_provider_group(
    tasks: list[tuple[str, str]], samples: int, debug: bool, repo: FileRepository
) -> None:
    """同一 provider 内的 evaluation run 并发执行。"""
    import asyncio
    from rich.console import Console

    console = Console(force_terminal=True)

    if not tasks:
        return

    from benchmark.cli.commands.evaluate import run_evaluation

    coros = [
        run_evaluation(model, dim, samples, debug, repo=repo) for model, dim in tasks
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            model, dim = tasks[i]
            logger.error(f"Task failed for {model}/{dim}: {result}")
            console.print(f"[red]Error: Evaluation failed for {model}/{dim}[/red]")


async def run_multi_evaluation(
    models: list[str], dimensions: list[str], samples: int, debug: bool
) -> None:
    """多模型 x 多维度评测。共享单一 FileRepository 实例。

    每个维度内部的任务仍然并发执行。
    """
    repo = FileRepository()
    groups = group_by_provider(models, dimensions)
    for provider, tasks in groups.items():
        logger.info(f"[EVAL] 开始评测 provider: {provider} | 维度数: {len(tasks)}")
        await run_provider_group(tasks, samples, debug, repo=repo)
        logger.info(f"[EVAL] 完成评测 provider: {provider}")
