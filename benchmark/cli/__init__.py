from __future__ import annotations

import click
from rich.console import Console

from benchmark.core.logging_config import setup_logging

# re-export 供外部引用（scheduler.py / tests 等依赖这些符号）
from benchmark.cli.registry import (  # noqa: F401
    DATASET_REGISTRY,
    DIMENSION_REGISTRY,
    THINKING_SYSTEM_MESSAGE,
)
from benchmark.cli.runner import (  # noqa: F401
    run_multi_evaluation as _run_multi_evaluation,
    run_provider_group as _run_provider_group,
    group_by_provider as _group_by_provider,
)
from benchmark.cli.utils import (  # noqa: F401
    get_provider_concurrency as _get_provider_concurrency,
    setup_proxy as _setup_proxy,
)

console = Console(force_terminal=True)


@click.group()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="启用 debug 模式，输出详细日志到控制台和 logs/ 目录",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    setup_logging(debug=debug)


from benchmark.cli.commands.analyze import analyze  # noqa: E402
from benchmark.cli.commands.download import download  # noqa: E402
from benchmark.cli.commands.evaluate import evaluate  # noqa: E402
from benchmark.cli.commands.export import export  # noqa: E402
from benchmark.cli.commands.list_datasets import list_datasets  # noqa: E402
from benchmark.cli.commands.report import report  # noqa: E402
from benchmark.cli.commands.scheduler import scheduler  # noqa: E402

cli.add_command(evaluate)
cli.add_command(export)
cli.add_command(report)
cli.add_command(download)
cli.add_command(analyze)
cli.add_command(scheduler)
cli.add_command(list_datasets)
