from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console

from benchmark.cli.utils import setup_proxy

console = Console(force_terminal=True)


@click.command()
@click.option(
    "--output-dir",
    default="benchmark/datasets",
    help="缓存输出目录",
)
def download(output_dir: str) -> None:
    setup_proxy()

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
