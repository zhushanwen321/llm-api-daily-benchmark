from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from benchmark.cli.utils import setup_proxy

console = Console(force_terminal=True)


@click.command("analyze")
@click.option("--model", default=None, help="要分析的模型（不指定则分析全部）")
@click.option("--classify", is_flag=True, default=False, help="同时运行跨模型分类")
def analyze(model: str | None, classify: bool) -> None:
    from benchmark.analysis.cluster_analyzer import (
        FingerprintClusterAnalyzer,
        ModelClassifier,
    )

    setup_proxy()
    analyzer = FingerprintClusterAnalyzer()

    if model:
        models = [model]
    else:
        fp_dir = Path("fingerprint_db")
        if not fp_dir.exists():
            console.print("[yellow]No fingerprint data found.[/yellow]")
            return
        models = sorted(
            d.name.replace("__", "/")
            for d in fp_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    for m in models:
        console.print(f"\n[bold]Model: {m}[/bold]")
        report = analyzer.analyze(m)

        if report.n_clusters == 0:
            console.print(f"  {report.summary}")
            continue

        c_color = "red" if report.n_clusters > 1 else "green"
        console.print(
            f"  Clusters: [{c_color}]{report.n_clusters}[/{c_color}]  "
            f"Noise: {report.n_noise}  {report.summary}"
        )
        for c in report.clusters:
            console.print(
                f"    Cluster {c.cluster_id}: {c.size} samples, "
                f"avg_score={c.avg_score:.1f}, "
                f"{c.time_range[0][:16]} ~ {c.time_range[1][:16]}"
            )
        if report.suspected_changes:
            console.print("  [bold red]Suspected model changes:[/bold red]")
            for change in report.suspected_changes:
                console.print(
                    f"    {change['at'][:19]}: "
                    f"cluster {change['from_cluster']} → {change['to_cluster']} "
                    f"(similarity={change['cosine_similarity']:.4f})"
                )

    if classify and len(models) >= 2:
        console.print("\n[bold]Cross-model classification:[/bold]")
        clf = ModelClassifier()
        train_report = clf.train()
        if train_report["status"] != "trained":
            console.print(f"  {train_report.get('message', 'Training failed')}")
            return

        console.print(f"  Trained on {train_report['total_samples']} samples")
        cv = clf.cross_validate()
        console.print(f"  LOO accuracy: {cv['accuracy']:.1%}")
        for name, acc in cv.get("per_model", {}).items():
            console.print(f"    {name}: {acc:.1%}")
