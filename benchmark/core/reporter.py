"""报告生成器。从数据库读取结果，生成 HTML 报告."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2

from benchmark.core.advanced_statistics import pairwise_comparison
from benchmark.models.database import Database


def generate_html_report(
    run_ids: list[str] | None = None,
    models: list[str] | None = None,
    dimensions: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    output_path: str = "report.html",
) -> str:
    """生成 HTML 报告."""
    db = Database()
    try:
        results = db.get_results(model=None, dimension=None)
        if not results:
            raise ValueError("No results found in database")

        # 过滤
        rows = [dict(r) for r in results]
        if models:
            rows = [r for r in rows if r["model"] in models]
        if dimensions:
            rows = [r for r in rows if r["dimension"] in dimensions]
        if date_range:
            start, end = date_range
            rows = [r for r in rows if start <= str(r.get("created_at", ""))[:10] <= end]

        # 提取模型和维度列表
        model_list = sorted(set(r["model"] for r in rows))
        dim_list = sorted(set(r["dimension"] for r in rows))

        # 构建得分表
        score_table = _build_score_table(rows, model_list, dim_list)

        # 统计检验
        stat_tests = []
        if len(model_list) >= 2:
            model_scores = {}
            for model in model_list:
                model_rows = [r for r in rows if r["model"] == model]
                model_scores[model] = [float(r.get("final_score", 0)) for r in model_rows]
            stat_tests = pairwise_comparison(model_scores)

        # 详细结果
        detailed = _build_detailed(rows, model_list, dim_list)

        # 渲染
        template_dir = Path(__file__).parent.parent / "templates"
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
        template = env.get_template("report.html")

        html = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            models=model_list,
            dimensions=dim_list,
            date_range=date_range if date_range else "All time",
            score_table=score_table,
            stat_tests=stat_tests,
            detailed=detailed,
        )

        output = Path(output_path)
        output.write_text(html, encoding="utf-8")
        return str(output)
    finally:
        db.close()


def _build_score_table(
    rows: list[dict], models: list[str], dimensions: list[str]
) -> list[dict]:
    """构建得分汇总表."""
    table = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        scores = {}
        for dim in dimensions:
            dim_rows = [r for r in model_rows if r["dimension"] == dim]
            if dim_rows:
                score_vals = [float(r.get("final_score", 0)) for r in dim_rows]
                passed_count = sum(1 for r in dim_rows if r.get("passed"))
                scores[dim] = {
                    "mean": sum(score_vals) / len(score_vals) if score_vals else 0,
                    "passed": passed_count,
                    "total": len(dim_rows),
                }
            else:
                scores[dim] = {"mean": 0, "passed": 0, "total": 0}

        all_means = [s["mean"] for s in scores.values() if s["total"] > 0]
        avg = sum(all_means) / len(all_means) if all_means else 0
        table.append({"model": model, "scores": scores, "average": avg})
    return table


def _build_detailed(
    rows: list[dict], models: list[str], dimensions: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """构建详细结果."""
    detailed = {}
    for dim in dimensions:
        dim_rows = [r for r in rows if r["dimension"] == dim]
        tasks: dict[str, dict[str, Any]] = {}
        for r in dim_rows:
            tid = r.get("task_id", "")
            tasks.setdefault(tid, {})
            tasks[tid][r["model"]] = {
                "score": float(r.get("final_score", 0)),
                "passed": bool(r.get("passed")),
            }
        detailed[dim] = tasks
    return detailed
