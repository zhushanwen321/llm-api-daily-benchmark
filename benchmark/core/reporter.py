"""报告生成器。从数据库读取结果，生成 HTML 报告."""

from __future__ import annotations

import json
import math
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
        rows = _query_results(db, models=models, dimensions=dimensions, date_range=date_range)
        if not rows:
            raise ValueError("No results found in database")

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

        # 构建雷达图
        radar_charts: dict[str, dict[str, str]] = {}
        for model in model_list:
            radar_charts[model] = {}
            for dim in dim_list:
                axes = _DIMENSION_AXES.get(dim, [])
                if not axes:
                    continue
                scores = _extract_dimension_scores(rows, dim, model)
                if scores:
                    svg = _build_radar_svg(scores, axes)
                    radar_charts[model][dim] = svg

        # 构建维度子分数表
        dim_score_table = _build_dimension_score_table(rows)

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
            radar_charts=radar_charts,
            dim_score_table=dim_score_table,
            dimension_axes=_DIMENSION_AXES,
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


_DIMENSION_AXES = {
    "backend-dev": [
        ("test_coverage", "测试覆盖"),
        ("performance", "性能"),
        ("code_style", "风格"),
        ("robustness", "健壮性"),
        ("architecture", "架构"),
        ("security", "安全"),
        ("extensibility", "可扩展性"),
    ],
    "frontend-dev": [
        ("functionality", "功能"),
        ("html_semantic", "语义化"),
        ("accessibility", "可访问性"),
        ("css_quality", "CSS质量"),
        ("code_organization", "代码组织"),
        ("performance", "性能"),
        ("browser_compat", "兼容性"),
    ],
    "reasoning": [
        ("answer_correctness", "答案正确性"),
        ("reasoning_completeness", "完整性"),
        ("reasoning_validity", "推理正确性"),
        ("method_elegance", "方法优雅度"),
        ("difficulty_adaptation", "难度适配"),
    ],
    "system-architecture": [
        ("answer_correctness", "答案正确性"),
        ("reasoning_completeness", "推理完整性"),
        ("option_analysis", "选项分析"),
        ("reasoning_confidence", "推理置信度"),
        ("subject_adaptation", "学科适配"),
    ],
}


def _query_results(db: Database, models=None, dimensions=None, date_range=None) -> list[dict]:
    """查询评测结果，包含 details 字段。"""
    conn = db._get_conn()
    query = """
        SELECT r.result_id, e.model, e.dimension,
               r.task_id, r.final_score, r.passed,
               r.execution_time, r.created_at,
               r.details
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        WHERE 1=1
    """
    params: list = []
    if models:
        placeholders = ",".join("?" for _ in models)
        query += f" AND e.model IN ({placeholders})"
        params.extend(models)
    if dimensions:
        placeholders = ",".join("?" for _ in dimensions)
        query += f" AND e.dimension IN ({placeholders})"
        params.extend(dimensions)
    if date_range:
        start, end = date_range
        query += " AND r.created_at >= ? AND r.created_at <= ?"
        params.extend([start, end + " 23:59:59"])
    query += " ORDER BY r.created_at DESC"
    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _extract_dimension_scores(rows: list[dict], dimension: str, model: str) -> dict[str, float]:
    """从评测结果中提取指定模型在指定维度的子分数。"""
    model_dim_rows = [r for r in rows if r["model"] == model and r["dimension"] == dimension]
    if not model_dim_rows:
        return {}
    all_sub_scores: dict[str, list[float]] = {}
    for row in model_dim_rows:
        details_str = row.get("details", "")
        if not details_str:
            continue
        try:
            details = json.loads(details_str) if isinstance(details_str, str) else details_str
        except (json.JSONDecodeError, TypeError):
            continue
        composite = details.get("composite", {})
        sub_scores = composite.get("scores", {})
        for key, val in sub_scores.items():
            if isinstance(val, (int, float)):
                all_sub_scores.setdefault(key, []).append(float(val))
    if not all_sub_scores:
        return {}
    return {k: sum(v) / len(v) for k, v in all_sub_scores.items()}


def _build_radar_svg(scores: dict[str, float], axes: list[tuple[str, str]], width: int = 400, height: int = 400, radius: int = 140) -> str:
    """生成雷达图 SVG 字符串。"""
    n = len(axes)
    if n < 3:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle">数据不足（需要至少 3 个维度）</text></svg>'
    cx, cy = width / 2, height / 2

    def _point(index: int, value: float) -> tuple[float, float]:
        angle = 2 * math.pi * index / n - math.pi / 2
        r = radius * (value / 100.0)
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    grid_levels = [0.33, 0.66, 1.0]
    grid_paths = []
    for level in grid_levels:
        points = []
        for i in range(n):
            x, y = _point(i, level * 100)
            points.append(f"{x:.1f},{y:.1f}")
        grid_paths.append(f'<polygon points="{" ".join(points)}" fill="none" stroke="#e5e7eb" stroke-width="1"/>')

    axis_lines = []
    for i in range(n):
        x, y = _point(i, 100)
        axis_lines.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1"/>')

    data_polygon = ""
    if scores:
        data_points = []
        for i, (key, label) in enumerate(axes):
            val = scores.get(key, 0)
            x, y = _point(i, val)
            data_points.append(f"{x:.1f},{y:.1f}")
        data_polygon = f'<polygon points="{" ".join(data_points)}" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" stroke-width="2"/>'

    data_dots = []
    if scores:
        for i, (key, label) in enumerate(axes):
            val = scores.get(key, 0)
            x, y = _point(i, val)
            data_dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#3b82f6"/>')

    labels = []
    for i, (key, label) in enumerate(axes):
        x, y = _point(i, 115)
        val = scores.get(key, 0) if scores else 0
        anchor = "middle"
        dx, dy = 0, 0
        angle = 2 * math.pi * i / n - math.pi / 2
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        if cos_a > 0.3:
            anchor = "start"
            dx = 5
        elif cos_a < -0.3:
            anchor = "end"
            dx = -5
        if sin_a < -0.3:
            dy = -5
        elif sin_a > 0.3:
            dy = 10
        labels.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" font-size="11" fill="#374151">{label}</text>')
        fx, fy = _point(i, val)
        labels.append(f'<text x="{fx:.1f}" y="{fy - 10:.1f}" text-anchor="middle" font-size="10" fill="#3b82f6" font-weight="bold">{val:.0f}</text>')

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        *grid_paths,
        *axis_lines,
        data_polygon,
        *data_dots,
        *labels,
        "</svg>",
    ]
    return "\n".join(parts)


def _build_dimension_score_table(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """构建维度子分数表。Returns: {model: {dimension: {sub_dimension: avg_score}}}"""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        model = row.get("model", "")
        dimension = row.get("dimension", "")
        details_str = row.get("details", "")
        if not details_str:
            continue
        try:
            details = json.loads(details_str) if isinstance(details_str, str) else details_str
        except (json.JSONDecodeError, TypeError):
            continue
        composite = details.get("composite", {})
        sub_scores = composite.get("scores", {})
        if not sub_scores:
            continue
        result.setdefault(model, {}).setdefault(dimension, {})
        for key, val in sub_scores.items():
            if isinstance(val, (int, float)):
                bucket = result[model][dimension]
                if key not in bucket:
                    bucket[key] = {"sum": 0.0, "count": 0}
                bucket[key]["sum"] += float(val)
                bucket[key]["count"] += 1
    for model_dims in result.values():
        for dim_scores in model_dims.values():
            for key in list(dim_scores.keys()):
                bucket = dim_scores[key]
                dim_scores[key] = bucket["sum"] / bucket["count"] if bucket["count"] > 0 else 0.0
    return result
