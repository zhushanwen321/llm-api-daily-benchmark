"""趋势图组件。展示分数随时间变化的趋势."""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import sqlite3


def get_trend_data(
    conn: sqlite3.Connection,
    model: str,
    dimension: str,
    days: int = 30
) -> dict[str, list]:
    """从数据库获取趋势数据.

    Args:
        conn: SQLite 连接.
        model: 模型名称.
        dimension: 评测维度.
        days: 天数范围.

    Returns:
        包含 dates 和 scores 的字典.
    """
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    query = """
        SELECT
            DATE(r.created_at) as date,
            AVG(r.final_score) as avg_score
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        WHERE e.model = ?
          AND e.dimension = ?
          AND DATE(r.created_at) >= ?
        GROUP BY DATE(r.created_at)
        ORDER BY date ASC
    """

    # 设置 row_factory 以获取字典形式的结果
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query, (model, dimension, cutoff_date))
    rows = cursor.fetchall()

    return {
        "dates": [row["date"] for row in rows],
        "scores": [row["avg_score"] for row in rows]
    }


def create_trend_figure(
    data: dict[str, list],
    title: str = "Score Trend"
) -> plt.Figure:
    """创建趋势图.

    Args:
        data: 包含 dates 和 scores 的字典.
        title: 图表标题.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if not data["dates"]:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    ax.plot(data["dates"], data["scores"], marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # 设置 Y 轴范围
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_multi_model_trend(
    conn: sqlite3.Connection,
    models: list[str],
    dimension: str,
    days: int = 30
) -> plt.Figure:
    """创建多模型对比趋势图.

    Args:
        conn: SQLite 连接.
        models: 模型名称列表.
        dimension: 评测维度.
        days: 天数范围.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for model in models:
        data = get_trend_data(conn, model, dimension, days)
        if data["dates"]:
            ax.plot(data["dates"], data["scores"], marker="o", label=model, linewidth=2, markersize=4)

    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title(f"{dimension} - Model Comparison (Last {days} days)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig
