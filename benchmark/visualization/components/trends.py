"""趋势图组件。展示分数随时间变化的趋势."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import sqlite3

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# 暗色主题配色方案
DARK_BG = "#0e1117"  # Streamlit 暗色背景
DARK_BG_SECONDARY = "#1e2129"
TEXT_COLOR = "#fafafa"
TEXT_COLOR_SECONDARY = "#b0b3b8"
GRID_COLOR = "#2d3139"
LINE_COLORS = [
    "#4ecdc4",
    "#ff6b6b",
    "#45b7d1",
    "#96ceb4",
    "#ffeaa7",
    "#dfe6e9",
    "#fd79a8",
]


def apply_dark_theme(fig: Figure, ax: Axes) -> None:
    """应用暗色主题到 matplotlib 图表."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR_SECONDARY)
    ax.yaxis.label.set_color(TEXT_COLOR_SECONDARY)
    ax.tick_params(axis="both", colors=TEXT_COLOR_SECONDARY)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def get_trend_data(
    conn: sqlite3.Connection, model: str, dimension: str, days: int = 30
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
        "scores": [row["avg_score"] for row in rows],
    }


def create_trend_figure(data: dict[str, list], title: str = "Score Trend") -> Figure:
    """创建趋势图."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    apply_dark_theme(fig, ax)

    if not data["dates"]:
        ax.text(
            0.5, 0.5, "No data available", ha="center", va="center", color=TEXT_COLOR
        )
        return fig

    ax.plot(
        data["dates"],
        data["scores"],
        marker="o",
        linewidth=1.5,
        markersize=3,
        color=LINE_COLORS[0],
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_multi_model_trend(
    conn: sqlite3.Connection, models: list[str], dimension: str, days: int = 30
) -> Figure:
    """创建多模型对比趋势图."""
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=100)
    apply_dark_theme(fig, ax)

    for i, model in enumerate(models):
        data = get_trend_data(conn, model, dimension, days)
        if data["dates"]:
            color = LINE_COLORS[i % len(LINE_COLORS)]
            ax.plot(
                data["dates"],
                data["scores"],
                marker="o",
                label=model,
                linewidth=1.5,
                markersize=3,
                color=color,
            )

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        f"{dimension} - Model Comparison (Last {days} days)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        loc="best",
        fontsize=8,
        facecolor=DARK_BG_SECONDARY,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig
