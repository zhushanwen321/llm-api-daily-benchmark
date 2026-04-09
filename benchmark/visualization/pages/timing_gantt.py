"""Benchmark 耗时分析甘特图页面。

展示评测过程中各阶段的耗时分布，包括等待时间和执行时间。
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from benchmark.models.database import Database


# Phase 颜色映射
PHASE_COLORS: dict[str, str] = {
    "semaphore_wait": "#FF6B6B",
    "llm_request": "#4ECDC4",
    "judge_request": "#45B7D1",
    "score_calculation": "#96CEB4",
    "db_write": "#FFEAA7",
    "quality_signals": "#DFE6E9",
}


def get_default_color() -> str:
    """其他 phase 使用的默认颜色."""
    return "#95A5A6"


@st.cache_resource
def get_db() -> Database:
    """获取数据库连接（缓存）."""
    return Database()


def get_models(db: Database) -> list[str]:
    """从数据库获取唯一模型列表.

    Returns:
        按字母排序的模型名称列表
    """
    df = db.get_timing_phases(limit=10000)
    if df.empty:
        return []
    models = df["model"].dropna().unique().tolist()
    return sorted(models)


def create_gantt_chart(df: pd.DataFrame) -> go.Figure:
    """创建交互式甘特图.

    Args:
        df: 包含 timing_phases 数据的 DataFrame

    Returns:
        plotly.graph_objects.Figure 对象
    """
    if df.empty:
        return go.Figure()

    # 为每个 phase 创建条形
    phases = df["phase_name"].unique()

    fig = go.Figure()

    for phase in phases:
        phase_df = df[df["phase_name"] == phase].copy()
        color = PHASE_COLORS.get(phase, get_default_color())

        # 计算悬停文本
        hover_texts = []
        for _, row in phase_df.iterrows():
            text = (
                f"<b>{row['task_id']}</b><br>"
                f"Duration: {row['duration']:.3f}s<br>"
                f"Wait: {row['wait_time']:.3f}s<br>"
                f"Active: {row['active_time']:.3f}s<br>"
                f"Run: {row['run_id'][:12]}..."
            )
            hover_texts.append(text)

        # 使用 go.Bar 创建水平条形
        fig.add_trace(
            go.Bar(
                x=phase_df["duration"].tolist(),
                y=phase_df["task_id"].tolist(),
                orientation="h",
                name=phase,
                marker_color=color,
                text=[f"{d:.3f}s" for d in phase_df["duration"].tolist()],
                textposition="inside",
                insidetextanchor="start",
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_texts,
            )
        )

    # 更新布局
    fig.update_layout(
        barmode="stack",
        height=max(400, len(df) * 25 + 100),  # 动态高度
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis_title="Duration (seconds)",
        yaxis_title="Task ID",
        margin=dict(l=150, r=50, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def render_timing_gantt_page() -> None:
    """渲染耗时甘特图页面."""
    st.title("⏱️ Benchmark 耗时分析")

    db = get_db()

    # 筛选条件 - 3列布局
    col1, col2, col3 = st.columns(3)

    with col1:
        models = get_models(db)
        model_options = ["All"] + models if models else ["All"]
        selected_model = st.selectbox("Model", model_options)

    with col2:
        run_id_input = st.text_input("Run ID", placeholder="e.g., run_2024_01_01")

    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(None, None),
            help="Select start and end dates",
        )

    # 解析日期范围
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        if date_range[0] and date_range[1]:
            start_date = datetime.combine(date_range[0], datetime.min.time())
            end_date = datetime.combine(date_range[1], datetime.max.time())

    # 获取数据
    model_filter = selected_model if selected_model != "All" else None
    run_id_filter = run_id_input if run_id_input.strip() else None

    df = db.get_timing_phases(
        model=model_filter,
        run_id=run_id_filter,
        start_date=start_date,
        end_date=end_date,
        limit=5000,
    )

    if df.empty:
        st.info("No timing data found for the selected filters.")
        return

    # 甘特图展示
    st.subheader("Phase Timeline")

    # 添加说明
    with st.expander("Color Legend", expanded=True):
        legend_cols = st.columns(len(PHASE_COLORS) + 1)
        for idx, (phase, color) in enumerate(PHASE_COLORS.items()):
            with legend_cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};'
                    f"width:20px;height:20px;display:inline-block;"
                    f'border-radius:3px;margin-right:5px;"></div>'
                    f"<b>{phase}</b>",
                    unsafe_allow_html=True,
                )
        with legend_cols[-1]:
            st.markdown(
                f'<div style="background-color:{get_default_color()};'
                f"width:20px;height:20px;display:inline-block;"
                f'border-radius:3px;margin-right:5px;"></div>'
                f"<b>Other</b>",
                unsafe_allow_html=True,
            )

    fig = create_gantt_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    # 耗时统计区域 - 3列布局
    st.subheader("Timing Statistics")

    total_avg = df["duration"].mean()
    wait_avg = df["wait_time"].mean()
    active_avg = df["active_time"].mean()

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("Avg Total Duration", f"{total_avg:.3f}s")
    with stat_col2:
        st.metric("Avg Wait Time", f"{wait_avg:.3f}s")
    with stat_col3:
        st.metric("Avg Active Time", f"{active_avg:.3f}s")

    # 阶段耗时分布表
    st.subheader("Phase Distribution")

    phase_stats = (
        df.groupby("phase_name")["duration"]
        .agg(["count", "mean", "max", "min", "std"])
        .round(4)
    )
    phase_stats = pd.DataFrame(phase_stats)
    phase_stats.columns = ["Count", "Mean", "Max", "Min", "Std"]
    phase_stats = phase_stats.sort_values("Mean", ascending=False)

    st.dataframe(
        phase_stats,
        use_container_width=True,
    )

    # 详细数据表格
    with st.expander("Raw Data", expanded=False):
        display_df = df[
            [
                "task_id",
                "run_id",
                "phase_name",
                "duration",
                "wait_time",
                "active_time",
                "created_at",
            ]
        ].copy()
        display_df["duration"] = pd.Series(display_df["duration"]).map(
            lambda x: f"{x:.4f}s"
        )
        display_df["wait_time"] = pd.Series(display_df["wait_time"]).map(
            lambda x: f"{x:.4f}s"
        )
        display_df["active_time"] = pd.Series(display_df["active_time"]).map(
            lambda x: f"{x:.4f}s"
        )
        st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    render_timing_gantt_page()
