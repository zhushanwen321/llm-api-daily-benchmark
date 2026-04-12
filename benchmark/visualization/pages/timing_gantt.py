"""Benchmark 耗时分析甘特图页面。

展示评测过程中各阶段的耗时分布，包括等待时间和执行时间。
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from benchmark.repository.file_repository import FileRepository


# 暗色主题配色 - 更鲜明的颜色
PHASE_COLORS: dict[str, str] = {
    "semaphore_wait": "#ff6b6b",
    "llm_request": "#4ecdc4",
    "score_calculation": "#96ceb4",
    "db_write": "#ffeaa7",
    "quality_signals": "#74b9ff",
}

# 暗色主题配色方案
DARK_BG = "#0e1117"
DARK_BG_SECONDARY = "#1e2129"
TEXT_COLOR = "#fafafa"
TEXT_COLOR_SECONDARY = "#b0b3b8"
GRID_COLOR = "#2d3139"

DATA_ROOT = Path("data")


def get_default_color() -> str:
    return "#95a5a6"


@st.cache_resource
def get_repository() -> FileRepository:
    """获取 FileRepository 实例（缓存）."""
    return FileRepository(DATA_ROOT)


def get_models(repo: FileRepository) -> list[str]:
    """获取所有模型名称."""
    runs = repo.get_runs()
    models = {run.get("model", "") for run in runs if run.get("model")}
    return sorted(models)


def create_gantt_chart(df: pd.DataFrame) -> go.Figure:
    """创建交互式甘特图（适配暗色主题）."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font_color=TEXT_COLOR,
        )
        return fig

    phases = df["phase_name"].unique()
    fig = go.Figure()

    for phase in phases:
        phase_df = df[df["phase_name"] == phase].copy()
        color = PHASE_COLORS.get(phase, get_default_color())

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

        fig.add_trace(
            go.Bar(
                x=phase_df["duration"].tolist(),
                y=phase_df["task_id"].tolist(),
                orientation="h",
                name=phase,
                marker_color=color,
                text=[f"{d:.2f}s" for d in phase_df["duration"].tolist()],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(size=9, color=DARK_BG),
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_texts,
                hoverlabel=dict(
                    bgcolor=DARK_BG_SECONDARY,
                    font_color=TEXT_COLOR,
                    font_size=11,
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        height=max(350, len(df) * 22 + 80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=TEXT_COLOR),
            bgcolor=DARK_BG_SECONDARY,
            bordercolor=GRID_COLOR,
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(
                text="Duration (seconds)",
                font=dict(size=11, color=TEXT_COLOR_SECONDARY),
            ),
            tickfont=dict(size=9, color=TEXT_COLOR_SECONDARY),
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            title=dict(text="Task ID", font=dict(size=11, color=TEXT_COLOR_SECONDARY)),
            tickfont=dict(size=9, color=TEXT_COLOR_SECONDARY),
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
        ),
        margin=dict(l=120, r=30, t=60, b=40),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        title=dict(
            text="Phase Timeline",
            font=dict(size=14, color=TEXT_COLOR, weight="bold"),
            x=0.5,
            xanchor="center",
        ),
    )

    return fig


def render_timing_gantt_page() -> None:
    """渲染耗时甘特图页面."""
    st.title("⏱️ Benchmark 耗时分析")

    repo = get_repository()

    # 筛选条件 - 3列布局
    col1, col2, col3 = st.columns(3)

    with col1:
        models = get_models(repo)
        model_options = ["All"] + models if models else ["All"]
        selected_model = st.selectbox("Model", model_options)

    with col2:
        run_id_input = st.text_input("Run ID", placeholder="e.g., run_2024_01_01")

    with col3:
        date_range = st.date_input(
            "Date Range",
            value=None,
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

    df = repo.get_timing_phases(
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
                    f'<div style="background-color:{DARK_BG_SECONDARY};padding:8px;border-radius:4px;">'
                    f'<span style="background-color:{color};width:14px;height:14px;'
                    f'display:inline-block;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
                    f'<span style="color:{TEXT_COLOR};font-size:12px;vertical-align:middle;">{phase}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with legend_cols[-1]:
            st.markdown(
                f'<div style="background-color:{DARK_BG_SECONDARY};padding:8px;border-radius:4px;">'
                f'<span style="background-color:{get_default_color()};width:14px;height:14px;'
                f'display:inline-block;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
                f'<span style="color:{TEXT_COLOR};font-size:12px;vertical-align:middle;">Other</span>'
                f"</div>",
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
