"""Streamlit 可视化界面。展示评测结果列表和单题详情."""

import json
import os
import sqlite3

import streamlit as st

from benchmark.core.statistics import calculate_confidence_interval, calculate_mean, calculate_std
from benchmark.visualization.components import trends

DB_PATH = "benchmark/data/results.db"


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    """获取 SQLite 连接（缓存）.

    先通过 Database 类触发迁移（确保 schema 最新），再返回原生连接.
    """
    from benchmark.models.database import Database

    Database().close()  # 触发 schema 迁移
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_models(conn: sqlite3.Connection) -> list[str]:
    """获取所有已评测的模型名称."""
    cursor = conn.execute("SELECT DISTINCT model FROM eval_runs")
    return [row["model"] for row in cursor.fetchall()]


def get_dimensions(conn: sqlite3.Connection) -> list[str]:
    """获取所有已评测的维度名称."""
    cursor = conn.execute("SELECT DISTINCT dimension FROM eval_runs")
    return [row["dimension"] for row in cursor.fetchall()]


def get_results(
    conn: sqlite3.Connection, model: str | None, dimension: str | None
) -> list[dict]:
    """查询结果并返回字典列表."""
    query = """
        SELECT
            r.result_id,
            e.model,
            e.dimension,
            r.task_id,
            r.final_score,
            r.passed,
            r.execution_time,
            m.tokens_per_second,
            m.ttft_content,
            m.reasoning_tokens,
            m.prompt_tokens,
            m.completion_tokens,
            r.created_at
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        LEFT JOIN api_call_metrics m ON r.result_id = m.result_id
        WHERE 1=1
    """
    params: list[str] = []

    if model and model != "All":
        query += " AND e.model = ?"
        params.append(model)
    if dimension and dimension != "All":
        query += " AND e.dimension = ?"
        params.append(dimension)

    query += " ORDER BY r.created_at DESC"
    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_result_detail(conn: sqlite3.Connection, result_id: str) -> dict | None:
    """查询单条结果的详情."""
    cursor = conn.execute(
        "SELECT * FROM eval_results WHERE result_id = ?",
        (result_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, row))


def main() -> None:
    st.set_page_config(page_title="LLM Benchmark", layout="wide")
    st.title("LLM Benchmark Results")

    # 初始化日志，确保 scheduler 等模块的 logger 输出到 stdout
    from benchmark.core.logging_config import setup_logging
    setup_logging()

    # 启动定时调度器（仅非 Docker 环境：Docker 中由 entrypoint 独立启动，避免重复）
    if "scheduler_started" not in st.session_state and os.getenv("RUNNING_IN_DOCKER") != "true":
        from benchmark.core.scheduler import BenchmarkScheduler

        sched = BenchmarkScheduler()
        sched.start()
        st.session_state["scheduler_started"] = True

    conn = get_connection()

    results_check = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()
    if results_check[0] == 0:
        st.info("No evaluation results yet. Run an evaluation to get started.")
        st.code(
            "uv run python -m benchmark evaluate --model my-provider/my-model --dimension reasoning"
        )
        return

    st.sidebar.header("Filters")

    models = get_models(conn)
    model_options = ["All"] + models
    selected_model = st.sidebar.selectbox("Model", model_options)

    dimensions = get_dimensions(conn)
    dim_options = ["All"] + dimensions
    selected_dimension = st.sidebar.selectbox("Dimension", dim_options)

    st.subheader("Evaluation Results")

    # 创建标签页结构
    tab1, tab2, tab3 = st.tabs(["Results", "Trends", "Detail"])

    results = get_results(conn, selected_model, selected_dimension)

    if not results:
        st.warning("No results match the selected filters.")
        return

    with tab1:
        st.subheader("Statistics Summary")

        # 计算统计数据
        scores = [row["final_score"] for row in results]
        if len(scores) >= 2:
            mean_score = calculate_mean(scores)
            std_score = calculate_std(scores)
            ci_lower, ci_upper = calculate_confidence_interval(scores)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Mean", f"{mean_score:.1f}")
            with col2:
                st.metric("Std Dev", f"±{std_score:.2f}")
            with col3:
                st.metric("95% CI", f"[{ci_lower:.1f}, {ci_upper:.1f}]")
            with col4:
                st.metric("Max", f"{max(scores):.1f}")
            with col5:
                st.metric("Min", f"{min(scores):.1f}")

        st.subheader("Results Table")

        _DROP_COLUMNS = {"prompt_tokens", "completion_tokens", "ttft_content", "reasoning_tokens"}
        _COLUMN_RENAME = {
            "result_id": "ID", "model": "Model", "dimension": "Dimension",
            "task_id": "Task", "final_score": "Score", "passed": "Passed",
            "execution_time": "Time", "tokens_per_second": "Token Speed",
            "created_at": "Date",
        }
        display_data = []
        for row in results:
            display_row = {}
            for key, value in row.items():
                if key in _DROP_COLUMNS:
                    continue
                new_key = _COLUMN_RENAME.get(key, key)
                if key == "passed":
                    display_row[new_key] = "Yes" if value else "No"
                elif key == "execution_time":
                    display_row[new_key] = f"{value:.2f}s" if value is not None else "-"
                elif key == "tokens_per_second":
                    display_row[new_key] = f"{value:.1f} tok/s" if value is not None else "-"
                else:
                    display_row[new_key] = value
            display_data.append(display_row)

        st.dataframe(display_data, width='stretch', hide_index=True)

    with tab2:
        st.subheader("Score Trends")

        # 添加时间范围选择器
        time_range = st.selectbox("Time Range", ["7 days", "30 days", "90 days", "All"], index=1)
        days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "All": 365}
        selected_days = days_map[time_range]

        # 获取当前选择的模型和维度
        selected_models = models if selected_model == "All" else [selected_model]
        selected_dimensions = dimensions if selected_dimension == "All" else [selected_dimension]

        # 展示趋势图
        if selected_model != "All" and selected_dimension != "All":
            # 单模型单维度的趋势图
            trend_data = trends.get_trend_data(conn, selected_model, selected_dimension, selected_days)
            if trend_data["dates"]:
                fig = trends.create_trend_figure(
                    trend_data,
                    title=f"{selected_model} - {selected_dimension} Trend ({time_range})"
                )
                st.pyplot(fig)
            else:
                st.info("No trend data available for the selected model and dimension.")
        elif selected_model != "All" and selected_dimension == "All":
            # 单模型多维度的趋势图
            for dimension in selected_dimensions:
                trend_data = trends.get_trend_data(conn, selected_model, dimension, selected_days)
                if trend_data["dates"]:
                    fig = trends.create_trend_figure(
                        trend_data,
                        title=f"{selected_model} - {dimension} Trend ({time_range})"
                    )
                    st.pyplot(fig)
        elif selected_model == "All" and selected_dimension != "All":
            # 多模型单维度的对比趋势图
            fig = trends.create_multi_model_trend(conn, selected_models, selected_dimension, selected_days)
            st.pyplot(fig)
        else:
            st.info("Please select a specific model or dimension to view trends.")

    with tab3:
        st.subheader("Result Detail")

        result_ids = [row["result_id"] for row in results]
        if (
            "selected_result_id" not in st.session_state
            or st.session_state["selected_result_id"] not in result_ids
        ):
            st.session_state["selected_result_id"] = result_ids[0]

        selected_result_id = st.session_state["selected_result_id"]

        if selected_result_id:
            detail = get_result_detail(conn, selected_result_id)
            if detail:
                selected_result_id = st.selectbox(
                    "Evaluation Result Detail",
                    options=result_ids,
                    index=result_ids.index(st.session_state["selected_result_id"]),
                    key="result_id_selector",
                )
                st.session_state["selected_result_id"] = selected_result_id
                detail = get_result_detail(conn, selected_result_id)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", f"{detail['final_score']:.1f}")
                    st.metric("Passed", "Yes" if detail["passed"] else "No")
                    st.metric("Execution Time", f"{detail['execution_time']:.2f}s")

                    # 查询 token 指标
                    metrics_row = conn.execute(
                        "SELECT * FROM api_call_metrics WHERE result_id = ?",
                        (selected_result_id,),
                    ).fetchone()
                    if metrics_row:
                        st.metric(
                            "Token Speed",
                            f"{metrics_row['tokens_per_second']:.1f} tok/s",
                        )
                        st.metric(
                            "Tokens",
                            f"{metrics_row['prompt_tokens']} in / {metrics_row['completion_tokens']} out",
                        )
                        reasoning_tokens = (
                            metrics_row["reasoning_tokens"]
                            if "reasoning_tokens" in metrics_row.keys()
                            else 0
                        )
                        if reasoning_tokens > 0:
                            st.metric(
                                "Reasoning Tokens",
                                f"{reasoning_tokens}",
                            )
                        ttft_content = (
                            metrics_row["ttft_content"]
                            if "ttft_content" in metrics_row.keys()
                            else 0.0
                        )
                        if ttft_content > 0:
                            st.metric(
                                "TTFT-C",
                                f"{ttft_content:.2f}s",
                            )

                with col2:
                    st.text_area(
                        "Prompt",
                        value=detail.get("task_content", "") or "",
                        height=200,
                        disabled=True,
                    )

                    # 展示思考过程（折叠），优先从 metrics 读取 API 原生 reasoning_content
                    think_content = detail.get("model_think", "") or ""
                    if metrics_row and metrics_row["reasoning_content"]:
                        think_content = metrics_row["reasoning_content"]
                    if think_content:
                        with st.expander("Thinking", expanded=False):
                            st.text_area(
                                "Thinking Process",
                                value=think_content,
                                height=200,
                                disabled=True,
                                label_visibility="collapsed",
                            )

                    # 展示最终答案
                    answer_content = detail.get("model_answer", "") or ""
                    st.text_area(
                        "Answer",
                        value=answer_content,
                        height=300,
                        disabled=True,
                    )

                detail_raw = detail.get("details")
                if detail_raw:
                    with st.expander("Score Details"):
                        try:
                            parsed = (
                                json.loads(detail_raw)
                                if isinstance(detail_raw, str)
                                else detail_raw
                            )
                            st.json(parsed)
                        except (json.JSONDecodeError, TypeError):
                            st.write(detail_raw)


if __name__ == "__main__":
    main()
