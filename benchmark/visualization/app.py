"""Streamlit 可视化界面。展示评测结果列表和单题详情."""

import json
import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = "benchmark/data/results.db"


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    """获取 SQLite 连接（缓存）."""
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


def get_results_df(
    conn: sqlite3.Connection, model: str | None, dimension: str | None
) -> pd.DataFrame:
    """查询结果并返回 DataFrame."""
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
    return pd.read_sql_query(query, conn, params=params)


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

    conn = get_connection()

    results_check = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()
    if results_check[0] == 0:
        st.info("No evaluation results yet. Run an evaluation to get started.")
        st.code(
            "python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 5"
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
    df = get_results_df(conn, selected_model, selected_dimension)

    if df.empty:
        st.warning("No results match the selected filters.")
        return

    display_df = df.copy()
    display_df["passed"] = display_df["passed"].map(lambda x: "Yes" if x else "No")
    display_df["execution_time"] = (
        display_df["execution_time"].round(2).astype(str) + "s"
    )
    display_df["tokens_per_second"] = display_df["tokens_per_second"].apply(
        lambda x: f"{x:.1f} tok/s" if pd.notna(x) else "-"
    )
    display_df = display_df.drop(columns=["prompt_tokens", "completion_tokens"])
    display_df.columns = [
        "ID",
        "Model",
        "Dimension",
        "Task",
        "Score",
        "Passed",
        "Time",
        "Token Speed",
        "Date",
    ]

    st.dataframe(display_df, width="stretch", hide_index=True)

    st.subheader("Result Detail")
    result_ids = df["result_id"].tolist()
    selected_result = st.selectbox("Select a result to view details", result_ids)

    if selected_result:
        detail = get_result_detail(conn, selected_result)
        if detail:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score", f"{detail['final_score']:.1f}")
                st.metric("Passed", "Yes" if detail["passed"] else "No")
                st.metric("Execution Time", f"{detail['execution_time']:.2f}s")

                # 查询 token 指标
                metrics_row = conn.execute(
                    "SELECT * FROM api_call_metrics WHERE result_id = ?",
                    (selected_result,),
                ).fetchone()
                if metrics_row:
                    cols = [d[0] for d in conn.execute(
                        "SELECT * FROM api_call_metrics WHERE result_id = ?",
                        (selected_result,),
                    ).description]
                    metrics = dict(zip(cols, metrics_row))
                    st.metric(
                        "Token Speed",
                        f"{metrics['tokens_per_second']:.1f} tok/s",
                    )
                    st.metric(
                        "Tokens",
                        f"{metrics['prompt_tokens']} in / {metrics['completion_tokens']} out",
                    )

            with col2:
                st.text_area(
                    "Prompt",
                    value=detail.get("task_content", "") or "",
                    height=200,
                    disabled=True,
                )
                st.text_area(
                    "Output",
                    value=detail.get("model_output", "") or "",
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
