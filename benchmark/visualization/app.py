"""Streamlit 可视化界面主入口，支持多页面导航."""

import json
import os
from pathlib import Path
from typing import Callable

import streamlit as st

from benchmark.core.statistics import (
    calculate_confidence_interval,
    calculate_mean,
    calculate_std,
)
from benchmark.repository.file_repository import FileRepository
from benchmark.visualization.components import trends
from benchmark.visualization.components.scoring_details import render_scoring_breakdown
from benchmark.visualization.pages.timing_gantt import render_timing_gantt_page


def render_overview_page() -> None:
    """概览页面 - 展示评测结果统计和列表."""
    st.title("📊 评测概览")
    repo = get_repository()

    # 检查结果数据
    results = repo.get_results()
    if not results:
        st.info("No evaluation results yet. Run an evaluation to get started.")
        st.code(
            "uv run python -m benchmark evaluate --model my-provider/my-model --dimension reasoning"
        )
        return

    st.sidebar.header("Filters")

    models = get_models(repo)
    model_options = ["All"] + models
    selected_model = st.sidebar.selectbox("Model", model_options)

    dimensions = get_dimensions(repo)
    dim_options = ["All"] + dimensions
    selected_dimension = st.sidebar.selectbox("Dimension", dim_options)

    st.subheader("Evaluation Results")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Results", "Trends", "Probe & Stability", "Detail"]
    )

    results = get_results(repo, selected_model, selected_dimension)

    if not results:
        st.warning("No results match the selected filters.")
        return

    with tab1:
        st.subheader("Statistics Summary")

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

        _DROP_COLUMNS = {
            "prompt_tokens",
            "completion_tokens",
            "ttft_content",
            "reasoning_tokens",
        }
        _COLUMN_RENAME = {
            "result_id": "ID",
            "model": "Model",
            "dimension": "Dimension",
            "task_id": "Task",
            "final_score": "Score",
            "passed": "Passed",
            "execution_time": "Time",
            "tokens_per_second": "Token Speed",
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
                    display_row[new_key] = (
                        f"{value:.1f} tok/s" if value is not None else "-"
                    )
                else:
                    display_row[new_key] = value
            display_data.append(display_row)

        st.dataframe(display_data, width="stretch", hide_index=True)

    with tab2:
        st.subheader("Score Trends")

        time_range = st.selectbox(
            "Time Range", ["7 days", "30 days", "90 days", "All"], index=1
        )
        days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "All": 365}
        selected_days = days_map[time_range]

        selected_models = models if selected_model == "All" else [selected_model]
        selected_dimensions = (
            dimensions if selected_dimension == "All" else [selected_dimension]
        )

        if selected_model != "All" and selected_dimension != "All":
            trend_data = trends.get_trend_data(
                repo, selected_model, selected_dimension, selected_days
            )
            if trend_data["dates"]:
                fig = trends.create_trend_figure(
                    trend_data,
                    title=f"{selected_model} - {selected_dimension} Trend ({time_range})",
                )
                st.pyplot(fig)
            else:
                st.info("No trend data available for the selected model and dimension.")
        elif selected_model != "All" and selected_dimension == "All":
            for dimension in selected_dimensions:
                trend_data = trends.get_trend_data(
                    repo, selected_model, dimension, selected_days
                )
                if trend_data["dates"]:
                    fig = trends.create_trend_figure(
                        trend_data,
                        title=f"{selected_model} - {dimension} Trend ({time_range})",
                    )
                    st.pyplot(fig)
        elif selected_model == "All" and selected_dimension != "All":
            fig = trends.create_multi_model_trend(
                repo, selected_models, selected_dimension, selected_days
            )
            st.pyplot(fig)
        else:
            st.info("Please select a specific model or dimension to view trends.")

    with tab3:
        st.subheader("Probe & Stability")

        probe_model = (
            selected_model
            if selected_model != "All"
            else (models[0] if models else None)
        )

        st.markdown("#### Probe Runs")
        all_runs = repo.get_runs(dimension="probe")
        probe_runs = sorted(
            all_runs, key=lambda x: x.get("started_at", ""), reverse=True
        )[:20]

        if probe_runs:
            probe_data = []
            for r in probe_runs:
                run_id = r.get("run_id", "")
                run_results = repo.get_results(run_id=run_id)
                total = len(run_results)
                passed = sum(1 for res in run_results if res.get("passed"))
                avg_score = (
                    sum(res.get("final_score", 0) for res in run_results) / total
                    if total > 0
                    else None
                )
                started_at = r.get("started_at", "")
                probe_data.append(
                    {
                        "Time": started_at[:19] if started_at else "-",
                        "Model": r.get("model", ""),
                        "Status": r.get("status", ""),
                        "Passed": f"{passed}/{total}",
                        "Avg Score": f"{avg_score:.1f}" if avg_score else "-",
                    }
                )
            st.dataframe(probe_data, width="stretch", hide_index=True)
        else:
            st.info("No probe runs yet.")

        st.markdown("#### Stability Reports")
        stability_model = probe_model
        stability_reports = repo.get_stability_reports(
            model=stability_model if stability_model else None
        )
        stability_reports = sorted(
            stability_reports, key=lambda x: x.get("created_at", ""), reverse=True
        )[:20]

        if stability_reports:
            for report in stability_reports:
                status = report.get("overall_status", "")
                color = (
                    "green"
                    if status == "stable"
                    else ("red" if status == "degraded" else "orange")
                )
                created_at = report.get("created_at", "")
                st.markdown(
                    f"**[{report.get('model', '')}]** {created_at[:19] if created_at else '-'} — :{color}[{status}]"
                )
                st.caption(report.get("summary", "") or "")
        else:
            st.info("No stability reports yet.")

        st.markdown("#### Model Identity Clustering")
        cluster_reports = repo.get_cluster_reports(
            model=stability_model if stability_model else None
        )
        cluster_reports = sorted(
            cluster_reports, key=lambda x: x.get("created_at", ""), reverse=True
        )[:20]

        if cluster_reports:
            for report in cluster_reports:
                created_at = report.get("created_at", "")
                st.markdown(
                    f"**[{report.get('model', '')}]** {created_at[:19] if created_at else '-'} — "
                    f"Clusters: {report.get('n_clusters', 0)}, Noise: {report.get('n_noise', 0)}"
                )
                st.caption(report.get("summary", "") or "")
        else:
            st.info("No cluster reports yet.")

    with tab4:
        st.subheader("Result Detail")

        result_ids = [row["result_id"] for row in results]
        if (
            "selected_result_id" not in st.session_state
            or st.session_state["selected_result_id"] not in result_ids
        ):
            st.session_state["selected_result_id"] = result_ids[0]

        selected_result_id = st.session_state["selected_result_id"]

        if selected_result_id:
            detail = get_result_detail(repo, selected_result_id)
            if detail is None:
                st.error(f"Result {selected_result_id} not found")
                return

            selected_result_id = st.selectbox(
                "Evaluation Result Detail",
                options=result_ids,
                index=result_ids.index(st.session_state["selected_result_id"]),
                key="result_id_selector",
            )
            st.session_state["selected_result_id"] = selected_result_id
            if selected_result_id is None:
                st.error("No result selected")
                return
            detail = get_result_detail(repo, selected_result_id)

            if detail is None:
                st.error(f"Result {selected_result_id} not found")
                return

            # Task Info Section
            st.markdown("### 📋 Task Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.markdown(f"**Task ID:** `{detail['task_id']}`")
            with info_col2:
                metadata = detail.get("metadata") or {}
                st.markdown(f"**Dataset:** `{metadata.get('dataset', 'N/A')}`")
            with info_col3:
                st.markdown(f"**Dimension:** `{metadata.get('dimension', 'N/A')}`")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Metrics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Score", f"{detail['final_score']:.1f}")
                with metric_col2:
                    st.metric("Passed", "Yes" if detail["passed"] else "No")
                with metric_col3:
                    st.metric("Time", f"{detail['execution_time']:.2f}s")

                api_metrics = detail.get("api_metrics", {})

                if api_metrics:
                    token_col1, token_col2 = st.columns(2)
                    with token_col1:
                        st.metric(
                            "Token Speed",
                            f"{api_metrics.get('tokens_per_second', 0):.1f} tok/s",
                        )
                    with token_col2:
                        st.metric(
                            "Tokens",
                            f"{api_metrics.get('prompt_tokens', 0)} in / {api_metrics.get('completion_tokens', 0)} out",
                        )

                    extra_col1, extra_col2, extra_col3 = st.columns(3)
                    reasoning_tokens = api_metrics.get("reasoning_tokens", 0)
                    if reasoning_tokens > 0:
                        with extra_col1:
                            st.metric("Reasoning Tokens", f"{reasoning_tokens}")

                    ttft_content = api_metrics.get("ttft_content", 0.0)
                    if ttft_content > 0:
                        with extra_col2:
                            st.metric("TTFT-C", f"{ttft_content:.2f}s")

                    ttft = api_metrics.get("ttft", 0.0)
                    if ttft > 0:
                        with extra_col3:
                            st.metric("TTFT-R", f"{ttft:.2f}s")

            with col2:
                st.markdown("### 📝 Content")
                st.text_area(
                    "Prompt",
                    value=detail.get("task_content", "") or "",
                    height=150,
                    disabled=True,
                )

                expected_content = detail.get("expected_output", "") or ""
                if expected_content:
                    st.markdown("**Expected Answer**")
                    st.text_area(
                        "Expected",
                        value=expected_content,
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                    )

                answer_content = detail.get("model_answer", "") or ""
                st.markdown("**Model Answer**")
                st.text_area(
                    "Answer",
                    value=answer_content,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                )

            st.divider()

            # Scoring Section
            st.markdown("### 🎯 Scoring Details")

            think_content = detail.get("model_think", "") or ""
            api_metrics = detail.get("api_metrics", {})
            if api_metrics and api_metrics.get("reasoning_content"):
                think_content = api_metrics["reasoning_content"]

            if think_content:
                with st.expander("Thinking Process", expanded=False):
                    st.text_area(
                        "Thinking",
                        value=think_content,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            detail_raw = detail.get("details")
            if detail_raw:
                try:
                    parsed_details = (
                        json.loads(detail_raw)
                        if isinstance(detail_raw, str)
                        else detail_raw
                    )
                    render_scoring_breakdown(parsed_details)
                except (json.JSONDecodeError, TypeError):
                    with st.expander("Raw Score Details"):
                        st.write(detail_raw)
            else:
                st.info("No scoring details available")


DATA_ROOT = Path("data")


@st.cache_resource
def get_repository() -> FileRepository:
    """获取 FileRepository 实例（缓存）."""
    return FileRepository(DATA_ROOT)


def get_models(repo: FileRepository) -> list[str]:
    """获取所有已评测的模型名称."""
    runs = repo.get_runs()
    models = {run.get("model", "") for run in runs if run.get("model")}
    return sorted(models)


def get_dimensions(repo: FileRepository) -> list[str]:
    """获取所有已评测的维度名称."""
    runs = repo.get_runs()
    dimensions = {run.get("dimension", "") for run in runs if run.get("dimension")}
    return sorted(dimensions)


def get_results(
    repo: FileRepository, model: str | None, dimension: str | None
) -> list[dict]:
    """查询结果并返回字典列表."""
    results = repo.get_results(
        model=model if model and model != "All" else None,
        dimension=dimension if dimension and dimension != "All" else None,
    )

    # 补充 api_metrics 数据
    for result in results:
        result_id = result.get("result_id")
        if result_id:
            detail = repo.get_result_detail(result_id)
            if detail and detail.get("api_metrics"):
                api_metrics = detail["api_metrics"]
                result["tokens_per_second"] = api_metrics.get("tokens_per_second")
                result["ttft"] = api_metrics.get("ttft")
                result["ttft_content"] = api_metrics.get("ttft_content")
                result["reasoning_tokens"] = api_metrics.get("reasoning_tokens")
                result["prompt_tokens"] = api_metrics.get("prompt_tokens")
                result["completion_tokens"] = api_metrics.get("completion_tokens")

    return results


def get_result_detail(repo: FileRepository, result_id: str) -> dict | None:
    """查询单条结果的详情."""
    return repo.get_result_detail(result_id)


def main() -> None:
    st.set_page_config(page_title="LLM Benchmark", layout="wide")

    from benchmark.core.logging_config import setup_logging

    setup_logging()

    if (
        "scheduler_started" not in st.session_state
        and os.getenv("RUNNING_IN_DOCKER") != "true"
    ):
        from benchmark.core.scheduler import BenchmarkScheduler

        sched = BenchmarkScheduler()
        sched.start()
        st.session_state["scheduler_started"] = True

    st.sidebar.title("页面导航")
    page_names_to_funcs: dict[str, Callable] = {
        "📊 评测概览": render_overview_page,
        "⏱️ 耗时分析": render_timing_gantt_page,
    }
    selected_page = st.sidebar.radio("选择页面", list(page_names_to_funcs.keys()))

    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
