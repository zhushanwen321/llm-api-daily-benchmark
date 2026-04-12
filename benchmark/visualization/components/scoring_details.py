"""改进的评分详情展示函数."""

import json
import streamlit as st


def render_scoring_breakdown(details: dict) -> None:
    """可视化展示评分计算过程."""
    if not details:
        st.info("No scoring details available")
        return

    # 展示总分
    total_score = details.get("score", 0)
    final_score = details.get("final_score", total_score)
    passed = details.get("passed", False)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Score", f"{final_score:.1f}")
    with col2:
        st.metric("Status", "Passed" if passed else "Failed")

    # 如果有复合评分，展示各维度得分
    composite_scores = details.get("composite.scores", {})
    composite_weights = details.get("composite.weights", {})

    if composite_scores and composite_weights:
        st.markdown("---")
        st.markdown("**Scoring Breakdown**")

        # 计算加权得分并展示
        for metric_name, score in composite_scores.items():
            weight = composite_weights.get(metric_name, 0)
            weighted_score = score * weight

            # 创建进度条展示原始得分
            progress_col, score_col, weight_col, contrib_col = st.columns([3, 1, 1, 1])

            with progress_col:
                st.progress(score / 100, text=metric_name.replace("_", " ").title())
            with score_col:
                st.caption(f"{score:.1f}")
            with weight_col:
                st.caption(f"×{weight:.0%}")
            with contrib_col:
                st.caption(f"={weighted_score:.1f}")

        # 展示加权总分计算
        total_weighted = sum(
            score * composite_weights.get(metric, 0)
            for metric, score in composite_scores.items()
        )
        st.markdown(
            f"**Total:** {' + '.join([f'{composite_scores[m]:.1f}×{composite_weights[m]:.0%}' for m in composite_scores])} = **{total_weighted:.1f}**"
        )

    # 展示详细说明
    reasoning = details.get("reasoning", "")
    if reasoning:
        st.markdown("---")
        st.markdown("**Reasoning**")
        st.write(reasoning)

    # 展示原始JSON（折叠）
    with st.expander("Raw JSON"):
        st.json(details)
