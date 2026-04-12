"""稳定性分析器：对质量信号做时序分析，输出 StabilityReport。

分析流程：
1. 加载当前 run 的 quality_signals + eval_results
2. 加载历史基线（最近 N 天）
3. z-score 异常检测（|z| > 2）
4. CUSUM 变化点检测
5. answer_entropy 计算
6. Welch's t-test + Bonferroni 校正
7. 判定 overall_status
8. 保存 stability_report
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from benchmark.analysis.models import AnomalyDetail, ChangePoint, StabilityReport

if TYPE_CHECKING:
    from benchmark.repository import FileRepository

# z-score 异常阈值
_ZSCORE_THRESHOLD = 2.0

# Bonferroni 校正后的 alpha（10 个检验：9 数值信号 + score）
_BONFERRONI_ALPHA = 0.05 / 10

# 参与 CUSUM 检测的信号名
_CUSUM_SIGNALS = ("tps_zscore", "ttft_zscore", "score", "thinking_ratio")

# 参与 z-score 异常检测的数值型信号
_NUMERIC_SIGNALS = (
    "format_compliance",
    "repetition_ratio",
    "garbled_text_ratio",
    "language_consistency",
    "output_length_zscore",
    "thinking_ratio",
    "token_efficiency_zscore",
    "tps_zscore",
    "ttft_zscore",
)


class StabilityAnalyzer:
    """对质量信号做时序分析并输出稳定性报告。"""

    def __init__(self, repo: FileRepository, history_days: int = 7) -> None:
        self._repo = repo
        self._history_days = history_days

    # ── 公共 API ──

    async def run(self, model: str, run_id: str, dimension: str) -> StabilityReport:
        """执行完整的稳定性分析流程。"""
        # 1. 加载当前 run 的 quality_signals
        current_signals = await self._repo.aget_quality_signals_for_run(run_id)
        # 2. 加载当前 run 的 eval_results（获取 final_score）
        current_scores = await self._get_current_scores(run_id)
        # 3. 加载历史基线
        history_signals = await self._repo.aget_quality_signals_history(
            model, self._history_days
        )
        history_scores = await self._get_history_scores(model, dimension)

        # 4. z-score 异常检测
        anomalies = self._detect_anomalies(current_signals, history_signals)

        # 5. answer_entropy
        answer_entropy = self._calc_answer_entropy(current_scores)

        # 6. CUSUM 变化点检测
        change_points = self._run_cusum_detection(
            current_signals, history_signals, history_scores, current_scores
        )

        # 7. Welch's t-test
        stat_tests = self._run_statistical_tests(
            current_signals, current_scores, history_signals, history_scores
        )

        # 8. 判定 overall_status
        overall_status = self._determine_status(anomalies, change_points, stat_tests)

        # 9. 生成 summary
        summary = self._generate_summary(
            overall_status, anomalies, change_points, stat_tests
        )

        report = StabilityReport(
            model=model,
            run_id=run_id,
            overall_status=overall_status,
            anomalies=anomalies,
            change_points=change_points,
            stat_tests=stat_tests,
            summary=summary,
            created_at=datetime.now(),
        )

        # 10. 保存到数据库
        await self._repo.asave_stability_report(report)

        return report

    # ── 数据加载 ──

    async def _get_current_scores(self, run_id: str) -> list[float]:
        """获取当前 run 的 final_score 列表。"""
        results = await self._repo.aget_results(run_id=run_id)
        return [r["final_score"] for r in results if r.get("final_score") is not None]

    async def _get_history_scores(self, model: str, dimension: str) -> list[dict]:
        """获取历史 final_score，包含 run_id、final_score、created_at。"""
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=self._history_days)
        results = await self._repo.aget_results(model=model, dimension=dimension)
        history = []
        for r in results:
            # 仅包含已完成的 run
            if r.get("status") != "completed":
                continue
            created_at = r.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            if created_at and created_at >= cutoff:
                history.append(
                    {
                        "final_score": r.get("final_score"),
                        "created_at": created_at,
                    }
                )
        return sorted(history, key=lambda x: x["created_at"])

    # ── z-score 异常检测 ──

    def _detect_anomalies(
        self,
        current_signals: list[dict],
        history_signals: list[dict],
    ) -> list[AnomalyDetail]:
        """逐信号计算 z-score，|z| > _ZSCORE_THRESHOLD 标记为异常。"""
        if not current_signals or len(history_signals) < 2:
            return []

        anomalies: list[AnomalyDetail] = []
        for signal_name in _NUMERIC_SIGNALS:
            current_values = [
                s[signal_name]
                for s in current_signals
                if s.get(signal_name) is not None
            ]
            history_values = [
                s[signal_name]
                for s in history_signals
                if s.get(signal_name) is not None
            ]
            if not current_values or len(history_values) < 2:
                continue

            current_mean = statistics.mean(current_values)
            baseline_mean = statistics.mean(history_values)
            baseline_std = statistics.pstdev(history_values)
            if baseline_std == 0:
                continue

            z = (current_mean - baseline_mean) / baseline_std
            if abs(z) > _ZSCORE_THRESHOLD:
                anomalies.append(
                    AnomalyDetail(
                        signal_name=signal_name,
                        current_value=round(current_mean, 4),
                        baseline_mean=round(baseline_mean, 4),
                        baseline_std=round(baseline_std, 4),
                        z_score=round(z, 4),
                    )
                )
        return anomalies

    # ── CUSUM 变化点检测 ──

    def _run_cusum_detection(
        self,
        current_signals: list[dict],
        history_signals: list[dict],
        history_scores: list[dict],
        current_scores: list[float],
    ) -> list[ChangePoint]:
        """对 TPS、TTFT、score、thinking_ratio 做 CUSUM 变化点检测。"""
        change_points: list[ChangePoint] = []

        # quality_signals 中的信号：使用历史基线 + 当前 run 的时序
        for signal_name in ("tps_zscore", "ttft_zscore", "thinking_ratio"):
            values, timestamps = self._build_timeseries(
                signal_name, history_signals, current_signals
            )
            if len(values) < 5:
                continue
            # 仅用历史基线计算 mean/std，避免当前异常值拉偏
            hist_vals = [
                s[signal_name]
                for s in history_signals
                if s.get(signal_name) is not None
            ]
            if len(hist_vals) < 2:
                continue
            baseline_mean = statistics.mean(hist_vals)
            baseline_std = statistics.pstdev(hist_vals)
            if baseline_std == 0:
                continue
            change_points.extend(
                self._cusum_detect(
                    values,
                    signal_name,
                    timestamps,
                    baseline_mean=baseline_mean,
                    baseline_std=baseline_std,
                )
            )

        # score 时序
        score_values, score_timestamps = self._build_score_timeseries(
            history_scores, current_scores
        )
        if len(score_values) < 5:
            pass
        else:
            hist_score_vals = [
                h["final_score"]
                for h in history_scores
                if h.get("final_score") is not None
            ]
            if len(hist_score_vals) >= 2:
                score_mean = statistics.mean(hist_score_vals)
                score_std = statistics.pstdev(hist_score_vals)
                if score_std > 0:
                    change_points.extend(
                        self._cusum_detect(
                            score_values,
                            "score",
                            score_timestamps,
                            baseline_mean=score_mean,
                            baseline_std=score_std,
                        )
                    )

        return change_points

    def _build_timeseries(
        self,
        signal_name: str,
        history_signals: list[dict],
        current_signals: list[dict],
    ) -> tuple[list[float], list[datetime]]:
        """合并历史和当前的信号值，构建时序。按 created_at 排序。"""
        all_signals = history_signals + current_signals
        if not all_signals:
            return [], []

        values = []
        timestamps = []
        for s in all_signals:
            val = s.get(signal_name)
            if val is None:
                continue
            ts = s.get("created_at")
            if ts is None:
                continue
            values.append(float(val))
            timestamps.append(self._parse_timestamp(ts))

        # 按时间排序
        paired = sorted(zip(timestamps, values), key=lambda x: x[0])
        if not paired:
            return [], []
        timestamps = [p[0] for p in paired]
        values = [p[1] for p in paired]
        return values, timestamps

    def _build_score_timeseries(
        self,
        history_scores: list[dict],
        current_scores: list[float],
    ) -> tuple[list[float], list[datetime]]:
        """构建 score 时序。"""
        values: list[float] = []
        timestamps: list[datetime] = []

        for row in history_scores:
            val = row.get("final_score")
            ts = row.get("created_at")
            if val is not None and ts is not None:
                values.append(float(val))
                timestamps.append(self._parse_timestamp(ts))

        # 追加当前 run 的分数
        if current_scores:
            now = datetime.now()
            for s in current_scores:
                values.append(float(s))
                timestamps.append(now)

        return values, timestamps

    def _cusum_detect(
        self,
        values: list[float],
        signal_name: str,
        timestamps: list[datetime],
        baseline_mean: float | None = None,
        baseline_std: float | None = None,
    ) -> list[ChangePoint]:
        """CUSUM 变化点检测。"""
        if len(values) < 5:
            return []

        mean = baseline_mean if baseline_mean is not None else statistics.mean(values)
        std = baseline_std if baseline_std is not None else statistics.pstdev(values)
        if std == 0:
            return []

        k = 0.5 * std  # slack parameter
        h = 5.0 * std  # threshold
        s_high = 0.0
        s_low = 0.0
        change_points: list[ChangePoint] = []

        for i, x in enumerate(values):
            s_high = max(0, s_high + (x - mean - k))
            s_low = min(0, s_low + (x - mean + k))

            if s_high > h:
                magnitude = abs(x - mean) / std
                change_points.append(
                    ChangePoint(
                        signal_name=signal_name,
                        detected_at=timestamps[i],
                        direction="increase",
                        magnitude=round(magnitude, 2),
                    )
                )
                s_high = 0.0  # reset
            elif s_low < -h:
                magnitude = abs(x - mean) / std
                change_points.append(
                    ChangePoint(
                        signal_name=signal_name,
                        detected_at=timestamps[i],
                        direction="decrease",
                        magnitude=round(magnitude, 2),
                    )
                )
                s_low = 0.0  # reset

        return change_points

    @staticmethod
    def _parse_timestamp(ts: str | datetime) -> datetime:
        """将字符串或 datetime 转为 datetime。"""
        if isinstance(ts, datetime):
            return ts
        return datetime.fromisoformat(ts)

    # ── answer_entropy ──

    @staticmethod
    def _calc_answer_entropy(scores: list[float]) -> float:
        """计算 passed 分布的熵（基于 final_score 二值化为 passed/not passed）。"""
        if not scores:
            return 0.0
        # 将分数二值化：>= 0.5 视为 passed
        passed = [1 if s >= 0.5 else 0 for s in scores]
        counts = Counter(passed)
        total = len(passed)
        entropy = 0.0
        for count in counts.values():
            if count == 0:
                continue
            p = count / total
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    # ── Welch's t-test ──

    def _run_statistical_tests(
        self,
        current_signals: list[dict],
        current_scores: list[float],
        history_signals: list[dict],
        history_scores: list[dict],
    ) -> list[dict]:
        """对 score + 每个数值信号做 Welch's t-test，使用 Bonferroni 校正。"""
        results: list[dict] = []

        # score t-test
        if len(current_scores) >= 2 and len(history_scores) >= 2:
            hist_vals = [h["final_score"] for h in history_scores]
            t_stat, p_value = self._welch_ttest(current_scores, hist_vals)
            effect = self._cohens_d(current_scores, hist_vals)
            results.append(
                {
                    "signal": "score",
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 6),
                    "effect_size": round(effect, 4),
                    "significant": p_value < _BONFERRONI_ALPHA,
                }
            )

        # 各信号 t-test
        for signal_name in _NUMERIC_SIGNALS:
            cur_vals = [
                s[signal_name]
                for s in current_signals
                if s.get(signal_name) is not None
            ]
            hist_vals = [
                s[signal_name]
                for s in history_signals
                if s.get(signal_name) is not None
            ]
            if len(cur_vals) < 2 or len(hist_vals) < 2:
                continue

            t_stat, p_value = self._welch_ttest(cur_vals, hist_vals)
            effect = self._cohens_d(cur_vals, hist_vals)
            results.append(
                {
                    "signal": signal_name,
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 6),
                    "effect_size": round(effect, 4),
                    "significant": p_value < _BONFERRONI_ALPHA,
                }
            )

        return results

    @staticmethod
    def _welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
        """Welch's t-test，返回 (t_statistic, p_value)。

        使用项目中 advanced_statistics.py 的实现（基于 t 分布 CDF）。
        """
        if len(a) < 2 or len(b) < 2:
            return 0.0, 1.0

        from benchmark.core.advanced_statistics import _ttest_ind_p_value

        n_a, n_b = len(a), len(b)
        m_a, m_b = statistics.mean(a), statistics.mean(b)
        v_a = statistics.variance(a)
        v_b = statistics.variance(b)

        p_value = _ttest_ind_p_value(m_a, m_b, v_a, v_b, n_a, n_b)

        var_sum = v_a / n_a + v_b / n_b
        if var_sum == 0:
            return 0.0, 1.0
        t_stat = (m_a - m_b) / math.sqrt(var_sum)

        return t_stat, max(0.0, min(1.0, p_value))

    @staticmethod
    def _cohens_d(a: list[float], b: list[float]) -> float:
        """Cohen's d 效应量。"""
        m_a, m_b = statistics.mean(a), statistics.mean(b)
        v_a, v_b = statistics.variance(a), statistics.variance(b)
        pooled_std = math.sqrt((v_a + v_b) / 2.0)
        if pooled_std == 0:
            return 0.0
        return float((m_a - m_b) / pooled_std)

    # ── Status 判定 ──

    @staticmethod
    def _determine_status(
        anomalies: list[AnomalyDetail],
        change_points: list[ChangePoint],
        stat_tests: list[dict],
    ) -> Literal["stable", "degraded", "suspicious"]:
        """判定 overall_status。"""
        # degraded 条件
        score_significant = any(
            t["signal"] == "score"
            and t["p_value"] < _BONFERRONI_ALPHA
            and t["effect_size"] < -0.5
            for t in stat_tests
        )
        many_anomalies = len(anomalies) >= 3
        format_degraded = any(
            a.signal_name == "format_compliance" and a.current_value < 0.5
            for a in anomalies
        )
        repetition_degraded = any(
            a.signal_name == "repetition_ratio" and a.current_value > 0.3
            for a in anomalies
        )

        if (
            score_significant
            or many_anomalies
            or format_degraded
            or repetition_degraded
        ):
            return "degraded"

        # suspicious 条件
        perf_change = any(
            cp.signal_name in ("tps_zscore", "ttft_zscore") for cp in change_points
        )
        thinking_change = any(
            t["signal"] == "thinking_ratio" and t["p_value"] < _BONFERRONI_ALPHA
            for t in stat_tests
        )

        if perf_change or thinking_change or len(anomalies) > 0:
            return "suspicious"

        return "stable"

    # ── Summary 生成 ──

    @staticmethod
    def _generate_summary(
        status: str,
        anomalies: list[AnomalyDetail],
        change_points: list[ChangePoint],
        stat_tests: list[dict],
    ) -> str:
        """生成人类可读的摘要。"""
        if status == "stable":
            return "No significant changes detected."
        parts: list[str] = []
        if anomalies:
            names = [a.signal_name for a in anomalies]
            parts.append(f"Anomalies: {', '.join(names)}")
        if change_points:
            cps = [f"{cp.signal_name} {cp.direction}" for cp in change_points]
            parts.append(f"Change points: {', '.join(cps)}")
        sig_tests = [t for t in stat_tests if t.get("significant")]
        if sig_tests:
            parts.append(f"Significant: {', '.join(t['signal'] for t in sig_tests)}")
        return "; ".join(parts)
