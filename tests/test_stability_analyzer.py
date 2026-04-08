"""测试 StabilityAnalyzer。"""

import asyncio
import statistics
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from benchmark.analysis.models import AnomalyDetail, ChangePoint, StabilityReport
from benchmark.analysis.stability_analyzer import (
    StabilityAnalyzer,
    _BONFERRONI_ALPHA,
)
from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun


# ── helpers ──


def _test_db() -> Database:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return Database(db_path=Path(tmp.name))


def _insert_run_and_result(
    db: Database,
    run_id: str,
    model: str = "test/model",
    dimension: str = "reasoning",
    scores: list[float] | None = None,
    days_ago: int = 0,
) -> list[str]:
    """插入一个 run 和多条 eval_result，返回 result_id 列表。"""
    now = datetime.now()
    run = EvalRun(
        run_id=run_id,
        model=model,
        dimension=dimension,
        dataset="gsm8k",
        started_at=now - timedelta(days=days_ago),
        finished_at=now - timedelta(days=days_ago),
        status="completed",
    )
    db.create_run(run)
    result_ids = []
    for i, score in enumerate(scores or [0.5]):
        rid = f"{run_id}-res-{i}"
        result = EvalResult(
            result_id=rid,
            run_id=run_id,
            task_id=f"t{i}",
            task_content="test",
            model_output="output",
            functional_score=score,
            quality_score=score,
            final_score=score,
            passed=score >= 0.5,
            execution_time=1.0,
            created_at=now - timedelta(days=days_ago, hours=i),
        )
        db.save_result(result)
        result_ids.append(rid)
    return result_ids


def _insert_signals(
    db: Database,
    result_ids: list[str],
    signals: dict | None = None,
) -> None:
    """为给定的 result_ids 插入 quality_signals。"""
    defaults = {
        "format_compliance": 0.95,
        "repetition_ratio": 0.05,
        "garbled_text_ratio": 0.0,
        "refusal_detected": 0,
        "language_consistency": 1.0,
        "output_length_zscore": 0.1,
        "thinking_ratio": 0.3,
        "empty_reasoning": 0,
        "truncated": 0,
        "token_efficiency_zscore": 0.0,
        "tps_zscore": 0.5,
        "ttft_zscore": -0.2,
        "answer_entropy": 0.0,
        "raw_output_length": 500,
    }
    merged = {**defaults, **(signals or {})}
    for rid in result_ids:
        db._save_quality_signals({"result_id": rid, **merged})


def _insert_history_data(
    db: Database,
    model: str = "test/model",
    n_days: int = 7,
    results_per_day: int = 3,
    signals_override: dict | None = None,
) -> None:
    """插入 N 天的历史数据。每个 result 的信号有微小差异以避免 std=0。"""
    for day in range(n_days):
        run_id = f"hist-day-{day}"
        scores = [0.7 + 0.01 * i for i in range(results_per_day)]
        rids = _insert_run_and_result(
            db, run_id, model, scores=scores, days_ago=day + 1
        )
        # 为每个 result 插入略有差异的信号
        for idx, rid in enumerate(rids):
            per_result = {
                "tps_zscore": 0.5 + 0.01 * idx,
                "ttft_zscore": -0.2 + 0.01 * idx,
                "thinking_ratio": 0.3 + 0.01 * idx,
                "output_length_zscore": 0.1 + 0.01 * idx,
                "token_efficiency_zscore": 0.0 + 0.01 * idx,
            }
            merged = {**(signals_override or {}), **per_result}
            _insert_signals(db, [rid], merged)


# ── CUSUM 检测 ──


class TestCusumDetect:
    def test_no_change_points_stable_data(self):
        """稳定数据不应检测到变化点。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.1]
        timestamps = [datetime(2026, 1, 1 + i) for i in range(10)]
        result = analyzer._cusum_detect(values, "test_signal", timestamps)
        assert result == []
        db.close()

    def test_detects_increase_change_point(self):
        """数据突然上升应检测到 increase 变化点。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        # 稳定基线 + 大幅跃升：需要持续偏差累积超过 5*std
        # 使用低方差基线让阈值更容易触发
        base = [10.0 + 0.01 * i for i in range(10)]
        # 跃升：每个值远高于 mean，累积 s_high 快速超过 h=5*std
        spike = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
        values = base + spike
        timestamps = [datetime(2026, 1, 1 + i) for i in range(len(values))]
        result = analyzer._cusum_detect(values, "tps_zscore", timestamps)
        assert len(result) >= 1
        assert result[0].direction == "increase"
        db.close()

    def test_detects_decrease_change_point(self):
        """数据突然下降应检测到 decrease 变化点。

        需要让 mean 接近基线，这样 drop 才能累积 s_low 超过阈值。
        使用：大量基线数据 + 少量骤降。
        """
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        # 20 个基线值 near 10 → mean ≈ 10, std ≈ 小值
        base = [10.0 + 0.01 * i for i in range(20)]
        # 10 个骤降值
        drop = [-20.0, -25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -55.0, -60.0, -65.0]
        values = base + drop
        timestamps = [datetime(2026, 1, 1 + i) for i in range(len(values))]
        result = analyzer._cusum_detect(values, "tps_zscore", timestamps)
        assert len(result) >= 1
        assert result[0].direction == "decrease"
        db.close()

    def test_insufficient_data_returns_empty(self):
        """少于 5 个数据点时返回空列表。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        values = [1.0, 2.0, 3.0, 4.0]
        timestamps = [datetime(2026, 1, 1 + i) for i in range(4)]
        result = analyzer._cusum_detect(values, "test", timestamps)
        assert result == []
        db.close()

    def test_zero_std_returns_empty(self):
        """所有值相同时 std=0，应返回空列表。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        values = [5.0] * 10
        timestamps = [datetime(2026, 1, 1 + i) for i in range(10)]
        result = analyzer._cusum_detect(values, "test", timestamps)
        assert result == []
        db.close()

    def test_reset_after_detection(self):
        """检测到变化点后应重置 CUSUM 累积量。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        # 大量基线 + 短上升 + 短下降
        # 基线 20 个 near 0，让 mean ≈ 0, std ≈ 小值
        base = [0.0 + 0.01 * i for i in range(20)]
        spike_up = [20.0, 30.0, 40.0, 50.0, 60.0]
        base2 = [0.0 + 0.01 * i for i in range(10)]
        spike_down = [-20.0, -30.0, -40.0, -50.0, -60.0]
        values = base + spike_up + base2 + spike_down
        timestamps = [datetime(2026, 1, 1 + i // 24, i % 24) for i in range(len(values))]
        result = analyzer._cusum_detect(values, "test", timestamps)
        # 应检测到 increase 和 decrease
        directions = [cp.direction for cp in result]
        assert "increase" in directions
        assert "decrease" in directions
        db.close()


# ── Welch's t-test ──


class TestWelchTtest:
    def test_significant_difference(self):
        """两组均值差异大时应显著。"""
        a = [10.0 + 0.1 * i for i in range(20)]
        b = [20.0 + 0.1 * i for i in range(20)]
        t_stat, p_value = StabilityAnalyzer._welch_ttest(a, b)
        assert p_value < 0.001
        assert t_stat < 0  # a < b

    def test_no_significant_difference(self):
        """两组均值接近时不应显著。"""
        db = _test_db()
        a = [10.0 + 0.1 * i for i in range(20)]
        b = [10.0 + 0.1 * i + 0.01 for i in range(20)]
        t_stat, p_value = StabilityAnalyzer._welch_ttest(a, b)
        assert p_value > 0.01
        db.close()

    def test_insufficient_data(self):
        """数据不足时返回 (0.0, 1.0)。"""
        t_stat, p_value = StabilityAnalyzer._welch_ttest([1.0], [2.0, 3.0])
        assert t_stat == 0.0
        assert p_value == 1.0

        t_stat, p_value = StabilityAnalyzer._welch_ttest([1.0], [2.0])
        assert t_stat == 0.0
        assert p_value == 1.0

    def test_identical_groups(self):
        """两组完全相同时 p_value 应接近 1.0。"""
        data = [5.0, 6.0, 7.0, 8.0, 9.0]
        t_stat, p_value = StabilityAnalyzer._welch_ttest(data, data[:])
        assert p_value > 0.99
        assert abs(t_stat) < 0.01

    def test_p_value_bounded(self):
        """p_value 应在 [0, 1] 范围内。"""
        a = [1.0] * 10 + [100.0] * 10
        b = [50.0] * 20
        t_stat, p_value = StabilityAnalyzer._welch_ttest(a, b)
        assert 0.0 <= p_value <= 1.0


# ── answer_entropy ──


class TestAnswerEntropy:
    def test_all_passed(self):
        """全部 passed 时熵为 0。"""
        assert StabilityAnalyzer._calc_answer_entropy([1.0, 1.0, 1.0]) == 0.0

    def test_all_failed(self):
        """全部 failed 时熵为 0。"""
        assert StabilityAnalyzer._calc_answer_entropy([0.0, 0.0, 0.0]) == 0.0

    def test_half_passed(self):
        """一半通过时熵 > 0。"""
        entropy = StabilityAnalyzer._calc_answer_entropy([1.0, 1.0, 0.0, 0.0])
        assert entropy > 0
        assert entropy == pytest.approx(1.0)  # 2 passed, 2 failed → log2(1) = 1.0

    def test_empty_list(self):
        assert StabilityAnalyzer._calc_answer_entropy([]) == 0.0

    def test_threshold_0_5(self):
        """0.5 分视为 passed。"""
        entropy = StabilityAnalyzer._calc_answer_entropy([0.5, 0.0])
        # 1 passed, 1 failed → entropy = 1.0
        assert entropy == pytest.approx(1.0)


# ── overall_status 判定 ──


class TestDetermineStatus:
    def test_stable_no_issues(self):
        """无异常、无变化点、无不显著测试 → stable。"""
        status = StabilityAnalyzer._determine_status([], [], [])
        assert status == "stable"

    def test_degraded_score_significant(self):
        """score 显著下降且 effect_size < -0.5 → degraded。"""
        stat_tests = [
            {
                "signal": "score",
                "p_value": 0.001,
                "effect_size": -1.5,
                "significant": True,
            }
        ]
        status = StabilityAnalyzer._determine_status([], [], stat_tests)
        assert status == "degraded"

    def test_degraded_many_anomalies(self):
        """>= 3 个异常 → degraded。"""
        anomalies = [
            AnomalyDetail("a", 1.0, 0.5, 0.1, 5.0),
            AnomalyDetail("b", 1.0, 0.5, 0.1, 5.0),
            AnomalyDetail("c", 1.0, 0.5, 0.1, 5.0),
        ]
        status = StabilityAnalyzer._determine_status(anomalies, [], [])
        assert status == "degraded"

    def test_degraded_format_compliance_low(self):
        """format_compliance 异常且当前值 < 0.5 → degraded。"""
        anomalies = [
            AnomalyDetail("format_compliance", 0.3, 0.9, 0.05, -12.0)
        ]
        status = StabilityAnalyzer._determine_status(anomalies, [], [])
        assert status == "degraded"

    def test_degraded_repetition_high(self):
        """repetition_ratio 异常且当前值 > 0.3 → degraded。"""
        anomalies = [
            AnomalyDetail("repetition_ratio", 0.5, 0.05, 0.01, 45.0)
        ]
        status = StabilityAnalyzer._determine_status(anomalies, [], [])
        assert status == "degraded"

    def test_suspicious_perf_change(self):
        """tps/ttft 变化点 → suspicious。"""
        change_points = [
            ChangePoint("tps_zscore", datetime.now(), "decrease", -3.0)
        ]
        status = StabilityAnalyzer._determine_status([], change_points, [])
        assert status == "suspicious"

    def test_suspicious_thinking_change(self):
        """thinking_ratio 显著变化 → suspicious。"""
        stat_tests = [
            {
                "signal": "thinking_ratio",
                "p_value": 0.001,
                "effect_size": 0.5,
                "significant": True,
            }
        ]
        status = StabilityAnalyzer._determine_status([], [], stat_tests)
        assert status == "suspicious"

    def test_suspicious_single_anomaly(self):
        """1-2 个异常 → suspicious（不是 degraded）。"""
        anomalies = [AnomalyDetail("a", 1.0, 0.5, 0.1, 5.0)]
        status = StabilityAnalyzer._determine_status(anomalies, [], [])
        assert status == "suspicious"

    def test_score_not_significant_not_degraded(self):
        """score t-test 不显著时，不应判定为 degraded（score 路径）。"""
        stat_tests = [
            {
                "signal": "score",
                "p_value": 0.1,  # > bonferroni alpha
                "effect_size": -1.0,
                "significant": False,
            }
        ]
        status = StabilityAnalyzer._determine_status([], [], stat_tests)
        assert status == "stable"

    def test_format_anomaly_above_threshold_not_degraded(self):
        """format_compliance 异常但值 >= 0.5 → 不触发 degraded（无其他条件时）。"""
        anomalies = [
            AnomalyDetail("format_compliance", 0.7, 0.9, 0.05, -4.0)
        ]
        # 0.7 >= 0.5，不触发 format_degraded
        # 但有 1 个 anomaly → suspicious
        status = StabilityAnalyzer._determine_status(anomalies, [], [])
        assert status == "suspicious"


# ── Summary 生成 ──


class TestGenerateSummary:
    def test_stable_summary(self):
        summary = StabilityAnalyzer._generate_summary("stable", [], [], [])
        assert summary == "No significant changes detected."

    def test_degraded_with_anomalies(self):
        anomalies = [
            AnomalyDetail("repetition_ratio", 0.5, 0.05, 0.01, 45.0),
            AnomalyDetail("format_compliance", 0.3, 0.9, 0.05, -12.0),
        ]
        summary = StabilityAnalyzer._generate_summary(
            "degraded", anomalies, [], []
        )
        assert "Anomalies:" in summary
        assert "repetition_ratio" in summary
        assert "format_compliance" in summary

    def test_with_change_points(self):
        change_points = [
            ChangePoint("tps_zscore", datetime.now(), "decrease", -3.0)
        ]
        summary = StabilityAnalyzer._generate_summary(
            "suspicious", [], change_points, []
        )
        assert "Change points:" in summary
        assert "tps_zscore decrease" in summary

    def test_with_significant_tests(self):
        stat_tests = [
            {"signal": "score", "p_value": 0.001, "significant": True},
            {"signal": "tps_zscore", "p_value": 0.5, "significant": False},
        ]
        summary = StabilityAnalyzer._generate_summary(
            "suspicious", [], [], stat_tests
        )
        assert "Significant:" in summary
        assert "score" in summary
        assert "tps_zscore" not in summary  # 不显著，不应出现在 summary 中

    def test_combined(self):
        anomalies = [AnomalyDetail("a", 1.0, 0.5, 0.1, 5.0)]
        change_points = [
            ChangePoint("tps_zscore", datetime.now(), "decrease", -3.0)
        ]
        stat_tests = [{"signal": "score", "p_value": 0.001, "significant": True}]
        summary = StabilityAnalyzer._generate_summary(
            "degraded", anomalies, change_points, stat_tests
        )
        assert "Anomalies:" in summary
        assert "Change points:" in summary
        assert "Significant:" in summary


# ── z-score 异常检测 ──


class TestDetectAnomalies:
    def test_no_anomalies_normal_data(self):
        """正常数据不应检测到异常。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        current = [{"tps_zscore": 0.1, "ttft_zscore": -0.1, "thinking_ratio": 0.3,
                     "format_compliance": 0.95, "repetition_ratio": 0.05,
                     "garbled_text_ratio": 0.0, "language_consistency": 1.0,
                     "output_length_zscore": 0.0, "token_efficiency_zscore": 0.0,
                     "created_at": datetime.now().isoformat()}]
        history = [{"tps_zscore": 0.0, "ttft_zscore": 0.0, "thinking_ratio": 0.3,
                     "format_compliance": 0.95, "repetition_ratio": 0.05,
                     "garbled_text_ratio": 0.0, "language_consistency": 1.0,
                     "output_length_zscore": 0.0, "token_efficiency_zscore": 0.0,
                     "created_at": datetime.now().isoformat()} for _ in range(10)]
        result = analyzer._detect_anomalies(current, history)
        assert result == []
        db.close()

    def test_anomaly_detected(self):
        """异常数据应被检测到。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        # 当前 tps_zscore 远高于历史（历史有小方差）
        current = [{"tps_zscore": 10.0, "ttft_zscore": 0.0, "thinking_ratio": 0.3,
                     "format_compliance": 0.95, "repetition_ratio": 0.05,
                     "garbled_text_ratio": 0.0, "language_consistency": 1.0,
                     "output_length_zscore": 0.0, "token_efficiency_zscore": 0.0,
                     "created_at": datetime.now().isoformat()}]
        # 历史有小方差，让 z-score 计算有意义
        history = [{"tps_zscore": 0.0 + 0.1 * i, "ttft_zscore": 0.0 + 0.05 * i,
                     "thinking_ratio": 0.3, "format_compliance": 0.95,
                     "repetition_ratio": 0.05, "garbled_text_ratio": 0.0,
                     "language_consistency": 1.0, "output_length_zscore": 0.0 + 0.1 * i,
                     "token_efficiency_zscore": 0.0 + 0.1 * i,
                     "created_at": datetime.now().isoformat()} for i in range(10)]
        result = analyzer._detect_anomalies(current, history)
        assert len(result) >= 1
        assert any(a.signal_name == "tps_zscore" for a in result)
        db.close()

    def test_insufficient_history(self):
        """历史数据不足时返回空列表。"""
        db = _test_db()
        analyzer = StabilityAnalyzer(db)
        current = [{"tps_zscore": 10.0, "created_at": datetime.now().isoformat()}]
        history = [{"tps_zscore": 0.0, "created_at": datetime.now().isoformat()}]
        result = analyzer._detect_anomalies(current, history)
        assert result == []
        db.close()


# ── 完整 run() 流程 ──


class TestRun:
    def test_stable_run(self):
        """正常数据 → stable 报告。"""
        db = _test_db()
        # 插入 7 天历史数据
        _insert_history_data(db, "test/model", n_days=7, results_per_day=5)

        # 插入当前 run（正常数据，分数与历史接近）
        now = datetime.now()
        run = EvalRun(
            run_id="run-stable-001",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            finished_at=now,
            status="completed",
        )
        db.create_run(run)
        rids = []
        for i in range(5):
            rid = f"run-stable-001-res-{i}"
            result = EvalResult(
                result_id=rid,
                run_id="run-stable-001",
                task_id=f"t{i}",
                task_content="test",
                model_output="output",
                # 与历史分数分布接近
                functional_score=0.7 + 0.01 * i,
                quality_score=0.7 + 0.01 * i,
                final_score=0.7 + 0.01 * i,
                passed=True,
                execution_time=1.0,
                created_at=now,
            )
            db.save_result(result)
            rids.append(rid)
            # 每个 result 单独插入信号，与历史分布一致
            per_result = {
                "tps_zscore": 0.5 + 0.01 * i,
                "ttft_zscore": -0.2 + 0.01 * i,
                "thinking_ratio": 0.3 + 0.01 * i,
                "output_length_zscore": 0.1 + 0.01 * i,
                "token_efficiency_zscore": 0.0 + 0.01 * i,
            }
            _insert_signals(db, [rid], per_result)

        analyzer = StabilityAnalyzer(db, history_days=7)
        report = asyncio.run(analyzer.run("test/model", "run-stable-001", "reasoning"))

        assert report.model == "test/model"
        assert report.run_id == "run-stable-001"
        assert report.overall_status == "stable"
        assert isinstance(report, StabilityReport)

        # 验证已保存到 DB
        rows = db._get_stability_reports(model="test/model")
        assert len(rows) >= 1
        assert rows[0]["overall_status"] == "stable"
        db.close()

    def test_degraded_run(self):
        """分数显著下降 + 多个异常 → degraded 报告。"""
        db = _test_db()
        # 历史数据：正常分数
        _insert_history_data(db, "test/model", n_days=7, results_per_day=3)

        # 当前 run：分数很低
        now = datetime.now()
        run = EvalRun(
            run_id="run-degraded-001",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            finished_at=now,
            status="completed",
        )
        db.create_run(run)
        rids = []
        for i in range(5):
            rid = f"run-degraded-001-res-{i}"
            result = EvalResult(
                result_id=rid,
                run_id="run-degraded-001",
                task_id=f"t{i}",
                task_content="test",
                model_output="output",
                functional_score=0.1,
                quality_score=0.1,
                final_score=0.1,
                passed=False,
                execution_time=1.0,
                created_at=now,
            )
            db.save_result(result)
            rids.append(rid)
        # 异常信号：format_compliance 很低, repetition_ratio 很高
        bad_signals = {
            "format_compliance": 0.0,
            "repetition_ratio": 0.5,
            "garbled_text_ratio": 0.3,
        }
        _insert_signals(db, rids, bad_signals)

        analyzer = StabilityAnalyzer(db, history_days=7)
        report = asyncio.run(
            analyzer.run("test/model", "run-degraded-001", "reasoning")
        )

        assert report.overall_status == "degraded"
        db.close()

    def test_suspicious_run(self):
        """单个小异常 → suspicious 报告。"""
        db = _test_db()
        _insert_history_data(db, "test/model", n_days=7, results_per_day=3)

        now = datetime.now()
        run = EvalRun(
            run_id="run-suspicious-001",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            finished_at=now,
            status="completed",
        )
        db.create_run(run)
        rids = []
        for i in range(5):
            rid = f"run-suspicious-001-res-{i}"
            result = EvalResult(
                result_id=rid,
                run_id="run-suspicious-001",
                task_id=f"t{i}",
                task_content="test",
                model_output="output",
                functional_score=0.7,
                quality_score=0.7,
                final_score=0.7,
                passed=True,
                execution_time=1.0,
                created_at=now,
            )
            db.save_result(result)
            rids.append(rid)
        # tps_zscore 异常高，但其他正常
        _insert_signals(db, rids, {"tps_zscore": 5.0})

        analyzer = StabilityAnalyzer(db, history_days=7)
        report = asyncio.run(
            analyzer.run("test/model", "run-suspicious-001", "reasoning")
        )

        assert report.overall_status in ("suspicious", "degraded")
        db.close()

    def test_no_history_stable(self):
        """无历史数据时应安全返回 stable。"""
        db = _test_db()

        now = datetime.now()
        run = EvalRun(
            run_id="run-nohist-001",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            finished_at=now,
            status="completed",
        )
        db.create_run(run)
        rid = "run-nohist-001-res-0"
        result = EvalResult(
            result_id=rid,
            run_id="run-nohist-001",
            task_id="t0",
            task_content="test",
            model_output="output",
            functional_score=0.7,
            quality_score=0.7,
            final_score=0.7,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)
        _insert_signals(db, [rid])

        analyzer = StabilityAnalyzer(db, history_days=7)
        report = asyncio.run(
            analyzer.run("test/model", "run-nohist-001", "reasoning")
        )

        assert report.overall_status == "stable"
        assert report.summary == "No significant changes detected."
        db.close()

    def test_report_saved_to_db(self):
        """验证报告正确保存到 stability_reports 表。"""
        db = _test_db()
        _insert_history_data(db, "test/model", n_days=3, results_per_day=2)

        now = datetime.now()
        run = EvalRun(
            run_id="run-save-001",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            finished_at=now,
            status="completed",
        )
        db.create_run(run)
        rid = "run-save-001-res-0"
        result = EvalResult(
            result_id=rid,
            run_id="run-save-001",
            task_id="t0",
            task_content="test",
            model_output="output",
            functional_score=0.7,
            quality_score=0.7,
            final_score=0.7,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)
        _insert_signals(db, [rid])

        analyzer = StabilityAnalyzer(db, history_days=3)
        report = asyncio.run(
            analyzer.run("test/model", "run-save-001", "reasoning")
        )

        rows = db._get_stability_reports(model="test/model")
        assert len(rows) >= 1
        saved = rows[0]
        assert saved["model"] == "test/model"
        assert saved["run_id"] == "run-save-001"
        assert saved["overall_status"] == report.overall_status
        db.close()


# ── Cohen's d ──


class TestCohensD:
    def test_positive_effect(self):
        d = StabilityAnalyzer._cohens_d(
            [20.0 + 0.1 * i for i in range(10)],
            [10.0 + 0.1 * i for i in range(10)],
        )
        assert d > 0

    def test_negative_effect(self):
        d = StabilityAnalyzer._cohens_d(
            [10.0 + 0.1 * i for i in range(10)],
            [20.0 + 0.1 * i for i in range(10)],
        )
        assert d < 0

    def test_zero_effect(self):
        d = StabilityAnalyzer._cohens_d([10.0] * 10, [10.0] * 10)
        assert d == 0.0

    def test_identical_variance(self):
        """方差相同但均值不同时效应量应合理。"""
        import math
        d = StabilityAnalyzer._cohens_d([15.0] * 10, [10.0] * 10)
        # variance = 0, pooled_std = 0 → returns 0.0
        assert d == 0.0
