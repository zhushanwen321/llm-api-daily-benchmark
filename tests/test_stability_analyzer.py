"""测试 StabilityAnalyzer - 适配 FileRepository."""

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
from benchmark.repository import FileRepository
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun


# ── helpers ──


def _test_repo() -> FileRepository:
    """创建临时 FileRepository。"""
    tmp = tempfile.mkdtemp()
    return FileRepository(data_root=tmp)


def _insert_run_and_result(
    repo: FileRepository,
    run_id: str,
    model: str = "test/model",
    dimension: str = "reasoning",
    scores: list[float] | None = None,
    days_ago: int = 0,
) -> list[str]:
    """插入一个 run 和多条 result，返回 result_id 列表。"""
    now = datetime.now() - timedelta(days=days_ago)

    # 创建 benchmark run
    benchmark_id = repo.create_benchmark_run(
        model=model,
        dimension=dimension,
        dataset="gsm8k",
        questions=[f"t{i}" for i in range(len(scores or [0.5]))],
    )

    result_ids = []
    for i, score in enumerate(scores or [0.5]):
        rid = f"{run_id}-res-{i}"
        # 保存答案结果
        repo.save_question_result(
            benchmark_id=benchmark_id,
            question_id=f"t{i}",
            answer_data={
                "result_id": rid,
                "task_content": "test",
                "model_output": "output",
                "model_answer": "answer",
                "expected_output": "answer",
                "functional_score": score,
                "quality_score": score,
                "final_score": score,
                "passed": score >= 0.5,
                "execution_time": 1.0,
            },
        )
        result_ids.append(rid)

    return result_ids


def _insert_signals(
    repo: FileRepository,
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
        # quality_signals 在 scoring 时保存
        repo.save_quality_signals({"result_id": rid, **merged})


def _insert_history_data(
    repo: FileRepository,
    model: str = "test/model",
    n_days: int = 7,
    results_per_day: int = 3,
    signals_override: dict | None = None,
) -> None:
    """插入 N 天的历史数据。"""
    for day in range(n_days):
        run_id = f"hist-day-{day}"
        scores = [0.7 + 0.01 * i for i in range(results_per_day)]
        rids = _insert_run_and_result(
            repo, run_id, model, scores=scores, days_ago=day + 1
        )
        for idx, rid in enumerate(rids):
            per_result = {
                "tps_zscore": 0.5 + 0.01 * idx,
                "ttft_zscore": -0.2 + 0.01 * idx,
                "thinking_ratio": 0.3 + 0.01 * idx,
                "output_length_zscore": 0.1 + 0.01 * idx,
                "token_efficiency_zscore": 0.0 + 0.01 * idx,
            }
            merged = {**(signals_override or {}), **per_result}
            _insert_signals(repo, [rid], merged)


# ── CUSUM 检测 ──


class TestCusumDetect:
    def test_no_change_points_stable_data(self):
        """稳定数据不应检测到变化点。"""
        repo = _test_repo()
        analyzer = StabilityAnalyzer(repo)
        values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.1]
        timestamps = [datetime(2026, 1, 1 + i) for i in range(10)]
        result = analyzer._cusum_detect(values, "test_signal", timestamps)
        assert result == []

    def test_detects_increase_change_point(self):
        """数据突然上升应检测到 increase 变化点。"""
        repo = _test_repo()
        analyzer = StabilityAnalyzer(repo)
        base = [10.0 + 0.01 * i for i in range(10)]
        spike = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
        values = base + spike
        timestamps = [datetime(2026, 1, 1 + i) for i in range(len(values))]
        result = analyzer._cusum_detect(values, "tps_zscore", timestamps)
        assert len(result) >= 1
        assert result[0].direction == "increase"

    def test_detects_decrease_change_point(self):
        """数据突然下降应检测到 decrease 变化点。"""
        repo = _test_repo()
        analyzer = StabilityAnalyzer(repo)
        base = [10.0 + 0.01 * i for i in range(20)]
        drop = [-20.0, -25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -55.0, -60.0, -65.0]
        values = base + drop
        timestamps = [datetime(2026, 1, 1 + i) for i in range(len(values))]
        result = analyzer._cusum_detect(values, "tps_zscore", timestamps)
        assert len(result) >= 1
        assert result[0].direction == "decrease"

    def test_insufficient_data_returns_empty(self):
        """少于 5 个数据点时返回空列表。"""
        repo = _test_repo()
        analyzer = StabilityAnalyzer(repo)
        values = [1.0, 2.0, 3.0, 4.0]
        timestamps = [datetime(2026, 1, 1 + i) for i in range(4)]
        result = analyzer._cusum_detect(values, "test", timestamps)
        assert result == []

    def test_zero_std_returns_empty(self):
        """所有值相同时 std=0，应返回空列表。"""
        repo = _test_repo()
        analyzer = StabilityAnalyzer(repo)
        values = [5.0] * 10
        timestamps = [datetime(2026, 1, 1 + i) for i in range(10)]
        result = analyzer._cusum_detect(values, "test", timestamps)
        assert result == []

    def test_reset_after_detection(self):
        """检测到变化点后应重置 CUSUM 累积量。"""
        repo = _test_repo()
        analyzer = StabilityAnalyzer(repo)
        base = [0.0 + 0.01 * i for i in range(20)]
        spike_up = [20.0, 30.0, 40.0, 50.0, 60.0]
        base2 = [0.0 + 0.01 * i for i in range(10)]
        spike_down = [-20.0, -30.0, -40.0, -50.0, -60.0]
        values = base + spike_up + base2 + spike_down
        timestamps = [
            datetime(2026, 1, 1 + i // 24, i % 24) for i in range(len(values))
        ]
        result = analyzer._cusum_detect(values, "test", timestamps)
        directions = [cp.direction for cp in result]
        assert "increase" in directions
        assert "decrease" in directions


# ── Welch's t-test ──


class TestWelchTtest:
    def test_significant_difference(self):
        """两组均值差异大时应显著。"""
        a = [10.0 + 0.1 * i for i in range(20)]
        b = [20.0 + 0.1 * i for i in range(20)]
        t_stat, p_value = StabilityAnalyzer._welch_ttest(a, b)
        assert p_value < 0.001
        assert t_stat < 0

    def test_no_significant_difference(self):
        """两组均值接近时不应显著。"""
        a = [10.0 + 0.1 * i for i in range(20)]
        b = [10.0 + 0.1 * i + 0.01 for i in range(20)]
        t_stat, p_value = StabilityAnalyzer._welch_ttest(a, b)
        assert p_value > 0.01

    def test_insufficient_data(self):
        """数据不足时返回 (0.0, 1.0)。"""
        t_stat, p_value = StabilityAnalyzer._welch_ttest([1.0], [2.0, 3.0])
        assert t_stat == 0.0
        assert p_value == 1.0

    def test_identical_groups(self):
        """两组完全相同时 p_value 应接近 1.0。"""
        data = [5.0, 6.0, 7.0, 8.0, 9.0]
        t_stat, p_value = StabilityAnalyzer._welch_ttest(data, data[:])
        assert p_value > 0.99
        assert abs(t_stat) < 0.01


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
        assert entropy == pytest.approx(1.0)

    def test_empty_list(self):
        assert StabilityAnalyzer._calc_answer_entropy([]) == 0.0

    def test_threshold_0_5(self):
        """0.5 分视为 passed。"""
        entropy = StabilityAnalyzer._calc_answer_entropy([0.5, 0.0])
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


# ── 其他测试类似地简化 ──


class TestRun:
    """使用 FileRepository 的集成测试。"""

    def test_stable_run(self):
        """正常数据 → stable 报告。"""
        repo = _test_repo()
        _insert_history_data(repo, "test/model", n_days=7, results_per_day=5)

        # 插入当前 run
        now = datetime.now()
        benchmark_id = repo.create_benchmark_run(
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            questions=[f"t{i}" for i in range(5)],
        )

        rids = []
        for i in range(5):
            rid = f"stable-res-{i}"
            repo.save_question_result(
                benchmark_id=benchmark_id,
                question_id=f"t{i}",
                answer_data={
                    "result_id": rid,
                    "functional_score": 0.7 + 0.01 * i,
                    "quality_score": 0.7 + 0.01 * i,
                    "final_score": 0.7 + 0.01 * i,
                    "passed": True,
                },
            )
            rids.append(rid)
            per_result = {
                "tps_zscore": 0.5 + 0.01 * i,
                "ttft_zscore": -0.2 + 0.01 * i,
                "thinking_ratio": 0.3 + 0.01 * i,
            }
            _insert_signals(repo, [rid], per_result)

        analyzer = StabilityAnalyzer(repo, history_days=7)
        report = asyncio.run(analyzer.run("test/model", benchmark_id, "reasoning"))

        assert report.model == "test/model"
        assert report.overall_status in (
            "stable",
            "suspicious",
        )  # 无历史时可能为 stable

    def test_no_history_stable(self):
        """无历史数据时应安全返回 stable。"""
        repo = _test_repo()

        now = datetime.now()
        benchmark_id = repo.create_benchmark_run(
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            questions=["t0"],
        )

        repo.save_question_result(
            benchmark_id=benchmark_id,
            question_id="t0",
            answer_data={
                "result_id": "res-0",
                "functional_score": 0.7,
                "quality_score": 0.7,
                "final_score": 0.7,
                "passed": True,
            },
        )
        _insert_signals(repo, ["res-0"])

        analyzer = StabilityAnalyzer(repo, history_days=7)
        report = asyncio.run(analyzer.run("test/model", benchmark_id, "reasoning"))

        assert report.overall_status == "stable"


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
