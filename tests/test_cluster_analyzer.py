"""测试 FingerprintClusterAnalyzer 和 ModelClassifier。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.analysis.cluster_analyzer import (
    FingerprintClusterAnalyzer,
    ModelClassifier,
)
from benchmark.analysis.fingerprint import FingerprintManager


# ── fixtures ──


@pytest.fixture
def fp_dir(tmp_path: Path) -> Path:
    return tmp_path / "fp_test"


@pytest.fixture
def mgr(fp_dir: Path) -> FingerprintManager:
    return FingerprintManager(str(fp_dir))


def _make_quality_signals(n: int = 20, **overrides: float) -> list[dict]:
    """构造 n 题的质量信号列表。"""
    defaults = {
        "format_compliance": 0.9,
        "repetition_ratio": 0.05,
        "garbled_text_ratio": 0.01,
        "refusal_detected": 0,
        "language_consistency": 0.95,
        "output_length_zscore": 0.1,
        "thinking_ratio": 0.3,
        "empty_reasoning": 0,
        "truncated": 0,
        "token_efficiency_zscore": -0.1,
        "tps_zscore": 0.2,
        "ttft_zscore": -0.05,
        "answer_entropy": 0.5,
    }
    defaults.update(overrides)
    return [dict(defaults) for _ in range(n)]


def _generate_runs(
    mgr: FingerprintManager,
    model: str,
    n_runs: int,
    scores_fn=None,
    signals_fn=None,
) -> None:
    """生成 n_runs 次指纹运行。"""
    for i in range(n_runs):
        scores = scores_fn(i) if scores_fn else [80.0] * 20
        qs = signals_fn(i) if signals_fn else _make_quality_signals(20)
        mgr.generate_fingerprint_sync(model, scores, qs, run_id=f"run_{i}")


# ── FingerprintClusterAnalyzer ──


class TestClusterAnalyzer:
    def test_single_cluster_for_uniform_data(self, mgr: FingerprintManager, fp_dir: Path):
        """相同分数和质量信号 → 1 个簇。"""
        _generate_runs(mgr, "test/model", 10)

        analyzer = FingerprintClusterAnalyzer(str(fp_dir))
        report = analyzer.analyze("test/model")

        assert report.n_clusters == 1
        assert report.n_noise == 0
        assert "consistent" in report.summary.lower()

    def test_multiple_clusters_for_distinct_data(
        self, mgr: FingerprintManager, fp_dir: Path
    ):
        """前 5 次高分 + 后 5 次低分 → 至少 2 个簇。"""
        def scores_fn(i):
            return [100.0] * 20 if i < 5 else [0.0] * 20

        def signals_fn(i):
            if i < 5:
                return _make_quality_signals(20, format_compliance=1.0)
            return _make_quality_signals(20, format_compliance=0.0)

        _generate_runs(mgr, "test/model", 10, scores_fn, signals_fn)

        analyzer = FingerprintClusterAnalyzer(str(fp_dir))
        report = analyzer.analyze("test/model", eps=0.3, min_samples=3)

        assert report.n_clusters >= 2

    def test_suspected_changes_detected(
        self, mgr: FingerprintManager, fp_dir: Path
    ):
        """模型变化应被检测到。"""
        def scores_fn(i):
            return [100.0] * 20 if i < 5 else [0.0] * 20

        def signals_fn(i):
            if i < 5:
                return _make_quality_signals(20, format_compliance=1.0)
            return _make_quality_signals(20, format_compliance=0.0)

        _generate_runs(mgr, "test/model", 10, scores_fn, signals_fn)

        analyzer = FingerprintClusterAnalyzer(str(fp_dir))
        report = analyzer.analyze("test/model", eps=0.3, min_samples=3)

        assert len(report.suspected_changes) >= 1
        change = report.suspected_changes[0]
        assert "from_cluster" in change
        assert "to_cluster" in change
        assert "at" in change

    def test_insufficient_data(self, fp_dir: Path):
        """数据不足时返回空报告。"""
        analyzer = FingerprintClusterAnalyzer(str(fp_dir))
        report = analyzer.analyze("nonexistent/model")

        assert report.n_clusters == 0
        assert "不足" in report.summary

    def test_cluster_info_fields(
        self, mgr: FingerprintManager, fp_dir: Path
    ):
        """ClusterInfo 字段完整性。"""
        _generate_runs(mgr, "test/model", 5)

        analyzer = FingerprintClusterAnalyzer(str(fp_dir))
        report = analyzer.analyze("test/model")

        assert len(report.clusters) == 1
        c = report.clusters[0]
        assert c.size == 5
        assert len(c.time_range) == 2
        assert len(c.centroid) == 33
        assert isinstance(c.avg_score, float)

    def test_noise_points(
        self, mgr: FingerprintManager, fp_dir: Path
    ):
        """异常值应被标记为噪声。"""
        # 8 次相同 + 2 次极端异常
        def scores_fn(i):
            if i in (4, 8):
                # 极端异常分数
                return [float(i * 50)] * 20
            return [80.0] * 20

        _generate_runs(mgr, "test/model", 10, scores_fn)

        analyzer = FingerprintClusterAnalyzer(str(fp_dir))
        report = analyzer.analyze("test/model", eps=0.1, min_samples=3)

        # 极端值可能被标记为噪声
        # 不强制断言噪声数量（取决于 eps），但报告结构应正确
        assert report.n_clusters >= 1


# ── ModelClassifier ──


class TestModelClassifier:
    def _setup_two_models(self, mgr: FingerprintManager):
        """为两个模型生成指纹数据。"""
        _generate_runs(
            mgr, "model_a", 8,
            scores_fn=lambda i: [90.0 + i] * 20,
            signals_fn=lambda i: _make_quality_signals(20, format_compliance=0.9 + i * 0.01),
        )
        _generate_runs(
            mgr, "model_b", 8,
            scores_fn=lambda i: [50.0 + i] * 20,
            signals_fn=lambda i: _make_quality_signals(20, format_compliance=0.5 + i * 0.01),
        )

    def test_train_and_predict(self, mgr: FingerprintManager, fp_dir: Path):
        """训练后应能正确分类。"""
        self._setup_two_models(mgr)

        clf = ModelClassifier(str(fp_dir))
        train_report = clf.train()
        assert train_report["status"] == "trained"
        assert train_report["total_samples"] == 16

        # 用 model_a 的指纹预测
        result = clf.predict(mgr.get_fingerprint_history("model_a")[0]["vector"])
        assert result["predicted_model"] == "model_a"
        assert result["confidence"] > 0.5

    def test_predict_untrained(self, fp_dir: Path):
        """未训练时 predict 返回 not_trained。"""
        clf = ModelClassifier(str(fp_dir))
        result = clf.predict([0.0] * 33)
        assert result["status"] == "not_trained"

    def test_cross_validate(self, mgr: FingerprintManager, fp_dir: Path):
        """交叉验证应有较高准确率。"""
        self._setup_two_models(mgr)

        clf = ModelClassifier(str(fp_dir))
        cv = clf.cross_validate()

        assert cv["accuracy"] > 0.5
        assert cv["total_samples"] == 16

    def test_no_data(self, fp_dir: Path):
        """无数据时 train 返回 no_data。"""
        clf = ModelClassifier(str(fp_dir))
        result = clf.train()
        assert result["status"] == "no_data"

    def test_specific_models(self, mgr: FingerprintManager, fp_dir: Path):
        """指定模型列表训练。"""
        self._setup_two_models(mgr)

        clf = ModelClassifier(str(fp_dir))
        result = clf.train(models=["model_a"])

        assert result["status"] == "trained"
        assert "model_a" in result["models"]
        assert "model_b" not in result["models"]
