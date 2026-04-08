"""测试 FingerprintManager：指纹生成、存储、对比。"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from benchmark.analysis.fingerprint import (
    FingerprintManager,
    _cosine_similarity,
    _sanitize_model,
)


# ── fixtures ──


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path / "fp_test"


@pytest.fixture
def mgr(tmp_dir: Path) -> FingerprintManager:
    return FingerprintManager(str(tmp_dir))


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


# ── _sanitize_model ──


class TestSanitizeModel:
    def test_slash_replaced(self):
        assert _sanitize_model("glm/glm-4.7") == "glm__glm-4.7"

    def test_no_slash(self):
        assert _sanitize_model("gpt-4o") == "gpt-4o"


# ── _cosine_similarity ──


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero(self):
        assert _cosine_similarity([0.0], [0.0]) == 0.0


# ── generate_fingerprint_sync ──


class TestGenerateFingerprint:
    def test_vector_dimension(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        fp = mgr.generate_fingerprint_sync("glm/glm-4.7", scores, qs)
        assert len(fp["vector"]) == 33

    def test_scores_normalized(self, mgr: FingerprintManager):
        scores = [50.0] * 20
        qs = _make_quality_signals(20)
        fp = mgr.generate_fingerprint_sync("test/model", scores, qs)
        # 前 20 维应该全为 0.5
        for v in fp["vector"][:20]:
            assert v == pytest.approx(0.5)

    def test_quality_signals_aggregated(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        # 所有题的 format_compliance = 0.8
        qs = _make_quality_signals(20, format_compliance=0.8)
        fp = mgr.generate_fingerprint_sync("test/model", scores, qs)
        # 第 21 维（第一个聚合信号）应该是 0.8
        assert fp["vector"][20] == pytest.approx(0.8)

    def test_aggregation_averages_across_questions(self, mgr: FingerprintManager):
        """验证聚合信号是取均值，而非取单个值。"""
        scores = [100.0] * 20
        # 构造 2 题的信号，其中 format_compliance 分别为 0.0 和 1.0
        qs_list = [
            {"format_compliance": 0.0, **{k: 0.0 for k in [
                "repetition_ratio", "garbled_text_ratio", "refusal_detected",
                "language_consistency", "output_length_zscore", "thinking_ratio",
                "empty_reasoning", "truncated", "token_efficiency_zscore",
                "tps_zscore", "ttft_zscore", "answer_entropy",
            ]}},
            {"format_compliance": 1.0, **{k: 0.0 for k in [
                "repetition_ratio", "garbled_text_ratio", "refusal_detected",
                "language_consistency", "output_length_zscore", "thinking_ratio",
                "empty_reasoning", "truncated", "token_efficiency_zscore",
                "tps_zscore", "ttft_zscore", "answer_entropy",
            ]}},
        ]
        fp = mgr.generate_fingerprint_sync("test/model", scores, qs_list)
        # 2 题均值应该为 0.5
        assert fp["vector"][20] == pytest.approx(0.5)

    def test_metadata_fields(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        fp = mgr.generate_fingerprint_sync("glm/glm-4.7", scores, qs)
        assert fp["model"] == "glm/glm-4.7"
        assert fp["timestamp"] is not None
        assert fp["run_id"] == ""

    def test_baseline_created_on_first_run(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        mgr.generate_fingerprint_sync("test/model", scores, qs)

        baseline_file = mgr._dir / "test__model" / "baseline.json"
        assert baseline_file.exists()
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        assert baseline["model"] == "test/model"

    def test_timestamp_file_created(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        fp = mgr.generate_fingerprint_sync("test/model", scores, qs)

        model_dir = mgr._dir / "test__model"
        ts_files = [f for f in model_dir.glob("*.json") if f.name != "baseline.json"]
        assert len(ts_files) == 1
        assert ts_files[0].stem == fp["timestamp"]


# ── async 版本 ──


class TestGenerateFingerprintAsync:
    @pytest.mark.anyio
    async def test_async_delegates_to_sync(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        fp = await mgr.generate_fingerprint("test/model", scores, qs)
        assert len(fp["vector"]) == 33


# ── compare_with_baseline ──


class TestCompareWithBaseline:
    def _make_two_runs(
        self, mgr: FingerprintManager, model: str
    ) -> tuple[dict, dict]:
        """生成两次运行，返回 (baseline, second) 指纹。

        使用完全相反的分数和质量信号，确保相似度很低。
        """
        scores1 = [100.0] * 20
        qs1 = _make_quality_signals(20, format_compliance=1.0, repetition_ratio=0.0)
        fp1 = mgr.generate_fingerprint_sync(model, scores1, qs1)

        scores2 = [0.0] * 20
        qs2 = _make_quality_signals(20, format_compliance=0.0, repetition_ratio=1.0)
        fp2 = mgr.generate_fingerprint_sync(model, scores2, qs2)

        return fp1, fp2

    def test_no_baseline(self, mgr: FingerprintManager):
        result = mgr.compare_with_baseline("nonexistent/model")
        assert result["status"] == "no_baseline"
        assert result["similarity"] == 0.0
        assert result["baseline_timestamp"] is None

    def test_match_when_identical(self, mgr: FingerprintManager):
        """相同分数应产生 similarity ≈ 1.0。"""
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        mgr.generate_fingerprint_sync("test/model", scores, qs)

        result = mgr.compare_with_baseline("test/model")
        assert result["status"] == "match"
        assert result["similarity"] == pytest.approx(1.0)

    def test_suspected_change_below_threshold(self, mgr: FingerprintManager):
        fp1, fp2 = self._make_two_runs(mgr, "test/model")

        result = mgr.compare_with_baseline("test/model")
        # 第一次运行全对 + 第二次全错，质量信号也反转，相似度应远低于 0.85
        assert result["similarity"] < 0.5
        assert result["status"] == "suspected_model_change"

    def test_custom_threshold(self, mgr: FingerprintManager):
        fp1, fp2 = self._make_two_runs(mgr, "test/model")

        # 用很低的阈值，即使有差异也应该 match
        result = mgr.compare_with_baseline("test/model", threshold=0.1)
        assert result["status"] == "match"

    def test_explicit_fingerprint(self, mgr: FingerprintManager):
        """可以传入显式指纹进行对比。"""
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        mgr.generate_fingerprint_sync("test/model", scores, qs)

        # 构造一个差异很大的指纹
        different_fp = {
            "model": "test/model",
            "timestamp": "2099-01-01T00:00:00",
            "run_id": "",
            "vector": [0.0] * 33,
        }

        result = mgr.compare_with_baseline("test/model", fingerprint=different_fp)
        assert result["status"] == "suspected_model_change"
        assert result["current_timestamp"] == "2099-01-01T00:00:00"

    def test_return_fields(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        mgr.generate_fingerprint_sync("test/model", scores, qs)

        result = mgr.compare_with_baseline("test/model")
        assert set(result.keys()) == {
            "model", "similarity", "status",
            "baseline_timestamp", "current_timestamp",
        }
        assert isinstance(result["similarity"], float)


# ── get_fingerprint_history ──


class TestGetFingerprintHistory:
    def test_empty_history(self, mgr: FingerprintManager):
        history = mgr.get_fingerprint_history("nonexistent/model")
        assert history == []

    def test_returns_all_fingerprints(self, mgr: FingerprintManager):
        for i in range(3):
            scores = [float(i * 10)] * 20
            qs = _make_quality_signals(20)
            mgr.generate_fingerprint_sync("test/model", scores, qs)

        history = mgr.get_fingerprint_history("test/model")
        # baseline + 3 timestamp files = 4
        assert len(history) == 4
        # 应按文件名（时间戳）升序
        names = [h["_source_file"] for h in history]
        assert names == sorted(names)

    def test_history_includes_source_file(self, mgr: FingerprintManager):
        scores = [100.0] * 20
        qs = _make_quality_signals(20)
        mgr.generate_fingerprint_sync("test/model", scores, qs)

        history = mgr.get_fingerprint_history("test/model")
        for h in history:
            assert "_source_file" in h
