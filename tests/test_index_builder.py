from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.repository.index_builder import IndexBuilder


def _write_status(
    data_root: Path,
    benchmark_id: str,
    model: str,
    dimension: str,
    total_questions: int,
    answered: int = 0,
    scored: int = 0,
    status: str = "running",
    created_at: str = "2025-01-01T00:00:00+00:00",
    execution_id: str = "bench_test_0001",
) -> None:
    d = data_root / execution_id / benchmark_id
    d.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "model": model,
        "dimension": dimension,
        "total_questions": total_questions,
        "answered": answered,
        "scored": scored,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
    }
    (d / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _write_metadata(
    data_root: Path,
    benchmark_id: str,
    model: str,
    dimension: str,
    dataset: str,
    started_at: str = "2025-01-01T00:00:00+00:00",
    execution_id: str = "bench_test_0001",
) -> None:
    d = data_root / execution_id / benchmark_id
    d.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "model": model,
        "dimension": dimension,
        "dataset": dataset,
        "started_at": started_at,
        "config_snapshot": "{}",
    }
    with open(d / "metadata.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_scoring(
    data_root: Path,
    benchmark_id: str,
    question_id: str,
    task_id: str,
    final_score: float = 80.0,
    scoring_status: str = "completed",
    execution_id: str = "bench_test_0001",
) -> None:
    d = data_root / execution_id / benchmark_id / question_id
    d.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "task_id": task_id,
        "functional_score": final_score,
        "quality_score": final_score,
        "final_score": final_score,
        "passed": final_score >= 60,
        "details": {},
        "reasoning": "",
        "quality_signals": {},
        "scoring_status": scoring_status,
    }
    with open(d / "scoring.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def builder(tmp_root: Path) -> IndexBuilder:
    return IndexBuilder(data_root=tmp_root)


class TestBuild:
    def test_creates_index_jsonl(self, builder: IndexBuilder, tmp_root: Path) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3, status="completed")
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        assert (tmp_root / "index.jsonl").exists()

    def test_correct_number_of_rows(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        for bid in ("b1", "b2", "b3"):
            _write_status(tmp_root, bid, "model-a", "reasoning", 3, status="completed")
            _write_metadata(tmp_root, bid, "model-a", "reasoning", "gsm8k")
        builder.build()
        lines = (
            (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        assert len(lines) == 3

    def test_each_row_is_valid_json(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        lines = (
            (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_row_contains_all_required_fields(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(
            tmp_root,
            "b1",
            "model-a",
            "reasoning",
            10,
            answered=5,
            scored=3,
            status="scoring",
        )
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        line = (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(line)
        required = [
            "benchmark_id",
            "model",
            "dimension",
            "dataset",
            "total_questions",
            "answered",
            "scored",
            "status",
            "avg_score",
            "created_at",
        ]
        for field in required:
            assert field in row, f"missing field: {field}"

    def test_row_values_match_source(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(
            tmp_root,
            "b1",
            "zai/glm-4.7",
            "reasoning",
            10,
            answered=10,
            scored=10,
            status="completed",
            created_at="2025-03-15T10:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b1",
            "zai/glm-4.7",
            "reasoning",
            "gsm8k",
            started_at="2025-03-15T10:00:00+00:00",
        )
        _write_scoring(tmp_root, "b1", "q1", "q1", final_score=80.0)
        _write_scoring(tmp_root, "b1", "q2", "q2", final_score=90.0)
        builder.build()
        line = (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(line)
        assert row["benchmark_id"] == "b1"
        assert row["model"] == "zai/glm-4.7"
        assert row["dimension"] == "reasoning"
        assert row["dataset"] == "gsm8k"
        assert row["total_questions"] == 10
        assert row["answered"] == 10
        assert row["scored"] == 10
        assert row["status"] == "completed"
        assert row["created_at"] == "2025-03-15T10:00:00+00:00"

    def test_avg_score_calculated_from_scoring_files(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(
            tmp_root, "b1", "model-a", "reasoning", 3, scored=3, status="completed"
        )
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        _write_scoring(tmp_root, "b1", "q1", "q1", final_score=80.0)
        _write_scoring(tmp_root, "b1", "q2", "q2", final_score=90.0)
        _write_scoring(tmp_root, "b1", "q3", "q3", final_score=70.0)
        builder.build()
        line = (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(line)
        assert row["avg_score"] == pytest.approx(80.0)

    def test_avg_score_none_when_no_scoring(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        line = (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(line)
        assert row["avg_score"] is None

    def test_skips_dirs_without_status_json(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        (tmp_root / "junk_dir").mkdir()
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3, status="completed")
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        lines = (
            (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        assert len(lines) == 1

    def test_dataset_from_metadata(self, builder: IndexBuilder, tmp_root: Path) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "humaneval")
        builder.build()
        row = json.loads((tmp_root / "index.jsonl").read_text(encoding="utf-8").strip())
        assert row["dataset"] == "humaneval"

    def test_dataset_fallback_when_no_metadata(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        builder.build()
        row = json.loads((tmp_root / "index.jsonl").read_text(encoding="utf-8").strip())
        assert row["dataset"] == ""

    def test_overwrites_existing_index(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        (tmp_root / "index.jsonl").write_text("old data\n", encoding="utf-8")
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        lines = (
            (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["benchmark_id"] == "b1"

    def test_only_counts_completed_scoring_for_avg(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(
            tmp_root, "b1", "model-a", "reasoning", 3, scored=2, status="scoring"
        )
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        _write_scoring(tmp_root, "b1", "q1", "q1", final_score=100.0)
        _write_scoring(tmp_root, "b1", "q2", "q2", final_score=50.0)
        _write_scoring(
            tmp_root, "b1", "q3", "q3", final_score=75.0, scoring_status="pending"
        )
        builder.build()
        row = json.loads((tmp_root / "index.jsonl").read_text(encoding="utf-8").strip())
        assert row["avg_score"] == pytest.approx(75.0)


class TestBuildWithEmptyRoot:
    def test_empty_root_creates_empty_index(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        builder.build()
        index_path = tmp_root / "index.jsonl"
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8").strip()
        assert content == ""


class TestGetRunsByModel:
    def test_filters_by_model(self, builder: IndexBuilder, tmp_root: Path) -> None:
        _write_status(tmp_root, "b1", "zai/glm-4.7", "reasoning", 3, status="completed")
        _write_metadata(tmp_root, "b1", "zai/glm-4.7", "reasoning", "gsm8k")
        _write_status(
            tmp_root, "b2", "openai/gpt-4", "reasoning", 3, status="completed"
        )
        _write_metadata(tmp_root, "b2", "openai/gpt-4", "reasoning", "gsm8k")
        _write_status(tmp_root, "b3", "zai/glm-4.7", "coding", 5, status="completed")
        _write_metadata(tmp_root, "b3", "zai/glm-4.7", "coding", "humaneval")
        builder.build()
        results = builder.get_runs_by_model("zai/glm-4.7")
        assert len(results) == 2
        assert all(r["model"] == "zai/glm-4.7" for r in results)

    def test_returns_empty_for_unknown_model(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        builder.build()
        assert builder.get_runs_by_model("nonexistent") == []

    def test_auto_builds_if_index_missing(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        assert not (tmp_root / "index.jsonl").exists()
        results = builder.get_runs_by_model("model-a")
        assert len(results) == 1


class TestGetRunsByDimension:
    def test_filters_by_dimension(self, builder: IndexBuilder, tmp_root: Path) -> None:
        _write_status(tmp_root, "b1", "model-a", "reasoning", 3)
        _write_metadata(tmp_root, "b1", "model-a", "reasoning", "gsm8k")
        _write_status(tmp_root, "b2", "model-a", "coding", 5)
        _write_metadata(tmp_root, "b2", "model-a", "coding", "humaneval")
        builder.build()
        results = builder.get_runs_by_dimension("reasoning")
        assert len(results) == 1
        assert results[0]["dimension"] == "reasoning"


class TestGetRecentRuns:
    def test_returns_sorted_by_created_at_desc(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(
            tmp_root,
            "b1",
            "model-a",
            "reasoning",
            3,
            created_at="2025-01-01T00:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b1",
            "model-a",
            "reasoning",
            "gsm8k",
            started_at="2025-01-01T00:00:00+00:00",
        )
        _write_status(
            tmp_root,
            "b2",
            "model-a",
            "reasoning",
            3,
            created_at="2025-01-03T00:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b2",
            "model-a",
            "reasoning",
            "gsm8k",
            started_at="2025-01-03T00:00:00+00:00",
        )
        _write_status(
            tmp_root,
            "b3",
            "model-a",
            "reasoning",
            3,
            created_at="2025-01-02T00:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b3",
            "model-a",
            "reasoning",
            "gsm8k",
            started_at="2025-01-02T00:00:00+00:00",
        )
        builder.build()
        results = builder.get_recent_runs()
        assert len(results) == 3
        assert results[0]["benchmark_id"] == "b2"
        assert results[1]["benchmark_id"] == "b3"
        assert results[2]["benchmark_id"] == "b1"

    def test_limits_results(self, builder: IndexBuilder, tmp_root: Path) -> None:
        for i in range(5):
            _write_status(
                tmp_root,
                f"b{i}",
                "model-a",
                "reasoning",
                3,
                created_at=f"2025-01-0{i + 1}T00:00:00+00:00",
            )
            _write_metadata(
                tmp_root,
                f"b{i}",
                "model-a",
                "reasoning",
                "gsm8k",
                started_at=f"2025-01-0{i + 1}T00:00:00+00:00",
            )
        builder.build()
        results = builder.get_recent_runs(limit=3)
        assert len(results) == 3


class TestQAScenario:
    def test_index_correctness_with_3_benchmarks(
        self, builder: IndexBuilder, tmp_root: Path
    ) -> None:
        _write_status(
            tmp_root,
            "b1",
            "zai/glm-4.7",
            "reasoning",
            2,
            answered=2,
            scored=2,
            status="completed",
            created_at="2025-03-01T00:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b1",
            "zai/glm-4.7",
            "reasoning",
            "gsm8k",
            started_at="2025-03-01T00:00:00+00:00",
        )
        _write_scoring(tmp_root, "b1", "q1", "q1", final_score=90.0)
        _write_scoring(tmp_root, "b1", "q2", "q2", final_score=70.0)

        _write_status(
            tmp_root,
            "b2",
            "openai/gpt-4",
            "coding",
            3,
            answered=3,
            scored=3,
            status="completed",
            created_at="2025-03-02T00:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b2",
            "openai/gpt-4",
            "coding",
            "humaneval",
            started_at="2025-03-02T00:00:00+00:00",
        )
        _write_scoring(tmp_root, "b2", "q1", "q1", final_score=80.0)
        _write_scoring(tmp_root, "b2", "q2", "q2", final_score=85.0)
        _write_scoring(tmp_root, "b2", "q3", "q3", final_score=75.0)

        _write_status(
            tmp_root,
            "b3",
            "zai/glm-4.7",
            "reasoning",
            1,
            answered=0,
            scored=0,
            status="running",
            created_at="2025-03-03T00:00:00+00:00",
        )
        _write_metadata(
            tmp_root,
            "b3",
            "zai/glm-4.7",
            "reasoning",
            "math500",
            started_at="2025-03-03T00:00:00+00:00",
        )

        builder.build()

        lines = (
            (tmp_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        assert len(lines) == 3

        rows = [json.loads(l) for l in lines]
        by_id = {r["benchmark_id"]: r for r in rows}

        assert by_id["b1"]["avg_score"] == pytest.approx(80.0)
        assert by_id["b2"]["avg_score"] == pytest.approx(80.0)
        assert by_id["b3"]["avg_score"] is None

        glm_runs = builder.get_runs_by_model("zai/glm-4.7")
        assert len(glm_runs) == 2
        assert {r["benchmark_id"] for r in glm_runs} == {"b1", "b3"}

        reasoning_runs = builder.get_runs_by_dimension("reasoning")
        assert len(reasoning_runs) == 2

        recent = builder.get_recent_runs()
        assert recent[0]["benchmark_id"] == "b3"
