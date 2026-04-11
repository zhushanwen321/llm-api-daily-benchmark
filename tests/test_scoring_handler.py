from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.repository.handlers.scoring_handler import ScoringHandler


QUALITY_SIGNALS_FIELDS = [
    "format_compliance",
    "repetition_ratio",
    "garbled_text_ratio",
    "refusal_detected",
    "language_consistency",
    "output_length_zscore",
    "thinking_ratio",
    "empty_reasoning",
    "truncated",
    "token_efficiency_zscore",
    "tps_zscore",
    "ttft_zscore",
    "answer_entropy",
    "raw_output_length",
]


def _make_quality_signals(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "format_compliance": 1.0,
        "repetition_ratio": 0.05,
        "garbled_text_ratio": 0.0,
        "refusal_detected": 0,
        "language_consistency": 0.98,
        "output_length_zscore": 0.3,
        "thinking_ratio": 0.4,
        "empty_reasoning": 0,
        "truncated": 0,
        "token_efficiency_zscore": -0.1,
        "tps_zscore": 0.5,
        "ttft_zscore": -0.2,
        "answer_entropy": 2.7,
        "raw_output_length": 150,
    }
    base.update(overrides)
    return base


def _make_scoring_data(
    task_id: str = "q1",
    functional_score: float = 100.0,
    quality_score: float = 90.0,
    final_score: float = 95.0,
    passed: bool = True,
    details: dict[str, Any] | None = None,
    reasoning: str = "correct answer",
    quality_signals: dict[str, Any] | None = None,
    scoring_status: str = "completed",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "functional_score": functional_score,
        "quality_score": quality_score,
        "final_score": final_score,
        "passed": passed,
        "details": details or {"match": "exact"},
        "reasoning": reasoning,
        "quality_signals": quality_signals or _make_quality_signals(),
        "scoring_status": scoring_status,
    }


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> ScoringHandler:
    return ScoringHandler(data_root=tmp_root)


class TestSaveScoring:
    def test_creates_scoring_jsonl(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        assert (tmp_root / "b1" / "q1" / "scoring.jsonl").exists()

    def test_each_line_is_valid_json(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        lines = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert isinstance(parsed, dict)

    def test_contains_all_scoring_fields(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        line = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().strip()
        parsed = json.loads(line)
        for key in (
            "task_id",
            "functional_score",
            "quality_score",
            "final_score",
            "passed",
            "details",
            "reasoning",
        ):
            assert key in parsed, f"missing key: {key}"

    def test_contains_quality_signals(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        line = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().strip()
        parsed = json.loads(line)
        qs = parsed["quality_signals"]
        for field in QUALITY_SIGNALS_FIELDS:
            assert field in qs, f"missing quality_signals field: {field}"

    def test_contains_scoring_status(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        line = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["scoring_status"] == "completed"

    def test_multiple_saves_append(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data1 = _make_scoring_data(task_id="q1", functional_score=80.0)
        data2 = _make_scoring_data(task_id="q1", functional_score=100.0)
        handler.save_scoring("b1", "q1", data1)
        handler.save_scoring("b1", "q1", data2)
        lines = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["functional_score"] == 80.0
        assert json.loads(lines[1])["functional_score"] == 100.0

    def test_default_scoring_status_is_completed(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        del data["scoring_status"]
        handler.save_scoring("b1", "q1", data)
        line = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().strip()
        parsed = json.loads(line)
        assert parsed["scoring_status"] == "completed"


class TestGetScoring:
    def test_returns_latest_scoring(self, handler: ScoringHandler) -> None:
        data1 = _make_scoring_data(functional_score=50.0)
        data2 = _make_scoring_data(functional_score=100.0)
        handler.save_scoring("b1", "q1", data1)
        handler.save_scoring("b1", "q1", data2)
        result = handler.get_scoring("b1", "q1")
        assert result["functional_score"] == 100.0

    def test_returns_all_fields(self, handler: ScoringHandler) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        result = handler.get_scoring("b1", "q1")
        assert result["task_id"] == "q1"
        assert result["functional_score"] == 100.0
        assert result["quality_score"] == 90.0
        assert result["final_score"] == 95.0
        assert result["passed"] is True
        assert "quality_signals" in result
        assert result["scoring_status"] == "completed"

    def test_nonexistent_raises_file_not_found(self, handler: ScoringHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.get_scoring("nonexistent", "q1")

    def test_empty_file_raises_file_not_found(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        path = tmp_root / "b1" / "q1" / "scoring.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        with pytest.raises(FileNotFoundError):
            handler.get_scoring("b1", "q1")


class TestMarkPending:
    def test_marks_status_as_pending(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        handler.mark_pending("b1", "q1")
        result = handler.get_scoring("b1", "q1")
        assert result["scoring_status"] == "pending"

    def test_rewrites_file_atomically(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data()
        handler.save_scoring("b1", "q1", data)
        handler.mark_pending("b1", "q1")
        lines = (tmp_root / "b1" / "q1" / "scoring.jsonl").read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["scoring_status"] == "pending"

    def test_preserves_other_fields(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data(functional_score=75.0, reasoning="partial match")
        handler.save_scoring("b1", "q1", data)
        handler.mark_pending("b1", "q1")
        result = handler.get_scoring("b1", "q1")
        assert result["functional_score"] == 75.0
        assert result["reasoning"] == "partial match"

    def test_nonexistent_raises_file_not_found(self, handler: ScoringHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.mark_pending("nonexistent", "q1")


class TestFindPending:
    def test_finds_pending_records(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data_pending = _make_scoring_data(scoring_status="pending")
        data_completed = _make_scoring_data(scoring_status="completed")
        handler.save_scoring("b1", "q1", data_pending)
        handler.save_scoring("b1", "q2", data_completed)
        handler.save_scoring("b1", "q3", data_pending)
        pending = handler.find_pending("b1")
        task_ids = {r["question_id"] for r in pending}
        assert task_ids == {"q1", "q3"}

    def test_returns_empty_when_all_completed(self, handler: ScoringHandler) -> None:
        data = _make_scoring_data(scoring_status="completed")
        handler.save_scoring("b1", "q1", data)
        handler.save_scoring("b1", "q2", data)
        assert handler.find_pending("b1") == []

    def test_returns_empty_for_nonexistent_benchmark(
        self, handler: ScoringHandler
    ) -> None:
        assert handler.find_pending("nonexistent") == []

    def test_result_includes_benchmark_and_question_id(
        self, handler: ScoringHandler
    ) -> None:
        data = _make_scoring_data(scoring_status="pending")
        handler.save_scoring("b1", "q1", data)
        pending = handler.find_pending("b1")
        assert len(pending) == 1
        assert pending[0]["benchmark_id"] == "b1"
        assert pending[0]["question_id"] == "q1"

    def test_finds_pending_in_latest_line_only(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        handler.save_scoring("b1", "q1", _make_scoring_data(scoring_status="pending"))
        handler.save_scoring("b1", "q1", _make_scoring_data(scoring_status="completed"))
        pending = handler.find_pending("b1")
        assert len(pending) == 0

    def test_finds_pending_when_latest_is_pending(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        handler.save_scoring("b1", "q1", _make_scoring_data(scoring_status="completed"))
        handler.save_scoring("b1", "q1", _make_scoring_data(scoring_status="pending"))
        pending = handler.find_pending("b1")
        assert len(pending) == 1
        assert pending[0]["question_id"] == "q1"

    def test_skips_non_directory_entries(
        self, handler: ScoringHandler, tmp_root: Path
    ) -> None:
        data = _make_scoring_data(scoring_status="pending")
        handler.save_scoring("b1", "q1", data)
        (tmp_root / "b1" / "some_file.txt").write_text("not a directory")
        pending = handler.find_pending("b1")
        assert len(pending) == 1


class TestRoundtrip:
    def test_write_and_read_preserves_data(self, handler: ScoringHandler) -> None:
        qs = _make_quality_signals(
            format_compliance=0.85,
            repetition_ratio=0.12,
            raw_output_length=300,
        )
        data = _make_scoring_data(
            task_id="q-rt",
            functional_score=70.0,
            quality_score=85.0,
            final_score=77.5,
            passed=False,
            details={"match": "partial", "expected": "42", "got": "41"},
            reasoning="off by one",
            quality_signals=qs,
        )
        handler.save_scoring("b-rt", "q-rt", data)
        result = handler.get_scoring("b-rt", "q-rt")
        assert result["task_id"] == "q-rt"
        assert result["functional_score"] == 70.0
        assert result["final_score"] == 77.5
        assert result["passed"] is False
        assert result["reasoning"] == "off by one"
        assert result["quality_signals"]["format_compliance"] == 0.85
        assert result["quality_signals"]["raw_output_length"] == 300
