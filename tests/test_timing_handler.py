from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.repository.handlers.timing_handler import TimingHandler


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> TimingHandler:
    return TimingHandler(data_root=tmp_root)


def _make_record(
    result_id: str = "r1",
    run_id: str = "run1",
    model: str = "gpt-4",
    task_id: str = "q1",
    phase_name: str = "api_call",
    start_time: float = 1000.0,
    end_time: float = 1001.5,
    duration: float = 1.5,
    wait_time: float = 0.3,
    active_time: float = 1.2,
    metadata: dict | None = None,
    created_at: str = "2025-01-01T00:00:00+00:00",
) -> dict:
    return {
        "result_id": result_id,
        "run_id": run_id,
        "model": model,
        "task_id": task_id,
        "phase_name": phase_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "wait_time": wait_time,
        "active_time": active_time,
        "metadata": metadata or {},
        "created_at": created_at,
    }


class TestSaveTiming:
    def test_creates_timing_jsonl(self, handler: TimingHandler, tmp_root: Path) -> None:
        record = _make_record()
        handler.save_timing("run1", "q1", [record])
        assert (tmp_root / "run1" / "q1" / "timing.jsonl").exists()

    def test_each_line_is_valid_json(
        self, handler: TimingHandler, tmp_root: Path
    ) -> None:
        records = [_make_record(phase_name="p1"), _make_record(phase_name="p2")]
        handler.save_timing("run1", "q1", records)
        lines = (
            (tmp_root / "run1" / "q1" / "timing.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_record_fields(self, handler: TimingHandler, tmp_root: Path) -> None:
        record = _make_record()
        handler.save_timing("run1", "q1", [record])
        line = (
            (tmp_root / "run1" / "q1" / "timing.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        parsed = json.loads(line)
        assert parsed["result_id"] == "r1"
        assert parsed["run_id"] == "run1"
        assert parsed["model"] == "gpt-4"
        assert parsed["task_id"] == "q1"
        assert parsed["phase_name"] == "api_call"
        assert parsed["duration"] == 1.5
        assert parsed["wait_time"] == 0.3
        assert parsed["active_time"] == 1.2

    def test_multiple_phases_append_lines(
        self, handler: TimingHandler, tmp_root: Path
    ) -> None:
        r1 = _make_record(phase_name="api_call", duration=1.0)
        r2 = _make_record(phase_name="scoring", duration=0.5)
        handler.save_timing("run1", "q1", [r1, r2])
        lines = (
            (tmp_root / "run1" / "q1" / "timing.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2
        assert json.loads(lines[0])["phase_name"] == "api_call"
        assert json.loads(lines[1])["phase_name"] == "scoring"

    def test_multiple_saves_append(
        self, handler: TimingHandler, tmp_root: Path
    ) -> None:
        handler.save_timing("run1", "q1", [_make_record(phase_name="p1")])
        handler.save_timing("run1", "q1", [_make_record(phase_name="p2")])
        lines = (
            (tmp_root / "run1" / "q1" / "timing.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2

    def test_empty_records_no_op(self, handler: TimingHandler, tmp_root: Path) -> None:
        handler.save_timing("run1", "q1", [])
        path = tmp_root / "run1" / "q1" / "timing.jsonl"
        assert not path.exists()


class TestGetTimingByQuestion:
    def test_returns_all_records(self, handler: TimingHandler) -> None:
        r1 = _make_record(phase_name="p1")
        r2 = _make_record(phase_name="p2")
        handler.save_timing("run1", "q1", [r1, r2])
        records = handler.get_timing_by_question("run1", "q1")
        assert len(records) == 2
        assert records[0]["phase_name"] == "p1"
        assert records[1]["phase_name"] == "p2"

    def test_nonexistent_raises_file_not_found(self, handler: TimingHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.get_timing_by_question("nonexistent", "q1")


class TestGetTimingByRun:
    def test_returns_all_records_across_questions(self, handler: TimingHandler) -> None:
        handler.save_timing("run1", "q1", [_make_record(phase_name="p1")])
        handler.save_timing("run1", "q2", [_make_record(phase_name="p2")])
        records = handler.get_timing_by_run("run1")
        assert len(records) == 2
        assert records[0]["question_id"] == "q1"
        assert records[1]["question_id"] == "q2"

    def test_returns_empty_for_nonexistent_run(self, handler: TimingHandler) -> None:
        records = handler.get_timing_by_run("nonexistent")
        assert records == []

    def test_includes_question_id(self, handler: TimingHandler) -> None:
        handler.save_timing("run1", "q1", [_make_record()])
        records = handler.get_timing_by_run("run1")
        assert records[0]["question_id"] == "q1"


class TestRoundtrip:
    def test_write_and_read_preserves_data(self, handler: TimingHandler) -> None:
        record = _make_record(
            result_id="r-rt",
            run_id="run-rt",
            model="test-model",
            task_id="q-rt",
            phase_name="eval",
            start_time=100.0,
            end_time=102.5,
            duration=2.5,
            wait_time=0.5,
            active_time=2.0,
            metadata={"key": "value"},
            created_at="2025-06-01T12:00:00+00:00",
        )
        handler.save_timing("run-rt", "q-rt", [record])
        records = handler.get_timing_by_question("run-rt", "q-rt")
        assert len(records) == 1
        r = records[0]
        assert r["result_id"] == "r-rt"
        assert r["phase_name"] == "eval"
        assert r["duration"] == 2.5
        assert r["metadata"] == {"key": "value"}
