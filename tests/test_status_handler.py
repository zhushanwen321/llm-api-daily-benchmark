from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.repository.handlers.status_handler import StatusHandler


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> StatusHandler:
    return StatusHandler(data_root=tmp_root)


class TestCreate:
    def test_creates_status_json(self, handler: StatusHandler, tmp_root: Path) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=15,
        )
        status_path = tmp_root / "test-123" / "status.json"
        assert status_path.exists()

    def test_initial_values(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=15,
        )
        data = handler.get("test-123")
        assert data["benchmark_id"] == "test-123"
        assert data["model"] == "zai/glm-4.7"
        assert data["dimension"] == "reasoning"
        assert data["total_questions"] == 15
        assert data["answered"] == 0
        assert data["scored"] == 0
        assert data["status"] == "running"
        assert data["created_at"] is not None
        assert data["updated_at"] is not None

    def test_valid_json(self, handler: StatusHandler, tmp_root: Path) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=15,
        )
        raw = (tmp_root / "test-123" / "status.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


class TestIncrementAnswered:
    def test_increments_answered(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=15,
        )
        handler.increment_answered("test-123")
        data = handler.get("test-123")
        assert data["answered"] == 1
        assert data["status"] == "running"

    def test_transitions_to_scoring(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        for _ in range(3):
            handler.increment_answered("test-123")
        data = handler.get("test-123")
        assert data["answered"] == 3
        assert data["status"] == "scoring"

    def test_updates_updated_at(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=15,
        )
        before = handler.get("test-123")["updated_at"]
        handler.increment_answered("test-123")
        after = handler.get("test-123")["updated_at"]
        assert after >= before


class TestIncrementScored:
    def test_increments_scored(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        for _ in range(3):
            handler.increment_answered("test-123")
        handler.increment_scored("test-123")
        data = handler.get("test-123")
        assert data["scored"] == 1

    def test_transitions_to_completed(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        for _ in range(3):
            handler.increment_answered("test-123")
        for _ in range(3):
            handler.increment_scored("test-123")
        data = handler.get("test-123")
        assert data["scored"] == 3
        assert data["status"] == "completed"


class TestIsCompleted:
    def test_not_completed_initially(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        assert handler.is_completed("test-123") is False

    def test_completed_after_all_scored(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        for _ in range(3):
            handler.increment_answered("test-123")
        for _ in range(3):
            handler.increment_scored("test-123")
        assert handler.is_completed("test-123") is True


class TestGetNonexistent:
    def test_raises_on_missing(self, handler: StatusHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.get("nonexistent-id")


class TestFullLifecycle:
    def test_running_to_scoring_to_completed(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=15,
        )
        data = handler.get("test-123")
        assert data["status"] == "running"
        assert data["answered"] == 0
        assert data["scored"] == 0

        for _ in range(15):
            handler.increment_answered("test-123")
        data = handler.get("test-123")
        assert data["status"] == "scoring"
        assert data["answered"] == 15
        assert data["scored"] == 0

        for _ in range(15):
            handler.increment_scored("test-123")
        data = handler.get("test-123")
        assert data["status"] == "completed"
        assert data["scored"] == 15

        assert handler.is_completed("test-123") is True


class TestFailedStatus:
    def test_set_failed(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        handler.set_failed("test-123")
        data = handler.get("test-123")
        assert data["status"] == "failed"

    def test_is_not_completed_when_failed(self, handler: StatusHandler) -> None:
        handler.create(
            benchmark_id="test-123",
            model="zai/glm-4.7",
            dimension="reasoning",
            total_questions=3,
        )
        handler.set_failed("test-123")
        assert handler.is_completed("test-123") is False
