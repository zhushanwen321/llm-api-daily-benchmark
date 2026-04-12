from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.repository.handlers.execution_log_handler import ExecutionLogHandler


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> ExecutionLogHandler:
    return ExecutionLogHandler(data_root=tmp_root)


class TestAppendLog:
    def test_creates_execution_log_file(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        handler.append_log("bench1", "q1", "first log entry")
        assert (tmp_root / "bench1" / "q1" / "execution.log").exists()

    def test_creates_parent_directories(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        handler.append_log("bench1", "q1", "hello")
        assert (tmp_root / "bench1" / "q1").is_dir()

    def test_appends_multiple_entries(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        handler.append_log("bench1", "q1", "line 1")
        handler.append_log("bench1", "q1", "line 2")
        content = (tmp_root / "bench1" / "q1" / "execution.log").read_text(
            encoding="utf-8"
        )
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 2

    def test_log_format_contains_timestamp_and_level(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        handler.append_log("bench1", "q1", "test message")
        content = (tmp_root / "bench1" / "q1" / "execution.log").read_text(
            encoding="utf-8"
        )
        line = content.strip()
        assert " - " in line

    def test_log_contains_message(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        handler.append_log("bench1", "q1", "something happened")
        content = (tmp_root / "bench1" / "q1" / "execution.log").read_text(
            encoding="utf-8"
        )
        assert "something happened" in content

    def test_custom_level(self, handler: ExecutionLogHandler, tmp_root: Path) -> None:
        handler.append_log("bench1", "q1", "error occurred", level="ERROR")
        content = (tmp_root / "bench1" / "q1" / "execution.log").read_text(
            encoding="utf-8"
        )
        assert "ERROR" in content

    def test_default_level_is_info(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        handler.append_log("bench1", "q1", "info msg")
        content = (tmp_root / "bench1" / "q1" / "execution.log").read_text(
            encoding="utf-8"
        )
        assert "INFO" in content


class TestReadLog:
    def test_returns_full_content(self, handler: ExecutionLogHandler) -> None:
        handler.append_log("bench1", "q1", "line 1")
        handler.append_log("bench1", "q1", "line 2")
        content = handler.read_log("bench1", "q1")
        assert "line 1" in content
        assert "line 2" in content

    def test_nonexistent_raises_file_not_found(
        self, handler: ExecutionLogHandler
    ) -> None:
        with pytest.raises(FileNotFoundError):
            handler.read_log("nonexistent", "q1")

    def test_empty_log_returns_empty_string(
        self, handler: ExecutionLogHandler, tmp_root: Path
    ) -> None:
        log_path = tmp_root / "bench1" / "q1" / "execution.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        content = handler.read_log("bench1", "q1")
        assert content == ""


class TestRoundtrip:
    def test_write_and_read(self, handler: ExecutionLogHandler) -> None:
        handler.append_log("bench-rt", "q-rt", "entry A", level="WARNING")
        handler.append_log("bench-rt", "q-rt", "entry B", level="ERROR")
        content = handler.read_log("bench-rt", "q-rt")
        assert "entry A" in content
        assert "entry B" in content
        assert "WARNING" in content
        assert "ERROR" in content

    def test_different_benchmarks_are_isolated(
        self, handler: ExecutionLogHandler
    ) -> None:
        handler.append_log("bench1", "q1", "bench1 log")
        handler.append_log("bench2", "q1", "bench2 log")
        content1 = handler.read_log("bench1", "q1")
        content2 = handler.read_log("bench2", "q1")
        assert "bench1 log" in content1
        assert "bench1 log" not in content2
        assert "bench2 log" in content2
        assert "bench2 log" not in content1
