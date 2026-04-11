from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.repository.handlers.metadata_handler import MetadataHandler


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> MetadataHandler:
    return MetadataHandler(data_root=tmp_root)


class TestWrite:
    def test_creates_metadata_jsonl(
        self, handler: MetadataHandler, tmp_root: Path
    ) -> None:
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
        )
        assert (tmp_root / "b1" / "metadata.jsonl").exists()

    def test_write_contains_valid_json_per_line(
        self, handler: MetadataHandler, tmp_root: Path
    ) -> None:
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
        )
        lines = (
            (tmp_root / "b1" / "metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert isinstance(parsed, dict)

    def test_write_appends_multiple_records(
        self, handler: MetadataHandler, tmp_root: Path
    ) -> None:
        handler.write(
            benchmark_id="b1", model="model-a", dimension="reasoning", dataset="gsm8k"
        )
        handler.write(
            benchmark_id="b1", model="model-b", dimension="coding", dataset="humaneval"
        )
        lines = (
            (tmp_root / "b1" / "metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2

    def test_custom_started_at(self, handler: MetadataHandler) -> None:
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
            started_at="2025-01-01T00:00:00+00:00",
        )
        data = handler.read("b1")
        assert data["started_at"] == "2025-01-01T00:00:00+00:00"

    def test_config_snapshot_default(self, handler: MetadataHandler) -> None:
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
        )
        data = handler.read("b1")
        assert data["config_snapshot"] == "{}"

    def test_config_snapshot_custom(self, handler: MetadataHandler) -> None:
        config = json.dumps({"temperature": 0.7, "max_tokens": 4096})
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
            config_snapshot=config,
        )
        data = handler.read("b1")
        assert json.loads(data["config_snapshot"])["temperature"] == 0.7


class TestRead:
    def test_read_returns_last_record(self, handler: MetadataHandler) -> None:
        handler.write(
            benchmark_id="b1", model="model-a", dimension="reasoning", dataset="gsm8k"
        )
        handler.write(
            benchmark_id="b1", model="model-b", dimension="coding", dataset="humaneval"
        )
        data = handler.read("b1")
        assert data["model"] == "model-b"
        assert data["dimension"] == "coding"

    def test_read_matches_write(self, handler: MetadataHandler) -> None:
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
        )
        data = handler.read("b1")
        assert data["benchmark_id"] == "b1"
        assert data["model"] == "zai/glm-4.7"
        assert data["dimension"] == "reasoning"
        assert data["dataset"] == "gsm8k"
        assert "started_at" in data
        assert "config_snapshot" in data

    def test_read_nonexistent_raises(self, handler: MetadataHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.read("nonexistent-id")


class TestReadAll:
    def test_returns_all_records(self, handler: MetadataHandler) -> None:
        handler.write(
            benchmark_id="b1", model="model-a", dimension="reasoning", dataset="gsm8k"
        )
        handler.write(
            benchmark_id="b1", model="model-b", dimension="coding", dataset="humaneval"
        )
        handler.write(
            benchmark_id="b1", model="model-c", dimension="math", dataset="math500"
        )
        records = handler.read_all("b1")
        assert len(records) == 3
        assert records[0]["model"] == "model-a"
        assert records[1]["model"] == "model-b"
        assert records[2]["model"] == "model-c"

    def test_returns_empty_for_nonexistent(self, handler: MetadataHandler) -> None:
        assert handler.read_all("nonexistent") == []


class TestQAScenario:
    def test_write_and_read_roundtrip(self, handler: MetadataHandler) -> None:
        handler.write(
            benchmark_id="b1",
            model="zai/glm-4.7",
            dimension="reasoning",
            dataset="gsm8k",
        )
        data = handler.read("b1")
        assert data["model"] == "zai/glm-4.7"
        assert data["dimension"] == "reasoning"
        assert data["dataset"] == "gsm8k"
