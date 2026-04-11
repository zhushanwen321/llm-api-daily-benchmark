from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchmark.analysis.models import ClusterInfo, ClusterReport
from benchmark.repository.handlers.cluster_handler import ClusterHandler


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> ClusterHandler:
    return ClusterHandler(data_root=tmp_root)


def _make_report(
    model: str = "zai/glm-4.7",
    n_clusters: int = 3,
    n_noise: int = 2,
    summary: str = "test summary",
) -> ClusterReport:
    return ClusterReport(
        model=model,
        n_clusters=n_clusters,
        n_noise=n_noise,
        clusters=[
            ClusterInfo(
                cluster_id=0,
                size=10,
                time_range=("2025-01-01", "2025-01-10"),
                centroid=[1.0, 2.0],
                avg_score=0.85,
            )
        ],
        suspected_changes=[{"signal": "accuracy", "direction": "decrease"}],
        summary=summary,
        created_at=datetime(2025, 1, 15, 12, 0, 0),
    )


class TestSaveReport:
    def test_creates_cluster_reports_jsonl(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        handler.save_report(_make_report())
        assert (tmp_root / "cluster_reports.jsonl").exists()

    def test_write_contains_valid_json_per_line(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        handler.save_report(_make_report())
        lines = (
            (tmp_root / "cluster_reports.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert isinstance(parsed, dict)

    def test_write_preserves_fields(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        handler.save_report(_make_report())
        data = json.loads(
            (tmp_root / "cluster_reports.jsonl").read_text(encoding="utf-8")
        )
        assert data["model"] == "zai/glm-4.7"
        assert data["n_clusters"] == 3
        assert data["n_noise"] == 2
        assert data["summary"] == "test summary"
        assert len(data["clusters"]) == 1
        assert data["clusters"][0]["cluster_id"] == 0
        assert len(data["suspected_changes"]) == 1
        assert "created_at" in data

    def test_write_appends_multiple_reports(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        handler.save_report(_make_report(model="model-a"))
        handler.save_report(_make_report(model="model-b"))
        lines = (
            (tmp_root / "cluster_reports.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2

    def test_clusters_serialized_as_list_of_dicts(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        handler.save_report(_make_report())
        data = json.loads(
            (tmp_root / "cluster_reports.jsonl").read_text(encoding="utf-8")
        )
        cluster = data["clusters"][0]
        assert cluster["size"] == 10
        assert cluster["time_range"] == ["2025-01-01", "2025-01-10"]
        assert cluster["centroid"] == [1.0, 2.0]
        assert cluster["avg_score"] == 0.85

    def test_created_at_serialized_as_iso_string(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        handler.save_report(_make_report())
        data = json.loads(
            (tmp_root / "cluster_reports.jsonl").read_text(encoding="utf-8")
        )
        assert isinstance(data["created_at"], str)

    def test_empty_clusters_and_changes(
        self, handler: ClusterHandler, tmp_root: Path
    ) -> None:
        report = ClusterReport(
            model="model-empty",
            n_clusters=0,
            n_noise=0,
        )
        handler.save_report(report)
        data = json.loads(
            (tmp_root / "cluster_reports.jsonl").read_text(encoding="utf-8")
        )
        assert data["clusters"] == []
        assert data["suspected_changes"] == []
        assert data["summary"] == ""


class TestGetReportsByModel:
    def test_returns_matching_reports(self, handler: ClusterHandler) -> None:
        handler.save_report(_make_report(model="model-a"))
        handler.save_report(_make_report(model="model-b"))
        handler.save_report(_make_report(model="model-a"))
        results = handler.get_reports_by_model("model-a")
        assert len(results) == 2
        assert all(r["model"] == "model-a" for r in results)

    def test_returns_empty_for_no_match(self, handler: ClusterHandler) -> None:
        handler.save_report(_make_report(model="model-a"))
        results = handler.get_reports_by_model("model-z")
        assert results == []

    def test_returns_empty_when_file_missing(self, handler: ClusterHandler) -> None:
        results = handler.get_reports_by_model("model-a")
        assert results == []

    def test_preserves_all_fields(self, handler: ClusterHandler) -> None:
        handler.save_report(_make_report())
        results = handler.get_reports_by_model("zai/glm-4.7")
        assert len(results) == 1
        r = results[0]
        assert r["n_clusters"] == 3
        assert r["n_noise"] == 2
        assert r["summary"] == "test summary"
        assert len(r["clusters"]) == 1
        assert len(r["suspected_changes"]) == 1

    def test_multiple_models_isolated(self, handler: ClusterHandler) -> None:
        handler.save_report(_make_report(model="model-a", n_clusters=2))
        handler.save_report(_make_report(model="model-b", n_clusters=5))
        a_reports = handler.get_reports_by_model("model-a")
        b_reports = handler.get_reports_by_model("model-b")
        assert len(a_reports) == 1
        assert len(b_reports) == 1
        assert a_reports[0]["n_clusters"] == 2
        assert b_reports[0]["n_clusters"] == 5
