from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.repository.handlers.analysis_handler import AnalysisHandler


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> AnalysisHandler:
    return AnalysisHandler(data_root=tmp_root)


def _make_report(
    model: str = "zai/glm-4.7",
    benchmark_id: str = "bench-001",
    overall_status: str = "stable",
    anomalies: list | None = None,
    change_points: list | None = None,
    stat_tests: list | None = None,
    summary: str = "",
    created_at: str | None = None,
) -> dict:
    return {
        "report_type": "stability",
        "model": model,
        "benchmark_id": benchmark_id,
        "overall_status": overall_status,
        "anomalies": anomalies or [],
        "change_points": change_points or [],
        "stat_tests": stat_tests or [],
        "summary": summary,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


class TestSaveAnalysis:
    def test_creates_jsonl_file(self, handler: AnalysisHandler, tmp_root: Path) -> None:
        report = _make_report()
        handler.save_analysis("bench-001", report)
        assert (tmp_root / "bench-001" / "analysis.jsonl").exists()

    def test_returns_report_id(self, handler: AnalysisHandler) -> None:
        report = _make_report()
        result_id = handler.save_analysis("bench-001", report)
        assert isinstance(result_id, str)
        assert len(result_id) > 0

    def test_write_valid_jsonl(self, handler: AnalysisHandler, tmp_root: Path) -> None:
        report = _make_report()
        handler.save_analysis("bench-001", report)
        content = (tmp_root / "bench-001" / "analysis.jsonl").read_text(
            encoding="utf-8"
        )
        for line in content.strip().splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_record_has_required_fields(self, handler: AnalysisHandler) -> None:
        report = _make_report()
        handler.save_analysis("bench-001", report)
        records = handler.get_analysis("bench-001")
        assert len(records) == 1
        rec = records[0]
        assert rec["report_type"] == "stability"
        assert rec["model"] == "zai/glm-4.7"
        assert rec["benchmark_id"] == "bench-001"
        assert rec["overall_status"] == "stable"
        assert rec["anomalies"] == []
        assert rec["change_points"] == []
        assert rec["stat_tests"] == []
        assert rec["summary"] == ""
        assert "created_at" in rec

    def test_appends_multiple_records(self, handler: AnalysisHandler) -> None:
        report1 = _make_report(overall_status="stable", summary="first")
        report2 = _make_report(overall_status="degraded", summary="second")
        handler.save_analysis("bench-001", report1)
        handler.save_analysis("bench-001", report2)
        records = handler.get_analysis("bench-001")
        assert len(records) == 2
        assert records[0]["summary"] == "first"
        assert records[1]["summary"] == "second"
        assert records[1]["overall_status"] == "degraded"

    def test_preserves_complex_fields(self, handler: AnalysisHandler) -> None:
        anomalies = [
            {
                "signal_name": "latency_p95",
                "current_value": 5.2,
                "baseline_mean": 2.1,
                "baseline_std": 0.5,
                "z_score": 6.2,
            }
        ]
        change_points = [
            {
                "signal_name": "score",
                "detected_at": "2025-04-11T10:00:00+00:00",
                "direction": "decrease",
                "magnitude": 0.15,
            }
        ]
        stat_tests = [{"test": "mann_whitney", "p_value": 0.03, "significant": True}]
        report = _make_report(
            overall_status="suspicious",
            anomalies=anomalies,
            change_points=change_points,
            stat_tests=stat_tests,
            summary="detected anomalies",
        )
        handler.save_analysis("bench-001", report)
        records = handler.get_analysis("bench-001")
        rec = records[0]
        assert rec["overall_status"] == "suspicious"
        assert len(rec["anomalies"]) == 1
        assert rec["anomalies"][0]["z_score"] == 6.2
        assert len(rec["change_points"]) == 1
        assert rec["change_points"][0]["direction"] == "decrease"
        assert len(rec["stat_tests"]) == 1
        assert rec["stat_tests"][0]["significant"] is True
        assert rec["summary"] == "detected anomalies"


class TestGetAnalysis:
    def test_returns_empty_when_no_file(self, handler: AnalysisHandler) -> None:
        records = handler.get_analysis("nonexistent")
        assert records == []

    def test_returns_all_records(self, handler: AnalysisHandler) -> None:
        for i in range(3):
            report = _make_report(summary=f"report-{i}")
            handler.save_analysis("bench-001", report)
        records = handler.get_analysis("bench-001")
        assert len(records) == 3

    def test_skips_empty_lines(self, handler: AnalysisHandler, tmp_root: Path) -> None:
        report = _make_report()
        handler.save_analysis("bench-001", report)
        path = tmp_root / "bench-001" / "analysis.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n")
        records = handler.get_analysis("bench-001")
        assert len(records) == 1


class TestGetLatest:
    def test_returns_latest_record(self, handler: AnalysisHandler) -> None:
        report1 = _make_report(overall_status="stable", summary="old")
        report2 = _make_report(overall_status="degraded", summary="new")
        handler.save_analysis("bench-001", report1)
        handler.save_analysis("bench-001", report2)
        latest = handler.get_latest("bench-001")
        assert latest["summary"] == "new"
        assert latest["overall_status"] == "degraded"

    def test_raises_when_no_file(self, handler: AnalysisHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.get_latest("nonexistent")

    def test_raises_when_empty_file(
        self, handler: AnalysisHandler, tmp_root: Path
    ) -> None:
        path = tmp_root / "bench-001" / "analysis.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            handler.get_latest("bench-001")
