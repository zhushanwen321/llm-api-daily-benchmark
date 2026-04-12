from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from benchmark.core.tz import now
from benchmark.repository.handlers.metadata_handler import FileAppend


class AnalysisHandler:
    """管理 {benchmark_id}/analysis.jsonl 的读写。

    每行一条 JSON 记录，字段包括：
      report_type, model, benchmark_id, overall_status,
      anomalies, change_points, stat_tests, summary, created_at
    """

    FILENAME = "analysis.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, benchmark_id: str) -> Path:
        return self._root / benchmark_id / self.FILENAME

    @staticmethod
    def _now() -> str:
        return now().isoformat()

    def save_analysis(
        self,
        benchmark_id: str,
        report: dict[str, Any],
    ) -> str:
        report_id = report.get("report_id") or uuid.uuid4().hex[:12]

        record: dict[str, Any] = {
            "report_id": report_id,
            "report_type": report.get("report_type", "stability"),
            "model": report["model"],
            "benchmark_id": benchmark_id,
            "overall_status": report["overall_status"],
            "anomalies": report.get("anomalies", []),
            "change_points": report.get("change_points", []),
            "stat_tests": report.get("stat_tests", []),
            "summary": report.get("summary", ""),
            "created_at": report.get("created_at") or self._now(),
        }

        path = self._path(benchmark_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with FileAppend(str(path)) as f:
            f.write(line)

        return report_id

    def get_analysis(self, benchmark_id: str) -> list[dict[str, Any]]:
        path = self._path(benchmark_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def get_latest(self, benchmark_id: str) -> dict[str, Any]:
        records = self.get_analysis(benchmark_id)
        if not records:
            raise FileNotFoundError(
                f"analysis.jsonl not found or empty for benchmark_id={benchmark_id}"
            )
        return records[-1]
