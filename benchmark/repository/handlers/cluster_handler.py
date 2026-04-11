from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.analysis.models import ClusterReport
from benchmark.repository.handlers.metadata_handler import FileAppend


class ClusterHandler:
    """管理 data/cluster_reports.jsonl（全局聚类报告）的读写。

    每行一条 JSON 记录，字段包括：
      model, n_clusters, n_noise, clusters, suspected_changes, summary, created_at
    """

    FILENAME = "cluster_reports.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    @property
    def _path(self) -> Path:
        return self._root / self.FILENAME

    def save_report(self, report: ClusterReport) -> None:
        record = self._serialize(report)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with FileAppend(str(self._path)) as f:
            f.write(line)

    def get_reports_by_model(self, model: str) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        results: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("model") == model:
                results.append(record)
        return results

    @staticmethod
    def _serialize(report: ClusterReport) -> dict[str, Any]:
        created_at = report.created_at
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        return {
            "model": report.model,
            "n_clusters": report.n_clusters,
            "n_noise": report.n_noise,
            "clusters": [asdict(c) for c in report.clusters],
            "suspected_changes": report.suspected_changes,
            "summary": report.summary,
            "created_at": created_at,
        }
