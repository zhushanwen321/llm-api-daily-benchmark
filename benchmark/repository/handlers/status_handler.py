from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.core.tz import now
from benchmark.repository.atomic_write import atomic_write


class StatusHandler:
    """管理 {benchmark_id}/status.json 的读写。"""

    FILENAME = "status.json"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, benchmark_id: str) -> Path:
        return self._root / benchmark_id / self.FILENAME

    @staticmethod
    def _now() -> str:
        return now().isoformat()

    def create(
        self,
        benchmark_id: str,
        model: str,
        dimension: str,
        total_questions: int,
    ) -> None:
        now = self._now()
        data: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "model": model,
            "dimension": dimension,
            "total_questions": total_questions,
            "answered": 0,
            "scored": 0,
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }
        self._write(benchmark_id, data)

    def increment_answered(self, benchmark_id: str) -> None:
        data = self.get(benchmark_id)
        data["answered"] += 1
        data["updated_at"] = self._now()
        if data["total_questions"] > 0 and data["answered"] >= data["total_questions"]:
            data["status"] = "scoring"
        self._write(benchmark_id, data)

    def increment_scored(self, benchmark_id: str) -> None:
        data = self.get(benchmark_id)
        data["scored"] += 1
        data["updated_at"] = self._now()
        if data["total_questions"] > 0 and data["scored"] >= data["total_questions"]:
            data["status"] = "completed"
        self._write(benchmark_id, data)

    def get(self, benchmark_id: str) -> dict[str, Any]:
        path = self._path(benchmark_id)
        if not path.exists():
            raise FileNotFoundError(
                f"status.json not found for benchmark_id={benchmark_id}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def is_completed(self, benchmark_id: str) -> bool:
        data = self.get(benchmark_id)
        return data["status"] == "completed"

    def set_failed(self, benchmark_id: str) -> None:
        data = self.get(benchmark_id)
        data["status"] = "failed"
        data["updated_at"] = self._now()
        self._write(benchmark_id, data)

    def set_completed(self, benchmark_id: str) -> None:
        data = self.get(benchmark_id)
        data["status"] = "completed"
        data["updated_at"] = self._now()
        self._write(benchmark_id, data)

    def _write(self, benchmark_id: str, data: dict[str, Any]) -> None:
        path = self._path(benchmark_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(str(path), json.dumps(data, ensure_ascii=False, indent=2))
