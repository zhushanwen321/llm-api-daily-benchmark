from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.repository.atomic_write import atomic_write


class IndexBuilder:
    """扫描 data/{execution_id}/{run_id}/ 两级目录，构建 data/index.jsonl。

    每行包含：benchmark_id, model, dimension, dataset, total_questions,
    answered, scored, status, avg_score, created_at
    """

    INDEX_FILENAME = "index.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root
        self._index_path = data_root / self.INDEX_FILENAME

    def build(self) -> None:
        rows: list[dict[str, Any]] = []
        for exec_entry in sorted(self._root.iterdir()):
            if not exec_entry.is_dir():
                continue
            for run_entry in sorted(exec_entry.iterdir()):
                if not run_entry.is_dir():
                    continue
                status_path = run_entry / "status.json"
                if not status_path.exists():
                    continue
                row = self._build_row(run_entry.name, status_path)
                if row is not None:
                    rows.append(row)
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        atomic_write(str(self._index_path), content)

    def get_runs_by_model(self, model: str) -> list[dict[str, Any]]:
        self._ensure_index()
        return [r for r in self._read_index() if r.get("model") == model]

    def get_runs_by_dimension(self, dimension: str) -> list[dict[str, Any]]:
        self._ensure_index()
        return [r for r in self._read_index() if r.get("dimension") == dimension]

    def get_recent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        self._ensure_index()
        rows = self._read_index()
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]

    def _build_row(self, benchmark_id: str, status_path: Path) -> dict[str, Any] | None:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        metadata = self._read_metadata(benchmark_id)
        avg_score = self._calc_avg_score(benchmark_id)

        return {
            "benchmark_id": benchmark_id,
            "model": status.get("model", ""),
            "dimension": status.get("dimension", ""),
            "dataset": metadata.get("dataset", "") if metadata else "",
            "total_questions": status.get("total_questions", 0),
            "answered": status.get("answered", 0),
            "scored": status.get("scored", 0),
            "status": status.get("status", "unknown"),
            "avg_score": avg_score,
            "created_at": status.get("created_at", ""),
        }

    def _read_metadata(self, benchmark_id: str) -> dict[str, Any] | None:
        meta_path = self._find_run_dir(benchmark_id) / "metadata.jsonl"
        if not meta_path.exists():
            return None
        lines = meta_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    def _calc_avg_score(self, benchmark_id: str) -> float | None:
        bench_dir = self._find_run_dir(benchmark_id)
        if not bench_dir.exists():
            return None
        scores: list[float] = []
        for question_dir in bench_dir.iterdir():
            if not question_dir.is_dir():
                continue
            scoring_path = question_dir / "scoring.jsonl"
            if not scoring_path.exists():
                continue
            for line in reversed(scoring_path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("scoring_status") == "completed":
                    final = record.get("final_score")
                    if isinstance(final, (int, float)):
                        scores.append(float(final))
                break
        return sum(scores) / len(scores) if scores else None

    def _find_run_dir(self, benchmark_id: str) -> Path:
        """在所有 execution 目录下查找指定的 run_id 目录。"""
        for exec_entry in self._root.iterdir():
            if not exec_entry.is_dir():
                continue
            candidate = exec_entry / benchmark_id
            if candidate.is_dir():
                return candidate
        return self._root / benchmark_id

    def _ensure_index(self) -> None:
        if not self._index_path.exists():
            self.build()

    def _read_index(self) -> list[dict[str, Any]]:
        if not self._index_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
