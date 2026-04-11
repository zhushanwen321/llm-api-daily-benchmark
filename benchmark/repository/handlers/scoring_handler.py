from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.repository.atomic_write import atomic_write
from benchmark.repository.handlers.metadata_handler import FileAppend


class ScoringHandler:
    """管理 {benchmark_id}/{question_id}/scoring.jsonl 的读写。

    每行一条 JSON 记录，包含评分结果、quality_signals 和 scoring_status。
    """

    FILENAME = "scoring.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, benchmark_id: str, question_id: str) -> Path:
        return self._root / benchmark_id / question_id / self.FILENAME

    def save_scoring(
        self,
        benchmark_id: str,
        question_id: str,
        data: dict[str, Any],
    ) -> None:
        record: dict[str, Any] = {
            "task_id": data["task_id"],
            "functional_score": data["functional_score"],
            "quality_score": data["quality_score"],
            "final_score": data["final_score"],
            "passed": data["passed"],
            "details": data.get("details", {}),
            "reasoning": data.get("reasoning", ""),
            "quality_signals": data.get("quality_signals", {}),
            "scoring_status": data.get("scoring_status", "completed"),
        }

        path = self._path(benchmark_id, question_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with FileAppend(str(path)) as f:
            f.write(line)

    def get_scoring(self, benchmark_id: str, question_id: str) -> dict[str, Any]:
        path = self._path(benchmark_id, question_id)
        if not path.exists():
            raise FileNotFoundError(
                f"scoring.jsonl not found for {benchmark_id}/{question_id}"
            )
        records = self._read_all_lines(path)
        if not records:
            raise FileNotFoundError(
                f"scoring.jsonl is empty for {benchmark_id}/{question_id}"
            )
        return records[-1]

    def mark_pending(self, benchmark_id: str, question_id: str) -> None:
        records = self._read_all_lines(self._path(benchmark_id, question_id))
        if not records:
            raise FileNotFoundError(
                f"scoring.jsonl is empty for {benchmark_id}/{question_id}"
            )
        records[-1]["scoring_status"] = "pending"
        self._rewrite(benchmark_id, question_id, records)

    def find_pending(self, benchmark_id: str) -> list[dict[str, Any]]:
        bench_dir = self._root / benchmark_id
        if not bench_dir.exists():
            return []

        pending: list[dict[str, Any]] = []
        for question_dir in sorted(bench_dir.iterdir()):
            if not question_dir.is_dir():
                continue
            scoring_path = question_dir / self.FILENAME
            if not scoring_path.exists():
                continue
            records = self._read_all_lines(scoring_path)
            if records and records[-1].get("scoring_status") == "pending":
                pending.append(
                    {
                        "benchmark_id": benchmark_id,
                        "question_id": question_dir.name,
                        **records[-1],
                    }
                )
        return pending

    def _rewrite(
        self,
        benchmark_id: str,
        question_id: str,
        records: list[dict[str, Any]],
    ) -> None:
        path = self._path(benchmark_id, question_id)
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        atomic_write(str(path), content)

    @staticmethod
    def _read_all_lines(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
