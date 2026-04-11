from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.repository.handlers.metadata_handler import FileAppend


class TimingHandler:
    """管理 {benchmark_id}/{question_id}/timing.jsonl 的读写。

    每行一条 JSON 记录，字段包括：
      result_id, run_id, model, task_id, phase_name,
      start_time, end_time, duration, wait_time, active_time,
      metadata, created_at
    """

    FILENAME = "timing.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, benchmark_id: str, question_id: str) -> Path:
        return self._root / benchmark_id / question_id / self.FILENAME

    def save_timing(
        self,
        benchmark_id: str,
        question_id: str,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            return
        path = self._path(benchmark_id, question_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileAppend(str(path)) as f:
            for record in records:
                line = json.dumps(record, ensure_ascii=False) + "\n"
                f.write(line)

    def get_timing_by_question(
        self, benchmark_id: str, question_id: str
    ) -> list[dict[str, Any]]:
        path = self._path(benchmark_id, question_id)
        if not path.exists():
            raise FileNotFoundError(
                f"timing.jsonl not found for {benchmark_id}/{question_id}"
            )
        records = self._read_all_lines(path)
        if not records:
            raise FileNotFoundError(
                f"timing.jsonl is empty for {benchmark_id}/{question_id}"
            )
        return records

    def get_timing_by_run(self, benchmark_id: str) -> list[dict[str, Any]]:
        bench_dir = self._root / benchmark_id
        if not bench_dir.exists():
            return []

        all_records: list[dict[str, Any]] = []
        for question_dir in sorted(bench_dir.iterdir()):
            if not question_dir.is_dir():
                continue
            timing_path = question_dir / self.FILENAME
            if not timing_path.exists():
                continue
            records = self._read_all_lines(timing_path)
            for r in records:
                r["question_id"] = question_dir.name
            all_records.extend(records)
        return all_records

    @staticmethod
    def _read_all_lines(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
