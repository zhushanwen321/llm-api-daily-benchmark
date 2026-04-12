from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.core.tz import now
from benchmark.repository.atomic_write import atomic_write


class MetadataHandler:
    """管理 {benchmark_id}/metadata.jsonl 的读写。

    每行一条 JSON 记录，字段包括：
      benchmark_id, model, dimension, dataset, started_at, config_snapshot
    """

    FILENAME = "metadata.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, benchmark_id: str) -> Path:
        return self._root / benchmark_id / self.FILENAME

    @staticmethod
    def _now() -> str:
        return now().isoformat()

    def write(
        self,
        benchmark_id: str,
        model: str,
        dimension: str,
        dataset: str,
        config_snapshot: str = "{}",
        started_at: str | None = None,
    ) -> None:
        if started_at is None:
            started_at = self._now()

        record: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "model": model,
            "dimension": dimension,
            "dataset": dataset,
            "started_at": started_at,
            "config_snapshot": config_snapshot,
        }

        path = self._path(benchmark_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with FileAppend(str(path)) as f:
            f.write(line)

    def read(self, benchmark_id: str) -> dict[str, Any]:
        path = self._path(benchmark_id)
        if not path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found for benchmark_id={benchmark_id}"
            )
        records = self.read_all(benchmark_id)
        if not records:
            raise FileNotFoundError(
                f"metadata.jsonl is empty for benchmark_id={benchmark_id}"
            )
        return records[-1]

    def read_all(self, benchmark_id: str) -> list[dict[str, Any]]:
        path = self._path(benchmark_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


class FileAppend:
    """对单个文件追加写入的简易锁保护。"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._f: Any = None

    def __enter__(self) -> FileAppend:
        import fcntl

        self._f = open(self._path, "a", encoding="utf-8")
        fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *args: Any) -> None:
        import fcntl

        if self._f is not None:
            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            self._f.close()

    def write(self, data: str) -> None:
        if self._f is not None:
            self._f.write(data)
            self._f.flush()
