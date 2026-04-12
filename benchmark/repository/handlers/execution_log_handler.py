from __future__ import annotations

from pathlib import Path

from benchmark.core.tz import now
from benchmark.repository.handlers.metadata_handler import FileAppend


class ExecutionLogHandler:
    """管理 {benchmark_id}/{question_id}/execution.log 的追加写入与读取。"""

    FILENAME = "execution.log"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, benchmark_id: str, question_id: str) -> Path:
        return self._root / benchmark_id / question_id / self.FILENAME

    @staticmethod
    def _now() -> str:
        return now().strftime("%Y-%m-%d %H:%M:%S")

    def append_log(
        self,
        benchmark_id: str,
        question_id: str,
        message: str,
        level: str = "INFO",
    ) -> None:
        path = self._path(benchmark_id, question_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = f"{self._now()} - {level} - {message}\n"
        with FileAppend(str(path)) as f:
            f.write(line)

    def read_log(self, benchmark_id: str, question_id: str) -> str:
        path = self._path(benchmark_id, question_id)
        if not path.exists():
            raise FileNotFoundError(
                f"execution.log not found for "
                f"benchmark_id={benchmark_id}, question_id={question_id}"
            )
        return path.read_text(encoding="utf-8")
