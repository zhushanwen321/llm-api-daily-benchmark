from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from benchmark.models.schemas import ApiCallMetrics, EvalResult
from benchmark.repository.handlers.metadata_handler import FileAppend


class AnswerHandler:
    """管理 {run_id}/{task_id}/answer.jsonl 的读写。

    每行一条 JSON 记录，包含两个顶层对象：
      - answer: 评测答案数据
      - api_metrics: API 调用指标
    """

    FILENAME = "answer.jsonl"

    def __init__(self, data_root: Path) -> None:
        self._root = data_root

    def _path(self, run_id: str, task_id: str) -> Path:
        return self._root / run_id / task_id / self.FILENAME

    @staticmethod
    def _serialize_model(obj: Any) -> dict[str, Any]:
        return json.loads(obj.model_dump_json())

    def save_answer(
        self,
        result: EvalResult,
        metrics: Optional[ApiCallMetrics] = None,
    ) -> str:
        record: dict[str, Any] = {
            "answer": self._serialize_model(result),
            "api_metrics": self._serialize_model(metrics) if metrics else None,
        }

        path = self._path(result.run_id, result.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with FileAppend(str(path)) as f:
            f.write(line)

        return result.result_id

    def get_answer(self, run_id: str, task_id: str) -> dict[str, Any]:
        path = self._path(run_id, task_id)
        if not path.exists():
            raise FileNotFoundError(
                f"answer.jsonl not found for run_id={run_id}, task_id={task_id}"
            )
        records = self._read_all_lines(path)
        if not records:
            raise FileNotFoundError(
                f"answer.jsonl is empty for run_id={run_id}, task_id={task_id}"
            )
        return records[-1]

    def get_answers_by_run(self, run_id: str) -> list[dict[str, Any]]:
        run_dir = self._root / run_id
        if not run_dir.exists():
            return []

        answers: list[dict[str, Any]] = []
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            answer_path = task_dir / self.FILENAME
            if not answer_path.exists():
                continue
            records = self._read_all_lines(answer_path)
            if records:
                answers.append(records[-1])
        return answers

    @staticmethod
    def _read_all_lines(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
