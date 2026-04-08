"""Probe 监控适配器。加载高频监控题目."""

from __future__ import annotations

import json
import os
from typing import List

from benchmark.adapters.base import DatasetAdapter
from benchmark.models.schemas import TaskDefinition


class ProbeAdapter(DatasetAdapter):
    """Probe 监控适配器。加载高频监控题目."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 Probe 题目.

        Args:
            path: 数据集路径，默认 benchmark/datasets/probe。

        Returns:
            TaskDefinition 列表。
        """
        data_dir = path or os.path.join("benchmark", "datasets", "probe")
        tasks_file = os.path.join(data_dir, "tasks.json")

        if not os.path.exists(tasks_file):
            raise FileNotFoundError(
                f"Probe tasks file not found: {tasks_file}. "
                f"Please create tasks.json in {data_dir}"
            )

        try:
            with open(tasks_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in Probe tasks file: {tasks_file}. "
                f"Parse error: {e}"
            ) from e

        tasks: List[TaskDefinition] = []
        for idx, item in enumerate(data.get("tasks", [])):
            required_fields = ["id", "prompt", "expected_answer"]
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                raise ValueError(
                    f"Task at index {idx} missing required fields: {missing_fields}"
                )

            # 透传可选字段到 metadata
            metadata: dict = {
                "type": item.get("type", "unknown"),
                "expected_format": item.get("expected_format", "text"),
                "instruction_constraints": item.get("instruction_constraints", []),
                "difficulty": item.get("difficulty", "medium"),
                "source": item.get("source", "probe"),
            }

            task = TaskDefinition(
                task_id=item["id"],
                dimension="probe",
                dataset="probe",
                prompt=item["prompt"],
                expected_output=item["expected_answer"],
                metadata=metadata,
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式.

        检查 task_id、prompt、expected_output 存在。
        """
        return bool(task.task_id and task.prompt and task.expected_output is not None)

    def get_dimension(self) -> str:
        return "probe"
