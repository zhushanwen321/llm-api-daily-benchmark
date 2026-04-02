"""FrontCode 数据集适配器。自建前端评测题目."""

from __future__ import annotations

import json
import os
from typing import List

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class FrontCodeAdapter(DatasetAdapter):
    """FrontCode 适配器，加载自建前端评测题目."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 FrontCode 题目.

        Args:
            path: 数据集路径.

        Returns:
            TaskDefinition 列表.
        """
        data_dir = path or os.path.join("benchmark", "datasets", "frontcode")
        tasks_file = os.path.join(data_dir, "tasks.json")

        if not os.path.exists(tasks_file):
            raise FileNotFoundError(
                f"FrontCode tasks file not found: {tasks_file}. "
                f"Please create tasks.json in {data_dir}"
            )

        with open(tasks_file) as f:
            data = json.load(f)

        tasks = []
        for item in data.get("tasks", []):
            task = TaskDefinition(
                task_id=item["id"],
                dimension="frontend-dev",
                dataset="frontcode",
                prompt=build_structured_prompt(item["prompt"], "frontend-dev"),
                expected_output="",  # FrontCode 使用关键词匹配，不需要 expected_output
                metadata={
                    "type": item["type"],
                    "keywords": item["keywords"],
                    "source": "frontcode",
                },
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式."""
        return bool(
            task.task_id
            and task.prompt
            and task.dimension == "frontend-dev"
            and "keywords" in task.metadata
        )

    def get_dimension(self) -> str:
        return "frontend-dev"
