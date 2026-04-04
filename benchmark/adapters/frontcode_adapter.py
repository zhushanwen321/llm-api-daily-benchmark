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

        try:
            with open(tasks_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in FrontCode tasks file: {tasks_file}. "
                f"Parse error: {e}"
            ) from e

        tasks = []
        for idx, item in enumerate(data.get("tasks", [])):
            # 验证必需字段存在
            required_fields = ["id", "type", "prompt", "keywords"]
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                raise ValueError(
                    f"Task at index {idx} missing required fields: {missing_fields}"
                )

            # 验证 keywords 是非空列表
            if not isinstance(item["keywords"], list):
                raise ValueError(
                    f"Task {item['id']}: 'keywords' must be a list, got {type(item['keywords']).__name__}"
                )
            if not item["keywords"]:
                raise ValueError(
                    f"Task {item['id']}: 'keywords' cannot be empty"
                )

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
        """验证任务格式.

        验证要求:
        - task_id 非空
        - prompt 非空
        - dimension 为 frontend-dev
        - metadata 中存在 keywords 且为非空列表
        """
        keywords = task.metadata.get("keywords")
        return bool(
            task.task_id
            and task.prompt
            and task.dimension == "frontend-dev"
            and isinstance(keywords, list)
            and keywords  # 非空列表
        )

    def get_dimension(self) -> str:
        return "frontend-dev"
