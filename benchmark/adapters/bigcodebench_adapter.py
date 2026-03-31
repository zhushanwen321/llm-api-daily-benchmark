"""BigCodeBench 数据集适配器.加载官方 Hard 子集，随机选 5 题."""

from __future__ import annotations

import os
import random
from typing import List

from datasets import load_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.models.schemas import TaskDefinition


class BigCodeBenchAdapter(DatasetAdapter):
    """BigCodeBench 适配器，从 Hard 子集随机选 5 题."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 BigCodeBench-Hard 子集,随机选 5 题."""
        cache_dir = path or os.path.join("benchmark", "datasets", "bigcodebench")
        dataset = load_dataset(
            "bigcode/bigcodebench-hard",
            split="v0.1.0_hf",
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

        # 随机选 5 题（固定随机种子保证可复现）
        rng = random.Random(42)
        indices = rng.sample(range(len(dataset)), min(5, len(dataset)))
        selected = [dataset[i] for i in indices]

        tasks = []
        for idx, item in enumerate(selected):
            task_id = f"bigcodebench_hard_{idx + 1}"
            # BigCodeBench 字段: task_id, instruct, test, entry_point
            task = TaskDefinition(
                task_id=task_id,
                dimension="backend-dev",
                dataset="bigcodebench",
                prompt=item.get("instruct_prompt", item.get("complete_prompt", "")),
                expected_output="",
                metadata={
                    "difficulty": "hard",
                    "source": "bigcode/bigcodebench-hard",
                    "test": item.get("test", ""),
                    "entry_point": item.get("entry_point", ""),
                },
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式.必须有 task_id, prompt, test 元数据."""
        has_test = "test" in task.metadata and task.metadata["test"]
        return bool(task.task_id and task.prompt and has_test)

    def get_dimension(self) -> str:
        """返回评测维度."""
        return "backend-dev"
