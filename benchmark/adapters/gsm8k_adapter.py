"""GSM8K 数据集适配器。加载最难的5道数学推理题。步骤数最多的题目被认为最难."""

from __future__ import annotations
import os
import re
from typing import List
from datasets import load_dataset
from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class GSM8KAdapter(DatasetAdapter):
    """GSM8K 适配器，选择解答步骤最多的 5 道题."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 GSM8K 最难的5题（按解答步骤数排序）."""
        cache_dir = path or os.path.join("benchmark", "datasets", "gsm8k")
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )
        items_with_steps = []
        for item in dataset:
            steps = len(item["answer"].split("\n"))
            items_with_steps.append((item, steps))
        items_with_steps.sort(key=lambda x: x[1], reverse=True)
        hardest_5 = [item for item, _ in items_with_steps[:5]]

        tasks = []
        for idx, item in enumerate(hardest_5):
            answer_text = item["answer"]
            match = re.search(r"####\s*(-?\d+\.?\d*)", answer_text)
            expected = match.group(1).strip() if match else ""

            task = TaskDefinition(
                task_id=f"gsm8k_hardest_{idx + 1}",
                dimension="reasoning",
                dataset="gsm8k",
                prompt=build_structured_prompt(item["question"], "reasoning"),
                expected_output=expected,
                metadata={"difficulty": "hard", "source": "openai/gsm8k"},
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        return bool(task.task_id and task.prompt and task.expected_output)

    def get_dimension(self) -> str:
        return "reasoning"
