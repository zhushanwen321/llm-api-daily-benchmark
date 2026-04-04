"""MMLU-Pro 数据集适配器。加载技术相关学科（CS、数学、物理），共 15 题."""

from __future__ import annotations

import os
import random
from typing import List

from datasets import load_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.models.schemas import TaskDefinition


class MMLUProAdapter(DatasetAdapter):
    """MMLU-Pro 适配器，选择技术学科各 5 题，共 15 题."""

    CATEGORIES = ["computer science", "math", "physics"]
    PER_CATEGORY = 5

    def load(self, path: str = "") -> List[TaskDefinition]:
        cache_dir = path or os.path.join("benchmark", "datasets", "mmlu_pro")
        dataset = load_dataset(
            "TIGER-Lab/MMLU-Pro",
            split="test",
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

        rng = random.Random(42)
        tasks = []

        for category in self.CATEGORIES:
            pool = [item for item in dataset if item["category"] == category]
            if not pool:
                continue
            selected = rng.sample(pool, min(self.PER_CATEGORY, len(pool)))

            for item in selected:
                options = item["options"]
                num_options = len(options)
                last_letter = chr(64 + num_options)  # 10 options -> 'J'

                # 选择题格式 prompt，不使用 build_structured_prompt
                options_text = "\n".join(
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
                )
                prompt_text = (
                    f"{item['question']}\n\n{options_text}\n\n"
                    f"Answer with the letter (A-{last_letter}). "
                    f"Think step by step, then on the LAST LINE write ONLY the single letter."
                )

                task = TaskDefinition(
                    task_id=f"mmlu_pro_{item['question_id']}",
                    dimension="system-architecture",
                    dataset="mmlu-pro",
                    prompt=prompt_text,
                    expected_output=item["answer"],
                    metadata={
                        "category": category,
                        "source": "TIGER-Lab/MMLU-Pro",
                        "answer_index": item["answer_index"],
                        "num_options": num_options,
                    },
                )
                tasks.append(task)

        return tasks[:15]

    def validate(self, task: TaskDefinition) -> bool:
        return bool(
            task.task_id
            and task.prompt
            and task.expected_output
            and task.dimension == "system-architecture"
        )

    def get_dimension(self) -> str:
        return "system-architecture"
