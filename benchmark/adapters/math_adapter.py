"""MATH 数据集适配器。加载 Level 3-5 数学题，覆盖多学科."""

from __future__ import annotations

import os
import random
from typing import List

from benchmark.adapters.hf_loader import load_hf_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class MATHAdapter(DatasetAdapter):
    """MATH 适配器，选择 Level 3-5 的题目，覆盖多个学科，共 15 题."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        cache_dir = path or os.path.join("benchmark", "datasets", "math")
        dataset = load_hf_dataset(
            "nlile/hendrycks-MATH-benchmark",
            split="test",
            cache_dir=cache_dir,
        )

        # 筛选 Level 3-5 的题目
        eligible = [
            item for item in dataset
            if item["level"] >= 3
        ]

        # 按学科分组，确保覆盖度
        rng = random.Random(42)
        by_subject: dict[str, list] = {}
        for item in eligible:
            subj = item["subject"]
            by_subject.setdefault(subj, []).append(item)

        selected = []
        subjects = list(by_subject.keys())
        rng.shuffle(subjects)

        # 每个学科至少选 1 题
        for subj in subjects:
            pool = by_subject[subj]
            rng.shuffle(pool)
            selected.append(pool[0])

        # 补足到 15 题
        remaining = []
        for subj in subjects:
            remaining.extend(by_subject[subj][1:])
        rng.shuffle(remaining)

        for item in remaining:
            if len(selected) >= 15:
                break
            selected.append(item)

        tasks = []
        for idx, item in enumerate(selected[:15]):
            task = TaskDefinition(
                task_id=f"math_{item['unique_id']}",
                dimension="reasoning",
                dataset="math",
                prompt=build_structured_prompt(item["problem"], "reasoning", dataset="math"),
                expected_output=item["answer"],
                metadata={
                    "level": item["level"],
                    "subject": item["subject"],
                    "source": "nlile/hendrycks-MATH-benchmark",
                },
            )
            tasks.append(task)

        return tasks

    def validate(self, task: TaskDefinition) -> bool:
        return bool(
            task.task_id
            and task.prompt
            and task.expected_output
            and task.dimension == "reasoning"
        )

    def get_dimension(self) -> str:
        return "reasoning"
