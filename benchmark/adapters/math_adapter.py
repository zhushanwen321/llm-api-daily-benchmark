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

    _CONFIGS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "precalculus",
    ]

    def load(self, path: str = "") -> List[TaskDefinition]:
        cache_dir = path or os.path.join("benchmark", "datasets", "math")

        all_items: list[dict] = []
        for config in self._CONFIGS:
            rows = load_hf_dataset(
                "EleutherAI/hendrycks_math",
                split="test",
                cache_dir=cache_dir,
                config_name=config,
            )
            for row in rows:
                row["_config"] = config
            all_items.extend(rows)

        def _parse_level(item: dict) -> int:
            raw = item.get("level", "")
            if isinstance(raw, int):
                return raw
            return int(raw.replace("Level ", ""))

        eligible = [item for item in all_items if _parse_level(item) >= 3]

        rng = random.Random(42)
        by_subject: dict[str, list] = {}
        for item in eligible:
            subj = item.get("type", "unknown")
            by_subject.setdefault(subj, []).append(item)

        selected = []
        subjects = list(by_subject.keys())
        rng.shuffle(subjects)

        for subj in subjects:
            pool = by_subject[subj]
            rng.shuffle(pool)
            selected.append(pool[0])

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
            level_int = _parse_level(item)
            task = TaskDefinition(
                task_id=f"math_{idx}",
                dimension="reasoning",
                dataset="math",
                prompt=build_structured_prompt(
                    item["problem"], "reasoning", dataset="math"
                ),
                expected_output=item["solution"],
                metadata={
                    "level": level_int,
                    "subject": item.get("type", "unknown"),
                    "source": "EleutherAI/hendrycks_math",
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
