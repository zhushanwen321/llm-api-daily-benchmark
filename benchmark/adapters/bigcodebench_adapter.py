"""BigCodeBench 数据集适配器.加载官方 Hard 子集，过滤重型依赖后随机选 15 题."""

from __future__ import annotations

import ast
import logging
import os
import random
from typing import List

from benchmark.adapters.hf_loader import load_hf_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition

logger = logging.getLogger(__name__)

# 安装慢或有系统依赖的重型库，含这些库的题目会被排除
_HEAVY_LIBS = frozenset({
    "tensorflow", "keras", "geopandas", "librosa", "soundfile",
    "gensim", "pytesseract", "cv2", "nltk", "wordcloud",
    "torch", "transformers", "tiktoken", "Levenshtein",
    "Crypto", "cryptography", "shapely", "sklearn",
})


def _is_heavy_task(item: dict) -> bool:
    """检查题目是否依赖重型库（基于 libs 字段）."""
    libs = item.get("libs")
    if not libs:
        return False
    # libs 在 HuggingFace 数据集中是字符串形式的列表，需要解析
    if isinstance(libs, str):
        try:
            libs = ast.literal_eval(libs)
        except (ValueError, SyntaxError):
            return False
    return bool(set(libs) & _HEAVY_LIBS)


class BigCodeBenchAdapter(DatasetAdapter):
    """BigCodeBench 适配器，从 Hard 子集过滤重型依赖后随机选 15 题."""

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 BigCodeBench-Hard 子集,过滤重型库后随机选 15 题."""
        cache_dir = path or os.path.join("benchmark", "datasets", "bigcodebench")
        dataset = load_hf_dataset(
            "bigcode/bigcodebench-hard",
            split="v0.1.0_hf",
            cache_dir=cache_dir,
        )

        # 过滤掉依赖重型库的题目
        lightweight = [item for item in dataset if not _is_heavy_task(item)]

        # 随机选 15 题（固定随机种子保证可复现）
        rng = random.Random(42)
        n = min(15, len(lightweight))
        if n < 15:
            logger.warning(f"BigCodeBench: only {n}/{len(dataset)} lightweight tasks available (wanted 15)")
        selected = rng.sample(lightweight, n)

        tasks = []
        for idx, item in enumerate(selected):
            task_id = f"bigcodebench_hard_{idx + 1}"
            # BigCodeBench 字段: task_id, instruct, test, entry_point
            task = TaskDefinition(
                task_id=task_id,
                dimension="backend-dev",
                dataset="bigcodebench",
                prompt=build_structured_prompt(
                    item.get("instruct_prompt", item.get("complete_prompt", "")),
                    "backend-dev",
                ),
                expected_output="",
                metadata={
                    "difficulty": "hard",
                    "source": "bigcode/bigcodebench-hard",
                    "test": item.get("test", ""),
                    "entry_point": item.get("entry_point", ""),
                    "canonical_solution": item.get("canonical_solution", ""),
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
