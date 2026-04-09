"""MMLU 数据集适配器。加载 computer_science 和 abstract_algebra 学科.

.. deprecated::
    system-architecture 维度已被移除，此适配器不再用于主动评测。
    保留此文件用于历史数据兼容。
"""

from __future__ import annotations

import os
from typing import List

from benchmark.adapters.hf_loader import load_hf_dataset

from benchmark.adapters.base import DatasetAdapter
from benchmark.core.prompt_builder import build_structured_prompt
from benchmark.models.schemas import TaskDefinition


class MMLUAdapter(DatasetAdapter):
    """MMLU 适配器，选择指定学科的最难题目."""

    def __init__(self, subjects: List[str] | None = None):
        """初始化适配器.

        Args:
            subjects: 学科列表. 默认为 ["computer_science", "abstract_algebra"].
        """
        self.subjects = subjects or ["college_computer_science", "abstract_algebra"]

    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载 MMLU 题目.

        从每个学科选择一定数量的题目，总共返回 5 题.
        computer_science 选择 3 题，abstract_algebra 选择 2 题.

        Args:
            path: 数据集缓存路径.

        Returns:
            TaskDefinition 列表.
        """
        cache_dir = path or os.path.join("benchmark", "datasets", "mmlu")
        all_tasks = []

        # 计算每个学科的题目数量
        num_subjects = len(self.subjects)
        if num_subjects == 2:
            counts = {self.subjects[0]: 3, self.subjects[1]: 2}
        else:
            # 平均分配，多余给第一个学科
            base_count = 5 // num_subjects
            remainder = 5 % num_subjects
            counts = {
                s: base_count + (1 if i == 0 else 0)
                for i, s in enumerate(self.subjects)
            }

        for subject in self.subjects:
            count = counts.get(subject, 1)
            try:
                dataset = load_hf_dataset(
                    "cais/mmlu",
                    split="test",
                    cache_dir=cache_dir,
                    config_name=subject,
                )
            except Exception as e:
                # 如果学科不存在，尝试添加前缀
                try:
                    dataset = load_hf_dataset(
                        "cais/mmlu",
                        split="test",
                        cache_dir=cache_dir,
                        config_name=f"mmlu_{subject}",
                    )
                except Exception:
                    raise ValueError(
                        f"Failed to load MMLU subject '{subject}'. "
                        f"Available subjects can be found at https://huggingface.co/datasets/cais/mmlu"
                    ) from e

            # 简单策略：取前 count 个题目
            # MMLU 题目格式：{"question": ..., "choices": [...], "answer": 0}（answer是索引）
            for idx in range(min(count, len(dataset))):
                item = dataset[idx]
                question = item["question"]
                choices = item.get("choices", [])
                answer_idx = item["answer"]  # 正确答案的索引

                # 构造题目文本
                if choices:
                    choices_text = "\n".join(
                        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
                    )
                    prompt_text = f"{question}\n\n{choices_text}\n\nAnswer with the letter (A, B, C, D)."
                    expected_answer = chr(
                        65 + answer_idx
                    )  # 索引转字母 (0->A, 1->B, etc.)
                else:
                    prompt_text = question
                    expected_answer = str(answer_idx)

                task = TaskDefinition(
                    task_id=f"mmlu_{subject}_{idx + 1}",
                    dimension="system-architecture",
                    dataset="mmlu",
                    prompt=build_structured_prompt(prompt_text, "system-architecture"),
                    expected_output=expected_answer,
                    metadata={
                        "subject": subject,
                        "source": "cais/mmlu",
                        "choices": choices,
                        "answer_idx": answer_idx,
                    },
                )
                all_tasks.append(task)

        return all_tasks[:5]  # 确保最多返回5题

    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式."""
        return bool(
            task.task_id
            and task.prompt
            and task.expected_output
            and task.dimension == "system-architecture"
        )

    def get_dimension(self) -> str:
        return "system-architecture"
