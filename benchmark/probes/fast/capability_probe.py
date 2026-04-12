"""高频能力探针."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter


class CapabilityProbe:
    """高频能力探针."""

    @property
    def frequency(self) -> str:
        return "fast"

    def load_probes(self, path: str = "") -> list[TaskDefinition]:
        """加载探针题目."""
        data_dir = path or os.path.join("benchmark", "datasets", "probe")
        tasks_file = os.path.join(data_dir, "tasks.json")

        if not os.path.exists(tasks_file):
            raise FileNotFoundError(f"Probe tasks file not found: {tasks_file}")

        with open(tasks_file, encoding="utf-8") as f:
            data = json.load(f)

        tasks: list[TaskDefinition] = []
        for item in data.get("tasks", []):
            task = TaskDefinition(
                task_id=item["id"],
                dimension="probe",
                dataset="fast_capability",
                prompt=item["prompt"],
                expected_output=item["expected_answer"],
                metadata={
                    "type": item.get("type", "unknown"),
                    "difficulty": item.get("difficulty", "medium"),
                },
            )
            tasks.append(task)

        return tasks

    async def execute_probe(
        self,
        probe: TaskDefinition,
        model: str,
        adapter: LLMEvalAdapter,
    ) -> EvalResult:
        """执行探针."""
        response = await adapter.agenerate(
            prompt=probe.prompt,
            model=model,
            temperature=0.0,
            max_tokens=100,
        )

        # 简单评分
        answer = response.content.strip()
        expected = (probe.expected_output or "").strip()
        passed = expected.lower() in answer.lower()
        score = 100.0 if passed else 0.0

        from benchmark.core.tz import now

        return EvalResult(
            result_id=f"{model}_{probe.task_id}_{time.time()}",
            run_id="",
            task_id=probe.task_id,
            task_content=probe.prompt,
            model_output=response.content,
            functional_score=score,
            final_score=score,
            passed=passed,
            execution_time=response.duration,
            created_at=now(),
        )

    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """提取特征."""
        return {
            "passed": result.passed,
            "score": result.functional_score,
            "execution_time": result.execution_time,
            "output_length": len(result.model_output),
        }
