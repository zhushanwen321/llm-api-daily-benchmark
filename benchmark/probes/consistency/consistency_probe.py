"""语义一致性探针 - 检测模型回答的自洽性."""

from __future__ import annotations

from typing import Any
from datetime import datetime

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.probes import BaseProbe


class ConsistencyProbe(BaseProbe):
    """一致性探针 - 通过多种方式提问检测模型回答稳定性."""

    @property
    def frequency(self) -> str:
        return "slow"

    def load_probes(self) -> list[TaskDefinition]:
        """加载一致性探针定义."""
        probe_groups = [
            {
                "group_id": "math_basic",
                "category": "paraphrase_consistency",
                "probes": [
                    {
                        "id": "cons_math_1a",
                        "prompt": "15加27等于多少？",
                        "expected": "42",
                    },
                    {
                        "id": "cons_math_1b",
                        "prompt": "计算15和27的和。",
                        "expected": "42",
                    },
                    {
                        "id": "cons_math_1c",
                        "prompt": "15 + 27 = ?",
                        "expected": "42",
                    },
                ],
            },
            {
                "group_id": "factual_knowledge",
                "category": "knowledge_consistency",
                "probes": [
                    {
                        "id": "cons_fact_1a",
                        "prompt": "中国的首都是哪里？",
                        "expected": "北京",
                    },
                    {
                        "id": "cons_fact_1b",
                        "prompt": "北京是哪个国家的首都？",
                        "expected": "中国",
                    },
                ],
            },
            {
                "group_id": "logical_reasoning",
                "category": "logical_consistency",
                "probes": [
                    {
                        "id": "cons_logic_1a",
                        "prompt": "如果A大于B，B大于C，那么A和C哪个更大？",
                        "expected": "A",
                    },
                    {
                        "id": "cons_logic_1b",
                        "prompt": "已知A>B，B>C，比较A和C的大小关系。",
                        "expected": "A>C",
                    },
                ],
            },
            {
                "group_id": "definition",
                "category": "definition_consistency",
                "probes": [
                    {
                        "id": "cons_def_1a",
                        "prompt": "什么是光合作用？",
                        "expected": "植物",
                    },
                    {
                        "id": "cons_def_1b",
                        "prompt": "请解释光合作用的定义。",
                        "expected": "植物",
                    },
                ],
            },
        ]

        tasks: list[TaskDefinition] = []
        for group in probe_groups:
            for probe in group["probes"]:
                tasks.append(
                    TaskDefinition(
                        task_id=probe["id"],
                        dimension="consistency",
                        dataset=group["category"],
                        prompt=probe["prompt"],
                        expected_output=probe["expected"],
                        metadata={
                            "group_id": group["group_id"],
                            "category": group["category"],
                        },
                    )
                )

        return tasks

    async def execute_probe(
        self,
        probe: TaskDefinition,
        model: str,
        adapter: LLMEvalAdapter,
    ) -> EvalResult:
        """执行一致性探针."""
        response = await adapter.agenerate(
            prompt=probe.prompt,
            model=model,
            temperature=0.0,
            max_tokens=200,
        )

        expected = probe.expected_output
        similarity = self._calculate_similarity(response.content, expected)

        return EvalResult(
            result_id=f"{model}_{probe.task_id}_{datetime.now().timestamp()}",
            run_id="",
            task_id=probe.task_id,
            task_content=probe.prompt,
            model_output=response.content,
            functional_score=similarity,
            final_score=similarity,
            passed=similarity >= 70.0,
            execution_time=response.duration,
            created_at=datetime.now(),
            details={
                "group_id": probe.metadata.get("group_id", "unknown"),
                "category": probe.metadata.get("category", "unknown"),
                "expected": expected,
            },
        )

    def _calculate_similarity(self, response: str, expected: str) -> float:
        """计算响应与期望答案的相似度."""
        response_lower = response.lower()
        expected_lower = expected.lower()

        if not expected_lower:
            return 0.0

        if expected_lower in response_lower:
            return 100.0

        response_words = set(response_lower.split())
        expected_words = set(expected_lower.split())

        if not expected_words:
            return 0.0

        intersection = response_words & expected_words
        similarity = len(intersection) / len(expected_words) * 100

        return min(similarity, 100.0)

    def calculate_group_consistency(
        self,
        results: list[EvalResult],
    ) -> dict[str, Any]:
        """计算同一组探针的一致性分数."""
        if not results:
            return {"consistency_score": 0.0, "variance": 0.0}

        scores = [r.functional_score for r in results]
        avg_score = sum(scores) / len(scores)

        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        else:
            variance = 0.0

        all_passed = all(r.passed for r in results)

        return {
            "consistency_score": avg_score,
            "variance": variance,
            "all_passed": all_passed,
            "sample_count": len(results),
        }

    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """提取一致性特征."""
        return {
            "consistency_score": result.functional_score,
            "group_id": result.details.get("group_id", "unknown"),
            "category": result.details.get("category", "unknown"),
            "passed": result.passed,
        }
