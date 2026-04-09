"""安全探针适配器."""

from __future__ import annotations

import yaml
from typing import Any
from datetime import datetime

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.probes import BaseProbe


class SafetyProbe(BaseProbe):
    """安全探针 - 检测模型对敏感内容的处理能力."""

    @property
    def frequency(self) -> str:
        return "medium"

    def load_probes(self, path: str = "") -> list[TaskDefinition]:
        """加载安全探针."""
        probe_file = path or "benchmark/probes/safety/safety_probes.yaml"

        with open(probe_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tasks: list[TaskDefinition] = []
        for probe in data.get("probes", []):
            task = TaskDefinition(
                task_id=probe["id"],
                dimension="safety",
                dataset=probe["category"],
                prompt=probe["prompt"],
                expected_output=probe.get("expected_behavior", ""),
                metadata={
                    "type": probe.get("type", "unknown"),
                    "category": probe["category"],
                    "severity": probe.get("severity", "medium"),
                    "assertions": probe.get("assertions", {}),
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
        """执行安全探针."""
        response = await adapter.agenerate(
            prompt=probe.prompt,
            model=model,
            temperature=0.0,
            max_tokens=200,
        )

        assertions = probe.metadata.get("assertions", {})
        score, passed = self._evaluate_response(response.content, assertions)

        return EvalResult(
            result_id=f"{model}_{probe.task_id}_{datetime.now().timestamp()}",
            run_id="",
            task_id=probe.task_id,
            task_content=probe.prompt,
            model_output=response.content,
            functional_score=score,
            final_score=score,
            passed=passed,
            execution_time=response.duration,
            created_at=datetime.now(),
            details={
                "category": probe.metadata.get("category", "unknown"),
                "severity": probe.metadata.get("severity", "medium"),
            },
        )

    def _evaluate_response(
        self,
        response: str,
        assertions: dict,
    ) -> tuple[float, bool]:
        """评估响应是否符合安全要求."""
        response_lower = response.lower()

        must_contain = assertions.get("must_contain", [])
        must_not_contain = assertions.get("must_not_contain", [])
        must_contain_one_of = assertions.get("must_contain_one_of", [])

        passed_count = 0
        total_checks = 0

        for keyword in must_contain:
            total_checks += 1
            if keyword.lower() in response_lower:
                passed_count += 1

        for keyword in must_not_contain:
            total_checks += 1
            if keyword.lower() not in response_lower:
                passed_count += 1

        if must_contain_one_of:
            total_checks += 1
            if any(kw.lower() in response_lower for kw in must_contain_one_of):
                passed_count += 1

        score = (passed_count / total_checks * 100) if total_checks > 0 else 0
        passed = passed_count == total_checks

        return score, passed

    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """提取安全探针特征."""
        return {
            "safety_score": result.functional_score,
            "passed": result.passed,
            "category": result.details.get("category", "unknown"),
            "severity": result.details.get("severity", "medium"),
        }
