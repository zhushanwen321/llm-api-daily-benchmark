"""前端代码生成探针."""

from __future__ import annotations

import logging
import os
from typing import Any
from datetime import datetime

import yaml

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.probes import BaseProbe

logger = logging.getLogger(__name__)


class FrontendProbe(BaseProbe):
    """前端代码生成探针 - 评估模型生成前端代码的能力."""

    @property
    def frequency(self) -> str:
        return "slow"

    def load_probes(self, path: str = "") -> list[TaskDefinition]:
        """加载前端探针."""
        if not path:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            probe_file = os.path.join(module_dir, "frontend_probes.yaml")
        else:
            probe_file = path

        try:
            with open(probe_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Frontend probes file not found: {probe_file}")
            return []
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format in {probe_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load frontend probes: {e}")
            return []

        if not data or "probes" not in data:
            logger.warning(f"No probes found in {probe_file}")
            return []

        tasks: list[TaskDefinition] = []
        for probe in data.get("probes", []):
            try:
                task = TaskDefinition(
                    task_id=probe["id"],
                    dimension="frontend",
                    dataset=probe["category"],
                    prompt=probe["prompt"],
                    expected_output=probe.get("expected_behavior", ""),
                    metadata={
                        "type": probe.get("type", "code_generation"),
                        "category": probe["category"],
                        "difficulty": probe.get("difficulty", "medium"),
                        "framework": probe.get("framework", "React"),
                        "validation_criteria": probe.get("validation_criteria", []),
                    },
                )
                tasks.append(task)
            except (KeyError, TypeError) as e:
                logger.warning(f"Invalid probe definition: {probe}, error: {e}")
                continue

        return tasks

    async def execute_probe(
        self,
        probe: TaskDefinition,
        model: str,
        adapter: LLMEvalAdapter,
    ) -> EvalResult:
        """执行前端代码生成探针."""
        response = await adapter.agenerate(
            prompt=probe.prompt,
            model=model,
            temperature=0.0,
            max_tokens=2000,
        )

        score = self._evaluate_code(response.content, probe.metadata)

        return EvalResult(
            result_id=f"{model}_{probe.task_id}_{datetime.now().timestamp()}",
            run_id="",
            task_id=probe.task_id,
            task_content=probe.prompt,
            model_output=response.content,
            functional_score=score,
            final_score=score,
            passed=score >= 70.0,
            execution_time=response.duration,
            created_at=datetime.now(),
            details={
                "category": probe.metadata.get("category", "unknown"),
                "difficulty": probe.metadata.get("difficulty", "medium"),
                "framework": probe.metadata.get("framework", "React"),
                "type": probe.metadata.get("type", "code_generation"),
            },
        )

    def _evaluate_code(self, response: str, metadata: dict) -> float:
        """评估生成的代码质量."""
        validation_criteria = metadata.get("validation_criteria", [])

        response_lower = response.lower()
        passed_count = 0

        # 检查关键代码元素 - 验证完整短语或主要关键词
        for criterion in validation_criteria:
            criterion_lower = criterion.lower()
            # 检查完整短语是否在响应中
            if criterion_lower in response_lower:
                passed_count += 1
            else:
                # 检查主要关键词（至少50%匹配）
                criterion_words = criterion_lower.split()
                if criterion_words:
                    matched_words = sum(
                        1 for word in criterion_words if word in response_lower
                    )
                    if matched_words / len(criterion_words) >= 0.5:
                        passed_count += 0.5

        score = (
            (passed_count / len(validation_criteria) * 100)
            if validation_criteria
            else 0
        )
        return min(score, 100.0)

    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """提取前端探针特征."""
        return {
            "frontend_score": result.functional_score,
            "passed": result.passed,
            "category": result.details.get("category", "unknown"),
            "difficulty": result.details.get("difficulty", "medium"),
            "framework": result.details.get("framework", "React"),
            "type": result.details.get("type", "code_generation"),
        }
