"""推理探针 - 检测模型的逻辑推理、数学推理和常识推理能力."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import yaml

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.probes import BaseProbe

logger = logging.getLogger(__name__)


class ReasoningProbe(BaseProbe):
    """推理探针 - 评估模型的逻辑推理、数学推理和常识推理能力."""

    @property
    def frequency(self) -> str:
        return "medium"

    def load_probes(self, path: str = "") -> list[TaskDefinition]:
        """加载推理探针题目."""
        if not path:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            probe_file = os.path.join(module_dir, "reasoning_probes.yaml")
        else:
            probe_file = path

        try:
            with open(probe_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Reasoning probes file not found: {probe_file}")
            return []
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format in {probe_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load reasoning probes: {e}")
            return []

        if not data or "probes" not in data:
            logger.warning(f"No probes found in {probe_file}")
            return []

        tasks: list[TaskDefinition] = []
        for probe in data.get("probes", []):
            try:
                task = TaskDefinition(
                    task_id=probe["id"],
                    dimension="reasoning",
                    dataset=probe["category"],
                    prompt=probe["prompt"],
                    expected_output=probe.get("expected_output", ""),
                    metadata={
                        "type": probe.get("type", "unknown"),
                        "category": probe["category"],
                        "difficulty": probe.get("difficulty", "medium"),
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
        """执行推理探针."""
        response = await adapter.agenerate(
            prompt=probe.prompt,
            model=model,
            temperature=0.0,
            max_tokens=150,
        )

        score, passed = self._evaluate_reasoning_response(
            response.content,
            probe.expected_output or "",
            probe.metadata.get("type", "unknown"),
        )

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
                "difficulty": probe.metadata.get("difficulty", "medium"),
                "reasoning_type": probe.metadata.get("type", "unknown"),
            },
        )

    def _evaluate_reasoning_response(
        self,
        response: str,
        expected: str,
        reasoning_type: str,
    ) -> tuple[float, bool]:
        """评估推理响应的正确性."""
        response_clean = response.strip()
        expected_clean = expected.strip()

        # 直接匹配（对于单选题等）
        if expected_clean in response_clean:
            return 100.0, True

        # 数值匹配（提取数字进行比较）
        response_numbers = self._extract_numbers(response_clean)
        expected_numbers = self._extract_numbers(expected_clean)

        if response_numbers and expected_numbers:
            # 对于多答案情况，检查是否包含任一答案
            # 支持多种分隔符：或、或者、/、,
            for separator in ["或", "或者", "/", ",", "，"]:
                if separator in expected_clean:
                    expected_values = [
                        v.strip() for v in expected_clean.split(separator)
                    ]
                    for val in expected_values:
                        if val in response_clean:
                            return 100.0, True
                    break
            else:
                # 没有分隔符，检查完整匹配
                if expected_clean in response_clean:
                    return 100.0, True

            # 数值答案精确匹配（第一个数字）
            if response_numbers[0] == expected_numbers[0]:
                return 100.0, True

            # 80% 部分正确（接近答案，20%误差范围内）
            if expected_numbers[0] != 0:
                relative_error = abs(response_numbers[0] - expected_numbers[0]) / abs(
                    expected_numbers[0]
                )
                if relative_error < 0.2:
                    return 80.0, False
            elif abs(response_numbers[0] - expected_numbers[0]) < 0.2:
                return 80.0, False

        # 检查关键答案词
        answer_keywords = self._get_answer_keywords(expected_clean)
        if answer_keywords:
            matched = sum(1 for kw in answer_keywords if kw in response_clean)
            if matched == len(answer_keywords):
                return 100.0, True
            elif matched > 0:
                return 50.0 * (matched / len(answer_keywords)), False

        return 0.0, False

    def _extract_numbers(self, text: str) -> list[float]:
        """从文本中提取数字."""
        import re

        numbers = re.findall(r"-?\d+\.?\d*", text)
        return [float(n) for n in numbers if n]

    def _get_answer_keywords(self, expected: str) -> list[str]:
        """从期望答案中提取关键词."""
        # 移除括号内的选项标记
        keywords = []
        parts = expected.split()

        for part in parts:
            # 移除 A) B) C) D) 等标记
            cleaned = part.strip("ABCDEFG).(")
            if cleaned and len(cleaned) > 1:
                keywords.append(cleaned)
            elif cleaned and cleaned not in ["A", "B", "C", "D", "E", "F"]:
                keywords.append(cleaned)

        return keywords

    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """提取推理探针特征."""
        return {
            "reasoning_score": result.functional_score,
            "passed": result.passed,
            "category": result.details.get("category", "unknown"),
            "difficulty": result.details.get("difficulty", "medium"),
            "reasoning_type": result.details.get("reasoning_type", "unknown"),
            "output_length": len(result.model_output),
        }
