"""Uncertainty probe - Logprobs fallback solution."""

from __future__ import annotations

from typing import Any
from datetime import datetime

from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.probes import BaseProbe


class LogprobsProbe(BaseProbe):
    """Logprobs probe - estimate model uncertainty through alternative methods."""

    @property
    def frequency(self) -> str:
        return "slow"

    def load_probes(self) -> list[TaskDefinition]:
        """Load uncertainty probe definitions."""
        probes = [
            {
                "id": "uncertainty_factual_1",
                "category": "factual_confidence",
                "prompt": "What is the official language of France? Answer directly.",
                "expected": "French",
            },
            {
                "id": "uncertainty_math_1",
                "category": "math_confidence",
                "prompt": "123 * 456 = ? Only give the number.",
                "expected": "56088",
            },
            {
                "id": "uncertainty_opinion_1",
                "category": "subjective_uncertainty",
                "prompt": "Do you think AI will replace human jobs? Explain your reasoning.",
                "expected": "",
            },
            {
                "id": "uncertainty_ambiguous_1",
                "category": "ambiguous_handling",
                "prompt": "Will it rain tomorrow?",
                "expected": "",
            },
            {
                "id": "uncertainty_refusal_1",
                "category": "uncertainty_expression",
                "prompt": "How certain are you about your answer? Express as percentage.",
                "expected": "",
            },
            {
                "id": "uncertainty_multiple_1",
                "category": "multiple_choice_confidence",
                "prompt": "Who created Python? A) Bill Gates B) Guido van Rossum C) Linus Torvalds D) Tim Berners-Lee",
                "expected": "B",
            },
        ]

        return [
            TaskDefinition(
                task_id=p["id"],
                dimension="uncertainty",
                dataset=p["category"],
                prompt=p["prompt"],
                expected_output=p.get("expected", ""),
                metadata={
                    "category": p["category"],
                },
            )
            for p in probes
        ]

    async def execute_probe(
        self,
        probe: TaskDefinition,
        model: str,
        adapter: LLMEvalAdapter,
    ) -> EvalResult:
        """Execute uncertainty probe with multiple sampling."""
        responses: list[str] = []
        durations: list[float] = []

        for _ in range(3):
            response = await adapter.agenerate(
                prompt=probe.prompt,
                model=model,
                temperature=0.7,
                max_tokens=150,
            )
            responses.append(response.content)
            durations.append(response.duration)

        features = self._extract_uncertainty_features(
            responses, probe.expected_output
        )

        avg_duration = sum(durations) / len(durations)
        score = features.get("consistency_score", 0.0)

        return EvalResult(
            result_id=f"{model}_{probe.task_id}_{datetime.now().timestamp()}",
            run_id="",
            task_id=probe.task_id,
            task_content=probe.prompt,
            model_output=responses[0],
            functional_score=score,
            final_score=score,
            passed=score >= 60.0,
            execution_time=avg_duration,
            created_at=datetime.now(),
            details={
                "category": probe.metadata.get("category", "unknown"),
                "all_responses": responses,
                "uncertainty_features": features,
            },
        )

    def _extract_uncertainty_features(
        self,
        responses: list[str],
        expected: str,
    ) -> dict[str, Any]:
        """Extract uncertainty features using fallback methods."""
        features: dict[str, Any] = {
            "response_count": len(responses),
            "avg_length": sum(len(r) for r in responses) / len(responses),
            "length_variance": 0.0,
            "consistency_score": 0.0,
            "has_uncertainty_markers": False,
            "has_confidence_indicator": False,
            "uncertainty_score": 0.0,
        }

        if len(responses) > 1:
            lengths = [len(r) for r in responses]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            features["length_variance"] = variance

        if len(responses) >= 2:
            similarity_matrix = []
            for i, r1 in enumerate(responses):
                for r2 in responses[i + 1 :]:
                    sim = self._text_similarity(r1, r2)
                    similarity_matrix.append(sim)

            if similarity_matrix:
                avg_similarity = sum(similarity_matrix) / len(similarity_matrix)
                features["consistency_score"] = avg_similarity

        combined_text = " ".join(responses).lower()

        uncertainty_markers = [
            "maybe", "perhaps", "probably", "might",
            "could be", "not sure", "uncertain", "possibly",
        ]

        confidence_markers = [
            "certainly", "definitely", "absolutely", "sure",
            "100%", "completely correct", "without doubt",
        ]

        uncertainty_count = sum(1 for m in uncertainty_markers if m in combined_text)
        confidence_count = sum(1 for m in confidence_markers if m in combined_text)

        features["has_uncertainty_markers"] = uncertainty_count > 0
        features["has_confidence_indicator"] = confidence_count > 0
        features["uncertainty_marker_count"] = uncertainty_count
        features["confidence_marker_count"] = confidence_count

        if expected:
            correct_count = sum(
                1 for r in responses if expected.lower() in r.lower()
            )
            features["factual_accuracy"] = correct_count / len(responses) * 100
        else:
            features["factual_accuracy"] = None

        if features["consistency_score"] > 80:
            features["uncertainty_score"] = 20.0
        elif features["consistency_score"] > 50:
            features["uncertainty_score"] = 50.0
        else:
            features["uncertainty_score"] = 80.0

        if features["has_uncertainty_markers"]:
            features["uncertainty_score"] = min(
                features["uncertainty_score"] + 10, 100
            )

        return features

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard."""
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return 100.0

        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)
        return jaccard * 100

    def extract_features(self, result: EvalResult) -> dict[str, Any]:
        """Extract uncertainty features from result."""
        uncertainty_features = result.details.get("uncertainty_features", {})

        return {
            "uncertainty_score": uncertainty_features.get("uncertainty_score", 0),
            "consistency_score": uncertainty_features.get("consistency_score", 0),
            "category": result.details.get("category", "unknown"),
            "has_uncertainty_markers": uncertainty_features.get(
                "has_uncertainty_markers", False
            ),
            "factual_accuracy": uncertainty_features.get("factual_accuracy"),
        }
