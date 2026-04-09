"""Logprobs probe unit tests."""

from __future__ import annotations

import pytest
from datetime import datetime

from benchmark.probes.uncertainty.logprobs_probe import LogprobsProbe
from benchmark.models.schemas import TaskDefinition, EvalResult


class TestLogprobsProbe:
    """Logprobs probe test class."""

    def test_load_probes(self):
        """Test loading uncertainty probes."""
        probe = LogprobsProbe()
        tasks = probe.load_probes()

        assert len(tasks) == 6
        assert all(isinstance(t, TaskDefinition) for t in tasks)
        assert all(t.dimension == "uncertainty" for t in tasks)

        categories = {t.dataset for t in tasks}
        assert "factual_confidence" in categories
        assert "math_confidence" in categories

    def test_frequency_property(self):
        """Test frequency property."""
        probe = LogprobsProbe()
        assert probe.frequency == "slow"

    def test_text_similarity_identical(self):
        """Test similarity for identical texts."""
        probe = LogprobsProbe()
        sim = probe._text_similarity("hello world", "hello world")
        assert sim == 100.0

    def test_text_similarity_partial(self):
        """Test similarity for partially matching texts."""
        probe = LogprobsProbe()
        sim = probe._text_similarity("hello world test", "hello world")
        assert 0 < sim < 100

    def test_text_similarity_no_match(self):
        """Test similarity for non-matching texts."""
        probe = LogprobsProbe()
        sim = probe._text_similarity("abc", "xyz")
        assert sim == 0.0

    def test_extract_uncertainty_features_consistency(self):
        """Test consistency feature extraction."""
        probe = LogprobsProbe()

        responses = ["Answer A", "Answer A", "Answer B"]
        features = probe._extract_uncertainty_features(responses, "")

        assert features["response_count"] == 3
        assert 0 < features["consistency_score"] < 100
        assert features["avg_length"] > 0

    def test_extract_uncertainty_features_perfect_consistency(self):
        """Test perfect consistency."""
        probe = LogprobsProbe()

        responses = ["Same answer", "Same answer", "Same answer"]
        features = probe._extract_uncertainty_features(responses, "")

        assert features["consistency_score"] == 100.0
        assert features["uncertainty_score"] == 20.0

    def test_extract_uncertainty_features_low_consistency(self):
        """Test low consistency."""
        probe = LogprobsProbe()

        responses = ["Completely different A", "Totally unrelated B", "Nothing alike C"]
        features = probe._extract_uncertainty_features(responses, "")

        assert features["consistency_score"] < 50
        assert features["uncertainty_score"] >= 80

    def test_extract_uncertainty_features_with_uncertainty_markers(self):
        """Test detection of uncertainty markers."""
        probe = LogprobsProbe()

        responses = ["Maybe it is correct", "Perhaps not sure"]
        features = probe._extract_uncertainty_features(responses, "")

        assert features["has_uncertainty_markers"] is True
        assert features["uncertainty_marker_count"] > 0

    def test_extract_uncertainty_features_with_factual_expected(self):
        """Test factual accuracy detection."""
        probe = LogprobsProbe()

        responses = ["The answer is French", "French is correct", "It is English"]
        features = probe._extract_uncertainty_features(responses, "French")

        assert features["factual_accuracy"] == pytest.approx(66.67, 0.1)

    def test_extract_uncertainty_features_empty_expected(self):
        """Test with no expected answer."""
        probe = LogprobsProbe()

        responses = ["Any answer", "Another response"]
        features = probe._extract_uncertainty_features(responses, "")

        assert features["factual_accuracy"] is None

    def test_extract_features(self):
        """Test feature extraction method."""
        probe = LogprobsProbe()

        result = EvalResult(
            result_id="test_1",
            run_id="run_1",
            task_id="uncertainty_factual_1",
            task_content="test prompt",
            model_output="French",
            functional_score=85.0,
            final_score=85.0,
            passed=True,
            execution_time=1.0,
            created_at=datetime.now(),
            details={
                "category": "factual_confidence",
                "uncertainty_features": {
                    "uncertainty_score": 30.0,
                    "consistency_score": 70.0,
                    "has_uncertainty_markers": False,
                    "factual_accuracy": 100.0,
                },
            },
        )

        features = probe.extract_features(result)

        assert features["uncertainty_score"] == 30.0
        assert features["consistency_score"] == 70.0
        assert features["category"] == "factual_confidence"
        assert features["has_uncertainty_markers"] is False
        assert features["factual_accuracy"] == 100.0

    @pytest.mark.asyncio
    async def test_execute_probe(self):
        """Test probe execution."""
        from unittest.mock import AsyncMock
        from benchmark.models.schemas import GenerateResponse

        mock_adapter = AsyncMock()
        mock_adapter.agenerate.return_value = GenerateResponse(
            content="French",
            duration=1.0,
        )

        probe = LogprobsProbe()

        task = TaskDefinition(
            task_id="uncertainty_factual_1",
            dimension="uncertainty",
            dataset="factual_confidence",
            prompt="What is the language?",
            expected_output="French",
            metadata={
                "category": "factual_confidence",
            },
        )

        result = await probe.execute_probe(task, "test_model", mock_adapter)

        assert result.task_id == "uncertainty_factual_1"
        assert result.details["category"] == "factual_confidence"
        assert "uncertainty_features" in result.details
        assert "all_responses" in result.details
        assert len(result.details["all_responses"]) == 3
        mock_adapter.agenerate.assert_called()
