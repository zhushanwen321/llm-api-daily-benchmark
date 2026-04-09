"""Probe integration tests."""

from __future__ import annotations

import pytest
from datetime import datetime

from benchmark.probes.registry import ProbeRegistry
from benchmark.probes.runner import ProbeRunner
from benchmark.probes import BaseProbe
from benchmark.models.schemas import TaskDefinition, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter


class MockProbe(BaseProbe):
    """Mock probe for testing."""

    @property
    def frequency(self) -> str:
        return "fast"

    def load_probes(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                task_id="mock_1",
                dimension="mock",
                dataset="test",
                prompt="Test prompt",
                expected_output="expected",
                metadata={},
            ),
        ]

    async def execute_probe(
        self,
        probe: TaskDefinition,
        model: str,
        adapter: LLMEvalAdapter,
    ) -> EvalResult:
        return EvalResult(
            result_id=f"{model}_{probe.task_id}",
            run_id="",
            task_id=probe.task_id,
            task_content=probe.prompt,
            model_output="output",
            functional_score=85.0,
            final_score=85.0,
            passed=True,
            execution_time=1.0,
            created_at=datetime.now(),
            details={},
        )

    def extract_features(self, result: EvalResult) -> dict:
        return {"score": result.functional_score}


class TestProbeRegistry:
    """Test probe registry."""

    def test_registry_singleton(self):
        """Test registry is singleton."""
        reg1 = ProbeRegistry()
        reg2 = ProbeRegistry()
        assert reg1 is reg2

    def test_default_probes_registered(self):
        """Test default probes are auto-registered."""
        registry = ProbeRegistry()
        probes = registry.list_probes()

        assert "safety" in probes
        assert "fingerprint" in probes
        assert "consistency" in probes
        assert "uncertainty" in probes

    def test_get_probe(self):
        """Test getting probe by name."""
        registry = ProbeRegistry()

        safety = registry.get("safety")
        assert safety is not None
        assert safety.frequency == "medium"

        fingerprint = registry.get("fingerprint")
        assert fingerprint is not None
        assert fingerprint.frequency == "slow"

    def test_get_nonexistent_probe(self):
        """Test getting non-existent probe returns None."""
        registry = ProbeRegistry()
        probe = registry.get("nonexistent")
        assert probe is None

    def test_register_probe(self):
        """Test registering custom probe."""
        registry = ProbeRegistry()
        mock_probe = MockProbe()

        registry.register("mock", mock_probe)

        assert "mock" in registry.list_probes()
        assert registry.get("mock") is mock_probe

    def test_get_by_frequency(self):
        """Test filtering probes by frequency."""
        registry = ProbeRegistry()

        slow_probes = registry.get_by_frequency("slow")
        assert "fingerprint" in slow_probes
        assert "consistency" in slow_probes
        assert "uncertainty" in slow_probes
        assert "safety" not in slow_probes

    def test_get_probe_summary(self):
        """Test getting probe summary."""
        registry = ProbeRegistry()
        summary = registry.get_probe_summary()

        assert "safety" in summary
        assert "task_count" in summary["safety"]
        assert "frequency" in summary["safety"]
        assert "categories" in summary["safety"]


class TestProbeRunner:
    """Test probe runner."""

    @pytest.mark.asyncio
    async def test_runner_initialization(self):
        """Test runner initialization."""
        runner = ProbeRunner()
        assert runner.registry is not None
        assert runner.adapter is not None

    @pytest.mark.asyncio
    async def test_run_single_probe(self):
        """Test running single probe."""
        from unittest.mock import AsyncMock, patch
        from benchmark.models.schemas import GenerateResponse

        mock_response = GenerateResponse(
            content="Safe response",
            duration=1.0,
        )

        with patch.object(LLMEvalAdapter, 'agenerate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response

            runner = ProbeRunner()
            results = await runner.run_probe("safety", "test_model", limit=2)

            assert len(results) == 2
            assert all(isinstance(r, EvalResult) for r in results)

    @pytest.mark.asyncio
    async def test_run_invalid_probe(self):
        """Test running invalid probe raises error."""
        runner = ProbeRunner()

        with pytest.raises(ValueError, match="Probe not found"):
            await runner.run_probe("nonexistent", "test_model")

    @pytest.mark.asyncio
    async def test_run_all_probes(self):
        """Test running all probes."""
        from unittest.mock import AsyncMock, patch
        from benchmark.models.schemas import GenerateResponse

        mock_response = GenerateResponse(
            content="Response",
            duration=1.0,
        )

        with patch.object(LLMEvalAdapter, 'agenerate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response

            runner = ProbeRunner()
            results = await runner.run_all_probes("test_model")

            assert "safety" in results
            assert "fingerprint" in results
            assert all(isinstance(r, EvalResult) for probe_results in results.values() for r in probe_results)

    @pytest.mark.asyncio
    async def test_run_by_frequency(self):
        """Test running probes by frequency."""
        from unittest.mock import AsyncMock, patch
        from benchmark.models.schemas import GenerateResponse

        mock_response = GenerateResponse(
            content="Response",
            duration=1.0,
        )

        with patch.object(LLMEvalAdapter, 'agenerate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response

            runner = ProbeRunner()
            results = await runner.run_probe_by_frequency("slow", "test_model")

            assert "fingerprint" in results
            assert "consistency" in results
            assert "uncertainty" in results
            assert "safety" not in results

    def test_results_summary(self):
        """Test results summary generation."""
        runner = ProbeRunner()

        results = {
            "probe1": [
                EvalResult(
                    result_id="1", run_id="", task_id="t1",
                    task_content="", model_output="",
                    functional_score=80.0, final_score=80.0,
                    passed=True, execution_time=1.0,
                    created_at=datetime.now(), details={},
                ),
                EvalResult(
                    result_id="2", run_id="", task_id="t2",
                    task_content="", model_output="",
                    functional_score=40.0, final_score=40.0,
                    passed=False, execution_time=1.0,
                    created_at=datetime.now(), details={},
                ),
            ],
        }

        summary = runner.get_results_summary(results)

        assert summary["total_probes"] == 1
        assert summary["total_tasks"] == 2
        assert summary["passed_tasks"] == 1
        assert summary["failed_tasks"] == 1
        assert summary["overall_pass_rate"] == 50.0
        assert summary["probe_summaries"]["probe1"]["pass_rate"] == 50.0

    def test_empty_results_summary(self):
        """Test summary with empty results."""
        runner = ProbeRunner()
        summary = runner.get_results_summary({})

        assert summary["total_probes"] == 0
        assert summary["total_tasks"] == 0
        assert summary["overall_pass_rate"] == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        from unittest.mock import AsyncMock, patch

        with patch.object(LLMEvalAdapter, 'close', new_callable=AsyncMock) as mock_close:
            async with ProbeRunner() as runner:
                assert runner is not None

            mock_close.assert_called_once()
