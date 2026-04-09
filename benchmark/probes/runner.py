"""Probe runner - executes probes and collects results."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from datetime import datetime

from benchmark.probes import BaseProbe
from benchmark.probes.registry import ProbeRegistry
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.models.schemas import EvalResult

logger = logging.getLogger(__name__)


class ProbeRunner:
    """Runner for executing probes and collecting results."""

    def __init__(self, adapter: LLMEvalAdapter | None = None) -> None:
        """Initialize runner with optional adapter."""
        self.registry = ProbeRegistry()
        self.adapter = adapter or LLMEvalAdapter()
        self.results: list[EvalResult] = []

    async def run_probe(
        self,
        probe_name: str,
        model: str,
        limit: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> list[EvalResult]:
        """Run a single probe type with timeout."""
        probe = self.registry.get(probe_name)
        if not probe:
            raise ValueError(f"Probe not found: {probe_name}")

        tasks = probe.load_probes()
        if limit:
            tasks = tasks[:limit]

        async def run_with_timeout(task):
            try:
                return await asyncio.wait_for(
                    probe.execute_probe(task, model, self.adapter),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Probe task {task.task_id} timed out after {timeout_seconds}s"
                )
                return EvalResult(
                    result_id=f"{model}_{task.task_id}_timeout",
                    run_id="",
                    task_id=task.task_id,
                    task_content=task.prompt,
                    model_output="[TIMEOUT]",
                    functional_score=0.0,
                    final_score=0.0,
                    passed=False,
                    execution_time=timeout_seconds,
                    created_at=datetime.now(),
                    details={"error": f"Timeout after {timeout_seconds}s"},
                )
            except Exception as e:
                logger.error(f"Probe task {task.task_id} failed: {e}")
                return EvalResult(
                    result_id=f"{model}_{task.task_id}_error",
                    run_id="",
                    task_id=task.task_id,
                    task_content=task.prompt,
                    model_output=f"[ERROR: {str(e)}]",
                    functional_score=0.0,
                    final_score=0.0,
                    passed=False,
                    execution_time=0.0,
                    created_at=datetime.now(),
                    details={"error": str(e)},
                )

        # Concurrent execution with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)

        async def run_with_semaphore(task):
            async with semaphore:
                return await run_with_timeout(task)

        results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
        return list(results)

    async def run_all_probes(
        self,
        model: str,
        include_frequencies: list[str] | None = None,
    ) -> dict[str, list[EvalResult]]:
        """Run all registered probes."""
        all_probes = self.registry.get_all_probes()
        results: dict[str, list[EvalResult]] = {}

        for name, probe in all_probes.items():
            if include_frequencies and probe.frequency not in include_frequencies:
                continue

            probe_results = await self.run_probe(name, model)
            results[name] = probe_results

        return results

    async def run_probe_by_frequency(
        self,
        frequency: str,
        model: str,
    ) -> dict[str, list[EvalResult]]:
        """Run all probes with specific frequency."""
        probes = self.registry.get_by_frequency(frequency)
        results: dict[str, list[EvalResult]] = {}

        for name, probe in probes.items():
            tasks = probe.load_probes()
            probe_results: list[EvalResult] = []
            for task in tasks:
                result = await probe.execute_probe(task, model, self.adapter)
                probe_results.append(result)
            results[name] = probe_results

        return results

    def get_results_summary(
        self, results: dict[str, list[EvalResult]]
    ) -> dict[str, Any]:
        """Generate summary statistics from results."""
        summary = {
            "total_probes": len(results),
            "total_tasks": 0,
            "passed_tasks": 0,
            "failed_tasks": 0,
            "probe_summaries": {},
        }

        for probe_name, probe_results in results.items():
            passed = sum(1 for r in probe_results if r.passed)
            total = len(probe_results)
            avg_score = (
                sum(r.functional_score for r in probe_results) / total
                if total > 0
                else 0
            )

            summary["total_tasks"] += total
            summary["passed_tasks"] += passed
            summary["failed_tasks"] += total - passed

            summary["probe_summaries"][probe_name] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total * 100 if total > 0 else 0,
                "avg_score": avg_score,
            }

        if summary["total_tasks"] > 0:
            summary["overall_pass_rate"] = (
                summary["passed_tasks"] / summary["total_tasks"] * 100
            )
        else:
            summary["overall_pass_rate"] = 0

        return summary

    async def __aenter__(self) -> ProbeRunner:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup adapter."""
        await self.adapter.close()
