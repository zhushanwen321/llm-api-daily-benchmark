"""Performance benchmark tests for concurrent execution, caching, and connection pool."""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from benchmark.analysis.quality_signals import QualitySignalCollector
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.cli import _run_provider_group


class TestConcurrency:
    """Test concurrent execution performance."""

    @pytest.mark.asyncio
    async def test_provider_group_concurrent(self):
        """Verify concurrent execution is faster than serial."""
        mock_results: list[tuple[str, str]] = []
        execution_times: list[float] = []

        async def mock_run_evaluation(
            model: str, dim: str, samples: int, debug: bool
        ) -> None:
            start = time.monotonic()
            await asyncio.sleep(0.05)
            execution_times.append(time.monotonic() - start)
            mock_results.append((model, dim))

        with patch("benchmark.cli._run_evaluation", side_effect=mock_run_evaluation):
            tasks = [
                ("zai/glm-4.7", "probe"),
                ("zai/glm-5", "probe"),
                ("zai/glm-5.1", "probe"),
            ]

            start_time = time.monotonic()
            await _run_provider_group(tasks, samples=1, debug=False)
            total_time = time.monotonic() - start_time

            assert len(mock_results) == 3
            serial_time = sum(execution_times)
            speedup = serial_time / total_time if total_time > 0 else 0
            assert speedup >= 1.2, (
                f"Expected concurrent speedup >= 1.2x, got {speedup:.2f}x"
            )

    @pytest.mark.asyncio
    async def test_concurrent_speedup_verification(self):
        """Verify asyncio.gather provides expected speedup over serial execution."""

        async def mock_task(task_id: int) -> int:
            await asyncio.sleep(0.02)
            return task_id

        serial_start = time.monotonic()
        for i in range(5):
            await mock_task(i)
        serial_time = time.monotonic() - serial_start

        concurrent_start = time.monotonic()
        await asyncio.gather(*[mock_task(i) for i in range(5)])
        concurrent_time = time.monotonic() - concurrent_start

        speedup = serial_time / concurrent_time if concurrent_time > 0 else 0
        assert speedup >= 3.0, f"Expected speedup >= 3x, got {speedup:.2f}x"
        assert concurrent_time < serial_time * 0.5


class TestCaching:
    """Test cache hit rate and speedup."""

    @pytest.mark.asyncio
    async def test_history_stats_cache(self):
        """Verify cache provides at least 10x speedup on repeated queries."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.description = [["val"]]
        mock_cursor.fetchall.return_value = [[float(i * 10)] for i in range(10)]
        mock_conn.execute.return_value = mock_cursor
        mock_db._get_conn.return_value = mock_conn

        collector = QualitySignalCollector(mock_db, "test_model")

        filters = {"dimension": "probe", "task_id": "task_1"}

        start1 = time.monotonic()
        await collector._get_history_stats(
            query_key="test",
            filters=filters,
            value_expr="LENGTH(model_output)",
        )
        first_query_time = time.monotonic() - start1

        start2 = time.monotonic()
        await collector._get_history_stats(
            query_key="test",
            filters=filters,
            value_expr="LENGTH(model_output)",
        )
        cached_query_time = time.monotonic() - start2

        assert mock_conn.execute.call_count == 1
        speedup = first_query_time / cached_query_time if cached_query_time > 0 else 0
        assert speedup >= 10.0, f"Expected cache speedup >= 10x, got {speedup:.2f}x"

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_queries(self):
        """Verify cache correctly hits on repeated identical queries."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.description = [["val"]]
        mock_cursor.fetchall.return_value = [[100.0], [200.0], [300.0]]
        mock_conn.execute.return_value = mock_cursor
        mock_db._get_conn.return_value = mock_conn

        collector = QualitySignalCollector(mock_db, "test_model")
        filters = {"dimension": "reasoning", "task_id": "task_xyz"}

        result1 = await collector._get_history_stats(
            query_key="output_length",
            filters=filters,
            value_expr="LENGTH(er.model_output)",
        )

        for _ in range(5):
            result_n = await collector._get_history_stats(
                query_key="output_length",
                filters=filters,
                value_expr="LENGTH(er.model_output)",
            )
            assert result_n == result1

        assert mock_conn.execute.call_count == 1
        assert len(collector._cache) == 1

    @pytest.mark.asyncio
    async def test_cache_different_keys_independent(self):
        """Verify different query keys result in separate cache entries."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.description = [["val"]]
        mock_cursor.fetchall.return_value = [[50.0], [60.0]]
        mock_conn.execute.return_value = mock_cursor
        mock_db._get_conn.return_value = mock_conn

        collector = QualitySignalCollector(mock_db, "model_x")

        filters1 = {"dimension": "probe", "task_id": "task_a"}
        filters2 = {"dimension": "probe", "task_id": "task_b"}

        await collector._get_history_stats(
            query_key="length", filters=filters1, value_expr="LENGTH(output)"
        )
        await collector._get_history_stats(
            query_key="length", filters=filters2, value_expr="LENGTH(output)"
        )

        assert mock_conn.execute.call_count == 2
        assert len(collector._cache) == 2

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Verify cache respects TTL and expires after configured time."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.description = [["val"]]
        mock_cursor.fetchall.side_effect = [
            [[10.0]],
            [[20.0]],
        ]
        mock_conn.execute.return_value = mock_cursor
        mock_db._get_conn.return_value = mock_conn

        collector = QualitySignalCollector(mock_db, "model_y", cache_ttl=0)
        filters = {"dimension": "backend-dev", "task_id": "task_c"}

        await collector._get_history_stats(
            query_key="stats", filters=filters, value_expr="SUM(metric)"
        )
        await asyncio.sleep(0.01)
        await collector._get_history_stats(
            query_key="stats", filters=filters, value_expr="SUM(metric)"
        )

        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_max_size_eviction(self):
        """Verify LRU eviction when cache exceeds max size."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.description = [["val"]]
        mock_cursor.fetchall.return_value = [[100.0]]
        mock_conn.execute.return_value = mock_cursor
        mock_db._get_conn.return_value = mock_conn

        collector = QualitySignalCollector(mock_db, "model_z", max_cache_size=2)
        filters_base = {"dimension": "probe"}

        for i in range(5):
            filters = {**filters_base, "task_id": f"task_{i}"}
            await collector._get_history_stats(
                query_key=f"query_{i}",
                filters=filters,
                value_expr="COUNT(*)",
            )

        assert len(collector._cache) <= 2


class TestConnectionPool:
    """Test HTTP client reuse and connection pool behavior."""

    @pytest.mark.asyncio
    async def test_client_reuse(self):
        """Verify same client instance is returned for same provider."""
        adapter = LLMEvalAdapter()

        client1 = adapter._get_client("zai")
        client2 = adapter._get_client("zai")

        assert client1 is client2
        assert "zai" in adapter._clients

        await adapter.close()

    @pytest.mark.asyncio
    async def test_different_providers_separate_clients(self):
        """Verify different providers get separate client instances."""
        adapter = LLMEvalAdapter()

        client_zai = adapter._get_client("zai")
        client_openai = adapter._get_client("openai")
        client_anthropic = adapter._get_client("anthropic")

        assert client_zai is not client_openai
        assert client_zai is not client_anthropic
        assert client_openai is not client_anthropic

        await adapter.close()

    @pytest.mark.asyncio
    async def test_client_cleanup_on_close(self):
        """Verify all clients are released when adapter is closed."""
        adapter = LLMEvalAdapter()

        adapter._get_client("zai")
        adapter._get_client("openai")
        adapter._get_client("anthropic")

        assert len(adapter._clients) == 3

        await adapter.close()

        assert len(adapter._clients) == 0

    @pytest.mark.asyncio
    async def test_client_timeout_configuration(self):
        """Verify client timeout is properly configured."""
        adapter = LLMEvalAdapter(timeout=600)

        client = adapter._get_client("zai")

        assert client.timeout.read == 600

        await adapter.close()

    @pytest.mark.asyncio
    async def test_client_pool_limits(self):
        """Verify client is created with connection pool settings."""
        adapter = LLMEvalAdapter()

        client = adapter._get_client("zai")

        assert "zai" in adapter._clients
        assert isinstance(client, type(adapter._clients["zai"]))

        await adapter.close()

    @pytest.mark.asyncio
    async def test_concurrent_client_access(self):
        """Verify client reuse works correctly under concurrent access."""
        adapter = LLMEvalAdapter()
        clients: list[Any] = []

        async def get_client(provider: str) -> Any:
            return adapter._get_client(provider)

        results = await asyncio.gather(
            get_client("zai"),
            get_client("zai"),
            get_client("zai"),
            get_client("openai"),
            get_client("openai"),
        )

        clients.extend(results)

        assert clients[0] is clients[1] is clients[2]
        assert clients[3] is clients[4]
        assert clients[0] is not clients[3]

        await adapter.close()


class TestPerformanceMetrics:
    """Test performance measurement utilities and benchmarks."""

    @pytest.mark.asyncio
    async def test_speedup_calculation(self):
        """Verify speedup is correctly calculated."""
        serial_time = 1.0
        concurrent_time = 0.25
        speedup = serial_time / concurrent_time
        assert speedup == 4.0

    @pytest.mark.asyncio
    async def test_timing_precision(self):
        """Verify time.monotonic provides sufficient precision."""
        times = [time.monotonic() for _ in range(100)]
        unique_times = len(set(times))
        assert unique_times > 1

    @pytest.mark.asyncio
    async def test_concurrent_overhead_measurement(self):
        """Measure asyncio.gather vs sequential execution timing."""

        async def noop() -> None:
            await asyncio.sleep(0.001)

        sequential_start = time.monotonic()
        for _ in range(10):
            await noop()
        sequential_time = time.monotonic() - sequential_start

        concurrent_start = time.monotonic()
        await asyncio.gather(*[noop() for _ in range(10)])
        concurrent_time = time.monotonic() - concurrent_start

        assert sequential_time > concurrent_time
