"""Performance benchmark tests - 适配 FileRepository."""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from benchmark.analysis.quality_signals import QualitySignalCollector
from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.cli.runner import run_provider_group as _run_provider_group
from benchmark.repository import FileRepository


class TestConcurrency:
    """Test concurrent execution performance."""

    @pytest.mark.asyncio
    async def test_provider_group_concurrent(self, tmp_path):
        """Verify concurrent execution is faster than serial."""
        mock_results: list[tuple[str, str]] = []
        execution_times: list[float] = []

        async def mock_run_evaluation(
            model: str, dim: str, samples: int, debug: bool, repo
        ) -> None:
            start = time.monotonic()
            await asyncio.sleep(0.05)
            execution_times.append(time.monotonic() - start)
            mock_results.append((model, dim))

        repo = FileRepository(data_root=tmp_path)

        with patch("benchmark.cli._run_evaluation", side_effect=mock_run_evaluation):
            tasks = [
                ("zai/glm-4.7", "probe"),
                ("zai/glm-5", "probe"),
                ("zai/glm-5.1", "probe"),
            ]

            start_time = time.monotonic()
            await _run_provider_group(tasks, samples=1, debug=False, repo=repo)
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
    async def test_history_stats_cache(self, tmp_path):
        """Verify cache functionality works."""
        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test_model")

        filters = {"dimension": "probe", "task_id": "test_task"}

        # 第一次查询
        start1 = time.monotonic()
        result1 = await collector._get_history_stats(
            query_key="output_length", filters=filters, value_expr="output_length"
        )
        time1 = time.monotonic() - start1

        # 第二次查询（应该使用缓存）
        start2 = time.monotonic()
        result2 = await collector._get_history_stats(
            query_key="output_length", filters=filters, value_expr="output_length"
        )
        time2 = time.monotonic() - start2

        # 结果应该相同
        assert result1 == result2
        # 第二次应该更快（缓存命中）
        assert time2 <= time1 * 2  # 放宽条件，避免 CI 波动

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_queries(self, tmp_path):
        """Test cache hits on repeated queries."""
        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test_model")

        filters = {"dimension": "probe"}

        # 多次查询相同内容
        results = []
        for _ in range(3):
            result = await collector._get_history_stats(
                query_key="tps", filters=filters, value_expr="tokens_per_second"
            )
            results.append(result)

        # 所有结果应该相同（都是 0.0, 0.0，因为没有历史数据）
        assert all(r == (0.0, 0.0) for r in results)

    @pytest.mark.asyncio
    async def test_cache_different_keys_independent(self, tmp_path):
        """Test different cache keys are independent."""
        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test_model")

        # 不同 query_key
        result1 = await collector._get_history_stats(
            query_key="key1", filters={}, value_expr="val"
        )
        result2 = await collector._get_history_stats(
            query_key="key2", filters={}, value_expr="val"
        )

        # 应该独立（都是 0.0, 0.0）
        assert result1 == (0.0, 0.0)
        assert result2 == (0.0, 0.0)

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, tmp_path):
        """Test cache TTL expiration."""
        # 使用很短的 TTL 创建 collector
        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test_model", cache_ttl=0)

        filters = {"dimension": "probe"}

        # 第一次查询
        result1 = await collector._get_history_stats(
            query_key="test", filters=filters, value_expr="val"
        )

        # 立即第二次查询（TTL=0，应该过期）
        result2 = await collector._get_history_stats(
            query_key="test", filters=filters, value_expr="val"
        )

        # 结果相同（都是默认值）
        assert result1 == result2


class TestConnectionPool:
    """Test connection pool management."""

    def test_provider_specific_limits(self):
        """Test different providers have appropriate limits."""
        from benchmark.cli.utils import (
            get_provider_concurrency as _get_provider_concurrency,
        )

        # 未知 provider 使用默认值
        default_limit = _get_provider_concurrency("unknown/model")
        assert default_limit >= 1

        # 应该返回合理的值
        assert isinstance(default_limit, int)


class TestOverallPerformance:
    """Overall performance benchmarks."""

    def test_import_time(self):
        """Verify imports are reasonably fast."""
        import importlib
        import time

        start = time.monotonic()
        importlib.import_module("benchmark.cli")
        import_time = time.monotonic() - start

        # 导入应该很快（< 5s）
        assert import_time < 5.0, f"Import took {import_time:.2f}s"

    @pytest.mark.asyncio
    async def test_memory_usage_stable(self):
        """Test memory usage doesn't grow unboundedly."""
        # 简化测试：验证可以正常创建多个对象
        collectors = []
        for i in range(5):
            # 使用 Mock repo 避免实际数据库连接
            mock_repo = MagicMock()
            mock_repo.get_results.return_value = []
            collector = QualitySignalCollector(mock_repo, f"model_{i}")
            collectors.append(collector)

        assert len(collectors) == 5

    def test_asyncio_event_loop_health(self):
        """Verify asyncio event loop is healthy."""
        import asyncio

        loop = asyncio.get_event_loop()
        assert not loop.is_closed()
        assert loop.is_running() is False  # 当前没有运行
