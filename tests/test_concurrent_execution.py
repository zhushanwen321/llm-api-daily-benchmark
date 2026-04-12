"""测试并发执行功能 - 适配 FileRepository."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from benchmark.cli.runner import run_provider_group as _run_provider_group
from benchmark.repository import FileRepository


@pytest.mark.asyncio
async def test_concurrent_execution(tmp_path):
    """测试同一 provider 的并发执行"""

    mock_results = []

    async def mock_run_evaluation(model, dim, samples, debug, repo):
        """模拟评测函数"""
        await asyncio.sleep(0.1)  # 模拟耗时操作
        mock_results.append((model, dim))

    repo = FileRepository(data_root=tmp_path)

    with patch("benchmark.cli._run_evaluation", side_effect=mock_run_evaluation):
        tasks = [
            ("zai/glm-4.7", "probe"),
            ("zai/glm-5", "probe"),
            ("zai/glm-5.1", "probe"),
        ]

        start_time = asyncio.get_event_loop().time()
        await _run_provider_group(tasks, samples=1, debug=False, repo=repo)
        end_time = asyncio.get_event_loop().time()

        duration = end_time - start_time

        # 验证所有任务都执行了
        assert len(mock_results) == 3

        # 验证并发执行（串行需要 0.3s，并发应该 < 0.2s）
        assert duration < 0.25, f"Expected concurrent execution, but took {duration}s"


@pytest.mark.asyncio
async def test_exception_handling(tmp_path):
    """测试异常处理"""

    call_count = 0

    async def mock_run_evaluation_with_error(model, dim, samples, debug, repo):
        """模拟会失败的评测函数"""
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("Test error")
        await asyncio.sleep(0.1)

    repo = FileRepository(data_root=tmp_path)

    with patch(
        "benchmark.cli._run_evaluation", side_effect=mock_run_evaluation_with_error
    ):
        tasks = [
            ("zai/glm-4.7", "probe"),
            ("zai/glm-5", "probe"),  # 这个会失败
            ("zai/glm-5.1", "probe"),
        ]

        # 不应该抛出异常，而是应该处理异常并继续
        await _run_provider_group(tasks, samples=1, debug=False, repo=repo)

        # 验证所有任务都被调用了（包括失败的）
        assert call_count == 3


def test_get_provider_concurrency():
    """测试获取并发限制"""
    from benchmark.cli import _get_provider_concurrency

    # 测试默认情况
    concurrency = _get_provider_concurrency("unknown/model")
    assert concurrency == 2  # 默认值
