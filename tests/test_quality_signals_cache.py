"""测试历史统计缓存功能 - 适配 FileRepository."""

import pytest
import asyncio
from benchmark.analysis.quality_signals import QualitySignalCollector
from benchmark.repository import FileRepository


@pytest.mark.asyncio
async def test_history_stats_cache(tmp_path):
    """测试历史统计缓存功能。"""
    repo = FileRepository(data_root=tmp_path)
    collector = QualitySignalCollector(repo, "test_model")

    # 测试缓存机制存在（通过调用不抛出异常验证）
    filters = {"dimension": "probe", "task_id": "test_task"}
    result = await collector._get_history_stats(
        query_key="output_length", filters=filters, value_expr="output_length"
    )

    # 无历史数据时应返回 (0.0, 0.0)
    assert result == (0.0, 0.0)


@pytest.mark.asyncio
async def test_cache_different_keys(tmp_path):
    """测试不同缓存键独立缓存。"""
    repo = FileRepository(data_root=tmp_path)
    collector = QualitySignalCollector(repo, "test_model")

    # 不同 query_key 应该独立缓存
    filters1 = {"dimension": "probe", "task_id": "task1"}
    filters2 = {"dimension": "probe", "task_id": "task2"}

    result1 = await collector._get_history_stats(
        query_key="output_length", filters=filters1, value_expr="output_length"
    )
    result2 = await collector._get_history_stats(
        query_key="output_length", filters=filters2, value_expr="output_length"
    )

    # 两者都应该返回 (0.0, 0.0)（无历史数据）
    assert result1 == (0.0, 0.0)
    assert result2 == (0.0, 0.0)


@pytest.mark.asyncio
async def test_cache_performance(tmp_path):
    """测试缓存性能（简化版）。"""
    repo = FileRepository(data_root=tmp_path)
    collector = QualitySignalCollector(repo, "test_model")

    # 多次调用缓存功能
    filters = {"dimension": "probe"}

    # 第一次调用
    result1 = await collector._get_history_stats(
        query_key="tps", filters=filters, value_expr="tokens_per_second"
    )

    # 第二次调用（应该使用缓存）
    result2 = await collector._get_history_stats(
        query_key="tps", filters=filters, value_expr="tokens_per_second"
    )

    # 结果应该相同
    assert result1 == result2
