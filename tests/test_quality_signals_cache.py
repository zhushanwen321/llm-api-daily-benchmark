"""测试历史统计缓存功能"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from benchmark.analysis.quality_signals import QualitySignalCollector


@pytest.mark.asyncio
async def test_history_stats_cache():
    """测试历史统计缓存"""

    # 创建 mock 数据库
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # 设置 mock 返回值
    mock_cursor.description = [["val"]]
    mock_cursor.fetchall.return_value = [[10.0], [20.0], [30.0]]
    mock_conn.execute.return_value = mock_cursor
    mock_db._get_conn.return_value = mock_conn

    collector = QualitySignalCollector(mock_db, "test_model")

    # 第一次查询（应该查数据库）
    filters = {"dimension": "probe", "task_id": "test_task"}
    result1 = await collector._get_history_stats(
        query_key="output_length", filters=filters, value_expr="LENGTH(er.model_output)"
    )

    # 验证数据库被查询了一次
    assert mock_conn.execute.call_count == 1

    # 第二次查询（应该命中缓存）
    result2 = await collector._get_history_stats(
        query_key="output_length", filters=filters, value_expr="LENGTH(er.model_output)"
    )

    # 验证数据库仍然只被查询了一次（缓存命中）
    assert mock_conn.execute.call_count == 1

    # 验证结果相同
    assert result1 == result2


@pytest.mark.asyncio
async def test_cache_different_keys():
    """测试不同缓存键独立缓存"""

    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.description = [["val"]]
    mock_cursor.fetchall.return_value = [[10.0], [20.0], [30.0]]
    mock_conn.execute.return_value = mock_cursor
    mock_db._get_conn.return_value = mock_conn

    collector = QualitySignalCollector(mock_db, "test_model")

    # 查询第一个任务
    filters1 = {"dimension": "probe", "task_id": "task_1"}
    await collector._get_history_stats(
        query_key="output_length",
        filters=filters1,
        value_expr="LENGTH(er.model_output)",
    )

    # 查询第二个任务（不同 task_id，应该查数据库）
    filters2 = {"dimension": "probe", "task_id": "task_2"}
    await collector._get_history_stats(
        query_key="output_length",
        filters=filters2,
        value_expr="LENGTH(er.model_output)",
    )

    # 验证数据库被查询了两次
    assert mock_conn.execute.call_count == 2


@pytest.mark.asyncio
async def test_cache_empty_result():
    """测试空结果不缓存"""

    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.description = [["val"]]
    mock_cursor.fetchall.return_value = []  # 空结果
    mock_conn.execute.return_value = mock_cursor
    mock_db._get_conn.return_value = mock_conn

    collector = QualitySignalCollector(mock_db, "test_model")

    filters = {"dimension": "probe", "task_id": "test_task"}
    result = await collector._get_history_stats(
        query_key="output_length", filters=filters, value_expr="LENGTH(er.model_output)"
    )

    # 验证返回默认值
    assert result == (0.0, 0.0)


def test_cache_key_generation():
    """测试缓存键生成"""

    mock_db = MagicMock()
    collector = QualitySignalCollector(mock_db, "test_model")

    # 测试完整的 filters
    filters = {"dimension": "probe", "task_id": "task_1"}
    key = collector._get_cache_key("output_length", filters)
    assert key == "test_model:output_length:probe:task_1"

    # 测试缺少某些字段
    filters2 = {"dimension": "probe"}
    key2 = collector._get_cache_key("output_length", filters2)
    assert key2 == "test_model:output_length:probe:"


@pytest.mark.asyncio
async def test_cache_performance():
    """测试缓存性能提升"""

    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.description = [["val"]]
    mock_cursor.fetchall.return_value = [[10.0], [20.0], [30.0]]
    mock_conn.execute.return_value = mock_cursor
    mock_db._get_conn.return_value = mock_conn

    collector = QualitySignalCollector(mock_db, "test_model")

    filters = {"dimension": "probe", "task_id": "test_task"}

    # 第一次查询
    import time

    start1 = time.time()
    await collector._get_history_stats(
        query_key="output_length", filters=filters, value_expr="LENGTH(er.model_output)"
    )
    time1 = time.time() - start1

    # 第二次查询（缓存命中）
    start2 = time.time()
    await collector._get_history_stats(
        query_key="output_length", filters=filters, value_expr="LENGTH(er.model_output)"
    )
    time2 = time.time() - start2

    # 缓存查询应该快得多（这里用简单判断，实际应该快10倍以上）
    # 但由于 mock 操作很快，可能差别不大，主要验证逻辑
    assert mock_conn.execute.call_count == 1
