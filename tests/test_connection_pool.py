"""测试 HTTP 连接池功能"""

import pytest
import httpx
from unittest.mock import patch, AsyncMock
from benchmark.core.llm_adapter import LLMEvalAdapter


@pytest.mark.asyncio
async def test_client_reuse():
    """测试 HTTP 客户端复用"""

    adapter = LLMEvalAdapter()

    # 第一次获取 client
    client1 = adapter._get_client("zai")

    # 第二次获取 client（应该复用）
    client2 = adapter._get_client("zai")

    # 验证是同一个实例
    assert client1 is client2

    # 验证是 httpx.AsyncClient 类型
    assert isinstance(client1, httpx.AsyncClient)

    # 清理
    await adapter.close()


@pytest.mark.asyncio
async def test_different_providers():
    """测试不同 provider 使用不同 client"""

    adapter = LLMEvalAdapter()

    # 获取不同 provider 的 client
    client1 = adapter._get_client("zai")
    client2 = adapter._get_client("openai")

    # 验证是不同的实例
    assert client1 is not client2

    # 清理
    await adapter.close()


@pytest.mark.asyncio
async def test_connection_pool_limits():
    """测试连接池限制"""

    adapter = LLMEvalAdapter()
    client = adapter._get_client("zai")

    # 验证客户端已创建
    assert isinstance(client, httpx.AsyncClient)
    assert "zai" in adapter._clients

    # 清理
    await adapter.close()


@pytest.mark.asyncio
async def test_close_releases_connections():
    """测试关闭释放连接"""

    adapter = LLMEvalAdapter()

    # 获取 client
    client = adapter._get_client("zai")

    # 关闭
    await adapter.close()

    # 验证 clients 被清空
    assert len(adapter._clients) == 0


@pytest.mark.asyncio
async def test_client_timeout_config():
    """测试客户端超时配置"""

    adapter = LLMEvalAdapter(timeout=600)
    client = adapter._get_client("zai")

    # 验证超时配置
    assert client.timeout.read == 600

    # 清理
    await adapter.close()
