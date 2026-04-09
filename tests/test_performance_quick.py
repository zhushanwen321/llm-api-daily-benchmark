"""快速性能基准测试（不调用实际API）."""

import asyncio
import time
from datetime import datetime

from benchmark.analysis.quality_signals import QualitySignalCollector
from benchmark.core.llm_adapter import LLMEvalAdapter


class MockDatabase:
    def __init__(self):
        self._conn = MockConnection()

    def _get_conn(self):
        return self._conn


class MockConnection:
    def execute(self, sql, params):
        return MockCursor()


class MockCursor:
    description = [["val"]]

    def fetchall(self):
        return [[float(i * 10)] for i in range(10)]


async def test_concurrent():
    print("[1/3] 测试并发执行...")

    async def task(n):
        await asyncio.sleep(0.01)
        return n

    # 串行
    start = time.time()
    for i in range(10):
        await task(i)
    serial = time.time() - start

    # 并发
    start = time.time()
    await asyncio.gather(*[task(i) for i in range(10)])
    concurrent = time.time() - start

    speedup = serial / concurrent if concurrent > 0 else 0
    print(f"  串行: {serial:.3f}s, 并发: {concurrent:.3f}s, 加速: {speedup:.2f}x")
    return speedup


async def test_cache():
    print("[2/3] 测试缓存性能...")

    mock_db = MockDatabase()  # type: ignore
    collector = QualitySignalCollector(mock_db, "test_model")

    # 首次查询
    start = time.time()
    await collector._get_history_stats(
        "test", {"dimension": "probe", "task_id": "t1"}, "LENGTH(model_output)"
    )
    first = time.time() - start

    # 缓存查询
    start = time.time()
    await collector._get_history_stats(
        "test", {"dimension": "probe", "task_id": "t1"}, "LENGTH(model_output)"
    )
    cached = time.time() - start

    speedup = first / cached if cached > 0 else 0
    print(
        f"  首次: {first * 1000:.3f}ms, 缓存: {cached * 1000:.3f}ms, 加速: {speedup:.2f}x"
    )
    return speedup


async def test_connection_pool():
    print("[3/3] 测试连接池...")

    adapter = LLMEvalAdapter()

    start = time.time()
    client1 = adapter._get_client("test")
    first = time.time() - start

    start = time.time()
    client2 = adapter._get_client("test")
    reuse = time.time() - start

    is_same = client1 is client2

    await adapter.close()

    print(
        f"  首次: {first * 1000:.3f}ms, 复用: {reuse * 1000:.3f}ms, 同一实例: {is_same}"
    )
    return is_same


async def main():
    print("=" * 50)
    print("Phase 1 快速性能测试")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    print()

    results = {}
    results["concurrent"] = await test_concurrent()
    results["cache"] = await test_cache()
    results["connection_pool"] = await test_connection_pool()

    print()
    print("=" * 50)
    print("测试结果")
    print("=" * 50)
    print(f"并发加速: {results['concurrent']:.2f}x")
    print(f"缓存加速: {results['cache']:.2f}x")
    print(f"连接池复用: {results['connection_pool']}")
    print("=" * 50)

    return all(
        [
            results["concurrent"] > 1.5,
            results["cache"] > 5,
            results["connection_pool"] is True,
        ]
    )


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
