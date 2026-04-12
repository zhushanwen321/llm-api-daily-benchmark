"""快速性能基准测试（不调用实际API）."""

import asyncio
import time
from datetime import datetime

from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.repository import FileRepository


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


async def test_file_repo_cache():
    """测试 FileRepository 重复读取的缓存效果."""
    print("[2/3] 测试 FileRepository 读取性能...")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        repo = FileRepository(data_root=tmp)

        # 预写数据
        from benchmark.models.schemas import EvalRun

        run_id = repo.create_run(
            EvalRun(
                run_id="",
                model="test_model",
                dimension="test",
                dataset="test",
                started_at=datetime.now(),
                status="running",
            )
        )

        # 首次查询
        start = time.time()
        runs1 = repo.get_runs(model="test_model")
        first = time.time() - start

        # 二次查询
        start = time.time()
        runs2 = repo.get_runs(model="test_model")
        cached = time.time() - start

        speedup = first / cached if cached > 0 else 0
        print(
            f"  首次: {first * 1000:.3f}ms, 二次: {cached * 1000:.3f}ms, 加速: {speedup:.2f}x"
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
    results["cache"] = await test_file_repo_cache()
    results["connection_pool"] = await test_connection_pool()

    print()
    print("=" * 50)
    print("测试结果")
    print("=" * 50)
    print(f"并发加速: {results['concurrent']:.2f}x")
    print(f"FileRepository 加速: {results['cache']:.2f}x")
    print(f"连接池复用: {results['connection_pool']}")
    print("=" * 50)

    return all(
        [
            results["concurrent"] > 1.5,
            results["connection_pool"] is True,
        ]
    )


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
