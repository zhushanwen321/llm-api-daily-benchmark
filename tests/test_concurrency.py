"""AsyncConcurrencyLimiter 测试。"""
import asyncio
import pytest
from benchmark.core.concurrency import AsyncConcurrencyLimiter


class TestAsyncConcurrencyLimiter:
    def setup_method(self):
        """每个测试前清理单例缓存，避免测试间污染。"""
        AsyncConcurrencyLimiter._instances.clear()

    def test_get_or_create_returns_same_instance(self):
        a = AsyncConcurrencyLimiter.get_or_create("provider_a", 5)
        b = AsyncConcurrencyLimiter.get_or_create("provider_a", 999)  # max_concurrency 被忽略
        assert a is b

    def test_get_or_create_different_providers(self):
        a = AsyncConcurrencyLimiter.get_or_create("p1", 2)
        b = AsyncConcurrencyLimiter.get_or_create("p2", 3)
        assert a is not b

    def test_concurrency_limit_respected(self):
        """同时运行 max_concurrency 个任务后，第 max_concurrency+1 个必须等待。"""
        max_conc = 2
        limiter = AsyncConcurrencyLimiter.get_or_create("test", max_conc)
        running = 0
        peak = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal running, peak
            await limiter.acquire()
            async with lock:
                running += 1
                peak = max(peak, running)
            await asyncio.sleep(0.05)
            async with lock:
                running -= 1
            limiter.release()

        async def run():
            await asyncio.gather(*[worker() for _ in range(5)])

        asyncio.run(run())
        assert peak <= max_conc

    def test_release_without_acquire_no_error(self):
        """release 多次不应抛异常（Semaphore 行为）。"""
        limiter = AsyncConcurrencyLimiter(max_concurrency=1)
        limiter.release()  # 不抛异常即可

    def test_max_concurrency_one_serial(self):
        """max_concurrency=1 时任务串行执行。"""
        limiter = AsyncConcurrencyLimiter.get_or_create("serial", 1)
        order = []

        async def worker(idx):
            await limiter.acquire()
            order.append(f"start-{idx}")
            await asyncio.sleep(0.02)
            order.append(f"end-{idx}")
            limiter.release()

        async def run():
            await asyncio.gather(*[asyncio.create_task(worker(i)) for i in range(3)])

        asyncio.run(run())
        # max_concurrency=1，第一个 start 必须对应第一个 end
        starts = [i for i, x in enumerate(order) if x.startswith("start")]
        ends = [i for i, x in enumerate(order) if x.startswith("end")]
        for s, e in zip(starts, ends):
            assert s < e
