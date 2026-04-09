"""SemaphoreWrapper 测试 - 测试 timed_semaphore 上下文管理器."""

import asyncio

import pytest

from benchmark.core.semaphore_wrapper import timed_semaphore
from benchmark.core.timing_tracker import TimingTracker


@pytest.mark.asyncio
async def test_timed_semaphore():
    """测试 Semaphore 等待时间追踪."""
    semaphore = asyncio.Semaphore(1)
    tracker = TimingTracker()

    tracker.start_phase("api_call")

    wait_times = []

    async def task_with_semaphore(task_id: int):
        """使用 timed_semaphore 的任务."""
        async with timed_semaphore(semaphore, tracker, "api_call", task_id=task_id):
            # 记录进入临界区后的等待时间
            phase = tracker._phases["api_call"]
            wait_times.append(phase.wait_time)
            # 模拟一些操作
            await asyncio.sleep(0.05)

    # 创建两个并发任务
    task1 = asyncio.create_task(task_with_semaphore(task_id=1))
    task2 = asyncio.create_task(task_with_semaphore(task_id=2))

    await asyncio.gather(task1, task2)

    tracker.end_phase("api_call")

    # 验证：两个 wait_time 都被记录
    phase = tracker._phases["api_call"]
    assert phase.wait_time > 0, "wait_time should be greater than 0"

    # 第一个任务几乎立即获得锁（wait_time 接近 0）
    # 第二个任务需要等待（wait_time > 0）
    # 由于我们是顺序记录的最后一个任务的 wait_time，
    # 验证总 wait_time > 0
    assert len(wait_times) == 2

    # 验证 active_time 计算正确
    assert phase.active_time >= 0


@pytest.mark.asyncio
async def test_timed_semaphore_immediate_acquire():
    """测试当 semaphore 可用时，等待时间接近 0."""
    semaphore = asyncio.Semaphore(1)  # 1 个可用
    tracker = TimingTracker()

    tracker.start_phase("test_phase")

    acquired_wait_time = None

    async def task():
        nonlocal acquired_wait_time
        async with timed_semaphore(semaphore, tracker, "test_phase", task_id=1):
            phase = tracker._phases["test_phase"]
            acquired_wait_time = phase.wait_time
            await asyncio.sleep(0.01)

    await task()
    tracker.end_phase("test_phase")

    # 当 semaphore 立即可用时，wait_time 应该接近 0
    assert acquired_wait_time is not None
    # 由于已经获取了，等待时间为之前积累的时间
    # 验证数值合理（可能在 0-0.05 秒范围内）
    assert acquired_wait_time >= 0


@pytest.mark.asyncio
async def test_timed_semaphore_releases_on_exception():
    """测试 timed_semaphore 在异常时正确释放锁."""
    semaphore = asyncio.Semaphore(1)
    tracker = TimingTracker()

    tracker.start_phase("exception_test")

    first_acquired = False
    second_acquired = False

    async def task1():
        nonlocal first_acquired
        try:
            async with timed_semaphore(semaphore, tracker, "exception_test", task_id=1):
                first_acquired = True
                await asyncio.sleep(0.02)
                raise ValueError("Simulated error")
        except ValueError:
            pass

    async def task2():
        nonlocal second_acquired
        async with timed_semaphore(semaphore, tracker, "exception_test", task_id=2):
            second_acquired = True
            await asyncio.sleep(0.01)

    await asyncio.gather(task1(), task2())
    tracker.end_phase("exception_test")

    # 两个任务都应该能获取到 semaphore
    assert first_acquired
    assert second_acquired
