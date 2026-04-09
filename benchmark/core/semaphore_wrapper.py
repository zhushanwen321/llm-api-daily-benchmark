"""Semaphore wrapper for precise wait time tracking."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, Optional

if TYPE_CHECKING:
    from asyncio import Semaphore
    from typing import Protocol

    class TimingTracker(Protocol):
        def record_wait_start(self, phase_name: str, task_id: str | None) -> None: ...
        def record_wait_end(self, phase_name: str, task_id: str | None) -> None: ...


@asynccontextmanager
async def timed_semaphore(
    semaphore: Semaphore,
    timing_tracker: TimingTracker,
    phase_name: str = "semaphore_wait",
    task_id: Optional[str] = None,
) -> AsyncIterator[None]:
    """
    Async context manager that tracks semaphore wait time with high precision.

    Records the time spent waiting to acquire the semaphore (from the moment
    the coroutine attempts to acquire until it successfully acquires).

    Args:
        semaphore: The asyncio.Semaphore instance to acquire.
        timing_tracker: TimingTracker instance for recording wait durations.
        phase_name: Name identifier for this wait phase (default: "semaphore_wait").
        task_id: Optional task identifier for multi-task tracking.

    Yields:
        None

    Example:
        ```python
        import asyncio
        from benchmark.core.semaphore_wrapper import timed_semaphore

        semaphore = asyncio.Semaphore(2)
        tracker = TimingTracker()

        async def worker():
            async with timed_semaphore(semaphore, tracker, "api_call", task_id=1):
                # Critical section - semaphore was acquired
                await fetch_data()
        ```
    """
    # Record the start of the wait period
    timing_tracker.record_wait_start(phase_name, task_id)

    try:
        # Wait to acquire the semaphore
        await semaphore.acquire()

        # Record the end of the wait (time when lock was acquired)
        timing_tracker.record_wait_end(phase_name, task_id)

        yield
    finally:
        # Always release the semaphore to prevent deadlocks
        semaphore.release()
