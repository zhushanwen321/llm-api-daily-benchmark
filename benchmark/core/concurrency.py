"""异步并发控制器。按 provider 维度控制同时进行的 API 请求数。"""
from __future__ import annotations

import asyncio


class AsyncConcurrencyLimiter:
    """provider 级并发流控制器。

    使用 asyncio.Semaphore 限制同一 provider 的并发请求数，
    通过 get_or_create 工厂方法实现 provider 维度的单例管理。
    """

    _instances: dict[str, AsyncConcurrencyLimiter] = {}

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self._sem = asyncio.Semaphore(max_concurrency)

    @classmethod
    def get_or_create(cls, provider: str, max_concurrency: int) -> AsyncConcurrencyLimiter:
        """获取或创建指定 provider 的并发控制器。

        已存在的实例会忽略 max_concurrency 参数，保证运行时行为一致。
        """
        if provider not in cls._instances:
            cls._instances[provider] = cls(max_concurrency)
        return cls._instances[provider]

    async def acquire(self) -> None:
        await self._sem.acquire()

    def release(self) -> None:
        self._sem.release()
