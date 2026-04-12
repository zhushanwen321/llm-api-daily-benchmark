"""异步并发控制器。按 provider 维度控制同时进行的 API 请求数。"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class AsyncConcurrencyLimiter:
    """provider 级并发流控制器。

    使用 asyncio.Semaphore 限制同一 provider 的并发请求数，
    通过 get_or_create 工厂方法实现 provider 维度的单例管理。

    429 全局退避：当一个 task 收到 429 后，设置 provider 级别的退避期，
    期间所有 task 的 acquire 都会等待，避免 429 风暴。
    """

    _instances: dict[str, AsyncConcurrencyLimiter] = {}

    def __init__(self, provider: str, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self._provider = provider
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max = max_concurrency
        self._rate_limited_until: float = 0.0
        logger.debug(
            f"[LIMITER] 创建 provider={provider} max_concurrency={max_concurrency}"
        )

    @classmethod
    def get_or_create(
        cls, provider: str, max_concurrency: int
    ) -> AsyncConcurrencyLimiter:
        """获取或创建指定 provider 的并发控制器。

        已存在的实例会忽略 max_concurrency 参数，保证运行时行为一致。
        """
        if provider not in cls._instances:
            cls._instances[provider] = cls(provider, max_concurrency)
        return cls._instances[provider]

    async def acquire(self) -> None:
        """获取信号量，同时在 429 全局退避期内阻塞。"""
        available = self._sem._value
        logger.debug(
            f"[LIMITER] provider={self._provider} 请求信号量 | 可用={available}/{self._max}"
        )
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            wait = self._rate_limited_until - now
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            await self._sem.acquire()
            # acquire 成功后再检查一次：等待 semaphore 期间可能被其他 task 设置了退避
            now = loop.time()
            wait = self._rate_limited_until - now
            if wait > 0:
                self._sem.release()
                await asyncio.sleep(wait)
                continue
            logger.debug(f"[LIMITER] provider={self._provider} 信号量获取成功")
            return

    def release(self) -> None:
        self._sem.release()
        available = self._sem._value
        logger.debug(
            f"[LIMITER] provider={self._provider} 信号量释放 | 可用={available}/{self._max}"
        )

    def set_rate_limited(self, until: float) -> None:
        """设置全局退避截止时间（monotonic clock）。

        取当前值和新值的较大者，确保退避期只延长不缩短。
        """
        self._rate_limited_until = max(self._rate_limited_until, until)
        wait = self._rate_limited_until - asyncio.get_running_loop().time()
        logger.debug(
            f"[LIMITER] provider={self._provider} 设置429退避至 {wait:.1f}s 后"
        )
