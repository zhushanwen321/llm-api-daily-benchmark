"""令牌桶限流器。控制每个 provider 的 API 调用频率。"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """令牌桶限流器。

    Args:
        rate: 每秒允许的请求数。桶容量等于 rate。
    """

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError(f"rate_limit must be positive, got {rate}")
        self._rate = rate
        self._tokens = rate
        self._max_tokens = rate
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """获取一个令牌。桶空时阻塞等待并打印 warning。"""
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                return
            wait = (1 - self._tokens) / self._rate

        logger.warning("Rate limited, waiting %.1fs", wait)
        time.sleep(wait)

        with self._lock:
            self._refill()
            self._tokens -= 1

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now
