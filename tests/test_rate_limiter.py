"""令牌桶限流器测试。"""

import time
from unittest.mock import patch

from benchmark.core.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def test_acquire_within_rate_no_wait(self):
        """rate=10 时，连续 acquire 10 次不应等待。"""
        limiter = TokenBucketRateLimiter(rate=10)
        start = time.monotonic()
        for _ in range(10):
            limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_acquire_exceeds_rate_waits(self):
        """rate=10 时，第 11 次 acquire 应等待约 0.1s。"""
        limiter = TokenBucketRateLimiter(rate=10)
        for _ in range(10):
            limiter.acquire()
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08  # 允许小幅误差

    def test_rate_limit_one_slow(self):
        """rate=1 时，连续 3 次调用至少耗时约 2s。"""
        limiter = TokenBucketRateLimiter(rate=1)
        start = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 1.8

    def test_warning_on_wait(self):
        """等待时应打印 warning。"""
        limiter = TokenBucketRateLimiter(rate=2)
        limiter.acquire()
        limiter.acquire()
        with patch("benchmark.core.rate_limiter.logger.warning") as mock_warn:
            limiter.acquire()
            mock_warn.assert_called_once()
            assert "Rate limited" in mock_warn.call_args[0][0]
