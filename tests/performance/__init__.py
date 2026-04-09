"""Performance benchmark tests.

This module contains tests for measuring and verifying performance characteristics:
- Concurrent execution speedup
- Cache hit rates and speedup
- HTTP client connection pool reuse
"""

from tests.performance.test_benchmark_performance import (
    TestConcurrency,
    TestCaching,
    TestConnectionPool,
)

__all__ = [
    "TestConcurrency",
    "TestCaching",
    "TestConnectionPool",
]
