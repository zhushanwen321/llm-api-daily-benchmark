"""scheduler.py 定时调度器测试。"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestBenchmarkScheduler:
    def test_disabled_by_default(self):
        """SCHEDULER_ENABLED 未设置时调度器不启动。"""
        with patch.dict(os.environ, {}, clear=True):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            assert sched.enabled is False

    def test_enabled_from_env(self):
        """SCHEDULER_ENABLED=true 时 enabled 为 True。"""
        with patch.dict(os.environ, {"SCHEDULER_ENABLED": "true"}):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            assert sched.enabled is True

    def test_models_parsed_from_env(self):
        """SCHEDULER_MODELS 应正确解析为列表。"""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "true",
            "SCHEDULER_MODELS": "glm/glm-4.7,kimi/kimi-2",
            "SCHEDULER_DIMENSIONS": "all",
            "SCHEDULER_CRON": "0 2 * * *",
            "SCHEDULER_SAMPLES": "15",
        }):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            assert sched.models == ["glm/glm-4.7", "kimi/kimi-2"]
            assert sched.dimensions == ["all"]
            assert sched.cron == "0 2 * * *"
            assert sched.samples == 15

    def test_start_does_nothing_when_disabled(self):
        """enabled=False 时 start() 应直接返回。"""
        with patch.dict(os.environ, {}, clear=True):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            sched.start()
            assert sched._scheduler is None

    def test_start_creates_background_scheduler(self):
        """enabled=True 时 start() 应创建 APScheduler 并启动。"""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "true",
            "SCHEDULER_MODELS": "glm/glm-4.7",
            "SCHEDULER_CRON": "0 2 * * *",
        }):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            sched.start()
            assert sched._scheduler is not None
            assert sched._scheduler.running
            sched.stop()

    def test_stop_shuts_down_scheduler(self):
        """stop() 应关闭 APScheduler。"""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "true",
            "SCHEDULER_MODELS": "glm/glm-4.7",
            "SCHEDULER_CRON": "0 2 * * *",
        }):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            sched.start()
            sched.stop()
            assert not sched._scheduler.running
