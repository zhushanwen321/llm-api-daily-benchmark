"""TimingTracker 和 TimingCollector 测试."""

import asyncio
import tempfile

import pytest

from benchmark.core.timing_tracker import (
    PhaseTiming,
    TimingCollector,
    TimingTracker,
)


class TestTimingTracker:
    """TimingTracker 单元测试."""

    def test_start_end_phase(self):
        """测试开始/结束单个 phase，验证 duration > 0."""
        tracker = TimingTracker()

        tracker.start_phase("api_call")
        # 模拟一些耗时操作
        import time

        time.sleep(0.01)
        tracker.end_phase("api_call")

        phase = tracker._phases["api_call"]
        assert phase.duration > 0, "duration should be greater than 0"
        assert phase.end_time > phase.start_time

    def test_multiple_phases(self):
        """测试多个 phase 同时追踪."""
        tracker = TimingTracker()

        tracker.start_phase("phase_a")
        tracker.start_phase("phase_b")

        import time

        time.sleep(0.01)
        tracker.end_phase("phase_a")

        time.sleep(0.01)
        tracker.end_phase("phase_b")

        assert "phase_a" in tracker._phases
        assert "phase_b" in tracker._phases
        assert tracker._phases["phase_a"].duration > 0
        assert tracker._phases["phase_b"].duration > 0

    def test_wait_time_tracking(self):
        """测试 wait_time 记录，验证 active_time = duration - wait_time."""
        tracker = TimingTracker()

        tracker.start_phase("api_call")
        tracker.record_wait_start("api_call", "task_1")

        import time

        time.sleep(0.02)  # 模拟等待

        tracker.record_wait_end("api_call", "task_1")
        tracker.end_phase("api_call")

        phase = tracker._phases["api_call"]
        assert phase.wait_time > 0, "wait_time should be greater than 0"
        assert phase.active_time >= 0, "active_time should be non-negative"
        # active_time = duration - wait_time (允许浮点误差)
        expected_active = phase.duration - phase.wait_time
        assert abs(phase.active_time - expected_active) < 0.001

    def test_to_gantt_data(self):
        """测试甘特图数据生成，验证 phases 列表和 total 字段."""
        tracker = TimingTracker()

        tracker.start_phase("phase_1")
        tracker.start_phase("phase_2")

        import time

        time.sleep(0.01)
        tracker.end_phase("phase_1")
        tracker.end_phase("phase_2")

        gantt_data = tracker.to_gantt_data()

        assert isinstance(gantt_data, list), "gantt_data should be a list"
        assert len(gantt_data) == 2, "should have 2 phases"

        # 验证每个 gantt 条目包含必要字段
        for entry in gantt_data:
            assert "phase" in entry
            assert "start_offset" in entry
            assert "duration" in entry
            assert "wait_time" in entry
            assert "active_time" in entry

    def test_error_handling(self):
        """测试结束未开始的 phase，应返回 0.0 不报错."""
        tracker = TimingTracker()

        # 结束一个不存在的 phase 不应抛出异常
        tracker.end_phase("non_existent")

        # get_total_duration 在没有 phases 时应返回 0.0
        assert tracker.get_total_duration() == 0.0
        assert tracker.get_active_duration() == 0.0
        assert tracker.get_wait_duration() == 0.0


class TestTimingCollector:
    """TimingCollector 单元测试."""

    def test_collect_and_write(self):
        """测试收集和写入数据库，验证数据正确保存."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_timing.db")
            collector = TimingCollector(db_path, max_queue_size=100)

            # 创建 TimingTracker 并记录数据
            tracker = TimingTracker()
            tracker.start_phase("api_call")
            import time

            time.sleep(0.01)
            tracker.end_phase("api_call")

            # 启动收集器
            collector.start()

            # 收集数据
            collector.collect(
                timing=tracker,
                result_id="result_001",
                model="test-model",
                task_id="task_001",
                run_id="run_001",
            )

            # 停止收集器，触发写入
            asyncio.run(collector.stop())

            # 验证统计
            stats = collector.get_stats()
            assert stats["collected"] == 1
            assert stats["written"] == 1
            assert stats["dropped"] == 0

            # 验证数据库中的数据
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT result_id, model, phase_name, duration FROM timing_phases"
            )
            rows = cursor.fetchall()
            conn.close()

            assert len(rows) == 1
            assert rows[0][0] == "result_001"
            assert rows[0][1] == "test-model"
            assert rows[0][2] == "api_call"
            assert rows[0][3] > 0

    def test_queue_overflow(self):
        """测试队列溢出处理，验证 dropped_count > 0."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_overflow.db")
            # 设置极小的队列大小
            collector = TimingCollector(db_path, max_queue_size=2)
            collector.start()

            # 快速收集大量数据，触发溢出
            for i in range(10):
                tracker = TimingTracker()
                tracker.start_phase("phase_1")
                import time

                time.sleep(0.001)
                tracker.end_phase("phase_1")

                collector.collect(
                    timing=tracker,
                    result_id=f"result_{i}",
                    model="test-model",
                    task_id=f"task_{i}",
                    run_id="run_001",
                )

            # 停止收集器
            asyncio.run(collector.stop())

            # 验证有数据被丢弃
            stats = collector.get_stats()
            assert stats["dropped"] > 0, (
                "should have dropped records due to queue overflow"
            )
            # 收集数应该等于成功入队数（不含丢弃）
            assert stats["collected"] == 10
