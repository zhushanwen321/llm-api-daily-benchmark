"""TimingTracker 和 TimingCollector 测试."""

import asyncio
import json
import os
import tempfile
import time

import pytest

from benchmark.core.timing_tracker import (
    PhaseTiming,
    TimingCollector,
    TimingTracker,
)


class TestTimingTracker:
    """TimingTracker 单元测试."""

    def test_start_end_phase(self):
        tracker = TimingTracker()

        tracker.start_phase("api_call")
        time.sleep(0.01)
        tracker.end_phase("api_call")

        phase = tracker._phases["api_call"]
        assert phase.duration > 0, "duration should be greater than 0"
        assert phase.end_time > phase.start_time

    def test_multiple_phases(self):
        tracker = TimingTracker()

        tracker.start_phase("phase_a")
        tracker.start_phase("phase_b")

        time.sleep(0.01)
        tracker.end_phase("phase_a")

        time.sleep(0.01)
        tracker.end_phase("phase_b")

        assert "phase_a" in tracker._phases
        assert "phase_b" in tracker._phases
        assert tracker._phases["phase_a"].duration > 0
        assert tracker._phases["phase_b"].duration > 0

    def test_wait_time_tracking(self):
        tracker = TimingTracker()

        tracker.start_phase("api_call")
        tracker.record_wait_start("api_call", "task_1")

        import time

        time.sleep(0.02)

        tracker.record_wait_end("api_call", "task_1")
        tracker.end_phase("api_call")

        phase = tracker._phases["api_call"]
        assert phase.wait_time > 0, "wait_time should be greater than 0"
        assert phase.active_time >= 0, "active_time should be non-negative"
        expected_active = phase.duration - phase.wait_time
        assert abs(phase.active_time - expected_active) < 0.001

    def test_to_gantt_data(self):
        tracker = TimingTracker()

        tracker.start_phase("phase_1")
        tracker.start_phase("phase_2")

        time.sleep(0.01)
        tracker.end_phase("phase_1")
        tracker.end_phase("phase_2")

        gantt_data = tracker.to_gantt_data()

        assert isinstance(gantt_data, list), "gantt_data should be a list"
        assert len(gantt_data) == 2, "should have 2 phases"

        for entry in gantt_data:
            assert "phase" in entry
            assert "start_offset" in entry
            assert "duration" in entry
            assert "wait_time" in entry
            assert "active_time" in entry

    def test_error_handling(self):
        tracker = TimingTracker()

        tracker.end_phase("non_existent")

        assert tracker.get_total_duration() == 0.0
        assert tracker.get_active_duration() == 0.0
        assert tracker.get_wait_duration() == 0.0


class TestTimingCollector:
    def test_collect_and_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TimingCollector(tmpdir, max_queue_size=100)

            tracker = TimingTracker()
            tracker.start_phase("api_call")
            import time

            time.sleep(0.01)
            tracker.end_phase("api_call")

            async def _run():
                collector.start()

                collector.collect(
                    timing=tracker,
                    result_id="result_001",
                    model="test-model",
                    task_id="task_001",
                    run_id="run_001",
                )

                await collector.stop()

            asyncio.run(_run())

            stats = collector.get_stats()
            assert stats["collected"] == 1
            assert stats["written"] == 1
            assert stats["dropped"] == 0

            timing_path = os.path.join(tmpdir, "run_001", "task_001", "timing.jsonl")
            assert os.path.exists(timing_path)

            with open(timing_path, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["result_id"] == "result_001"
            assert record["model"] == "test-model"
            assert record["phase_name"] == "api_call"
            assert record["duration"] > 0

    def test_queue_overflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TimingCollector(tmpdir, max_queue_size=2)

            async def _run():
                collector.start()

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

                await collector.stop()

            asyncio.run(_run())

            stats = collector.get_stats()
            assert stats["dropped"] > 0, (
                "should have dropped records due to queue overflow"
            )
            assert stats["collected"] + stats["dropped"] >= 10
