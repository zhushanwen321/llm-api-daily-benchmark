"""耗时追踪模块：支持 phase 级别的细粒度耗时统计和异步持久化."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class PhaseTiming:
    """单个 phase 的耗时数据."""

    phase_name: str
    start_time: float
    end_time: float
    duration: float
    wait_time: float = 0.0
    active_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "phase_name": self.phase_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "wait_time": self.wait_time,
            "active_time": self.active_time,
            "metadata": self.metadata,
        }


# ============================================================================
# TimingTracker
# ============================================================================


class TimingTracker:
    """追踪单个执行单元（result/task）的各 phase 耗时."""

    def __init__(self) -> None:
        self._phases: dict[str, PhaseTiming] = {}
        self._wait_stacks: dict[
            str, list[tuple[float, str | None]]
        ] = {}  # phase_name -> [(start_time, task_id), ...]
        self._reference_time: Optional[float] = (
            None  # 首个 phase 的 start_time，用于甘特图
        )

    def start_phase(self, name: str, **metadata: Any) -> None:
        """开始追踪一个 phase."""
        now = time.monotonic()
        if self._reference_time is None:
            self._reference_time = now

        if name not in self._phases:
            self._phases[name] = PhaseTiming(
                phase_name=name,
                start_time=now,
                end_time=now,
                duration=0.0,
                metadata=dict(metadata),
            )
        else:
            # 已存在的 phase，复用但更新 start_time
            self._phases[name].start_time = now
            self._phases[name].metadata.update(metadata)

    def end_phase(self, name: str) -> None:
        """结束追踪一个 phase，计算 duration 和 active_time."""
        if name not in self._phases:
            logger.warning("end_phase called for non-existent phase: %s", name)
            return

        phase = self._phases[name]
        now = time.monotonic()
        phase.end_time = now
        phase.duration = now - phase.start_time
        # active_time = duration - total_wait_time (ensure non-negative)
        phase.active_time = max(0.0, phase.duration - phase.wait_time)

    def record_wait_start(self, phase_name: str, task_id: str | None) -> None:
        """记录等待开始（不阻塞主流程）."""
        now = time.monotonic()
        if phase_name not in self._wait_stacks:
            self._wait_stacks[phase_name] = []
        self._wait_stacks[phase_name].append((now, task_id))

    def record_wait_end(self, phase_name: str, task_id: str | None) -> None:
        """记录等待结束，累加 wait_time 到对应 phase."""
        if phase_name not in self._wait_stacks or not self._wait_stacks[phase_name]:
            logger.warning(
                "record_wait_end called without matching record_wait_start: %s, %s",
                phase_name,
                task_id,
            )
            return

        now = time.monotonic()
        # 匹配最新的等待记录
        stack = self._wait_stacks[phase_name]
        for i in range(len(stack) - 1, -1, -1):
            start_time, stacked_task_id = stack[i]
            if stacked_task_id == task_id:
                wait_duration = now - start_time
                if phase_name in self._phases:
                    self._phases[phase_name].wait_time += wait_duration
                stack.pop(i)
                break

    def add_metadata(self, phase_name: str, key: str, value: Any) -> None:
        """添加元数据到指定 phase."""
        if phase_name not in self._phases:
            logger.warning("add_metadata called for non-existent phase: %s", phase_name)
            return
        self._phases[phase_name].metadata[key] = value

    def get_total_duration(self) -> float:
        """获取总耗时（从首个 phase start 到最后一个 phase end）。"""
        if not self._phases:
            return 0.0
        first_start = min(p.start_time for p in self._phases.values())
        last_end = max(p.end_time for p in self._phases.values())
        return last_end - first_start

    def get_active_duration(self) -> float:
        """获取实际执行时间（不含等待）。"""
        return sum(p.active_time for p in self._phases.values())

    def get_wait_duration(self) -> float:
        """获取总等待时间。"""
        return sum(p.wait_time for p in self._phases.values())

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "phases": {name: phase.to_dict() for name, phase in self._phases.items()},
            "total_duration": self.get_total_duration(),
            "active_duration": self.get_active_duration(),
            "wait_duration": self.get_wait_duration(),
        }

    def to_gantt_data(self) -> list[dict[str, Any]]:
        """生成甘特图数据（基于绝对时间的偏移）。"""
        if not self._reference_time:
            return []
        return [
            {
                "phase": name,
                "start_offset": phase.start_time - self._reference_time,
                "duration": phase.duration,
                "wait_time": phase.wait_time,
                "active_time": phase.active_time,
                "metadata": phase.metadata,
            }
            for name, phase in self._phases.items()
        ]


# ============================================================================
# TimingCollector
# ============================================================================


@dataclass
class _TimingRecord:
    """写入队列中的单条记录."""

    result_id: str
    run_id: str
    model: str
    task_id: str
    timing: TimingTracker
    created_at: str


class TimingCollector:
    """异步收集器：将 TimingTracker 数据非阻塞写入 timing.jsonl."""

    def __init__(self, data_dir: str, max_queue_size: int = 10000) -> None:
        self.data_dir = data_dir
        self._queue: asyncio.Queue[_TimingRecord | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._writer_task: Optional[asyncio.Task[None]] = None
        self._running = False

        self._collected = 0
        self._dropped = 0
        self._written = 0
        self._write_errors = 0

    def _record_to_jsonl_lines(self, record: _TimingRecord) -> list[str]:
        lines: list[str] = []
        for phase in record.timing._phases.values():
            obj = {
                "result_id": record.result_id,
                "run_id": record.run_id,
                "model": record.model,
                "task_id": record.task_id,
                "phase_name": phase.phase_name,
                "start_time": phase.start_time,
                "end_time": phase.end_time,
                "duration": phase.duration,
                "wait_time": phase.wait_time,
                "active_time": phase.active_time,
                "metadata": phase.metadata,
                "created_at": record.created_at,
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        return lines

    def _write_record_sync(self, record: _TimingRecord) -> None:
        lines = self._record_to_jsonl_lines(record)
        if not lines:
            return
        path = Path(self.data_dir) / record.run_id / record.task_id / "timing.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    async def _write_loop(self) -> None:
        retry_delay = 1.0
        max_retry_delay = 60.0

        while self._running:
            try:
                async with asyncio.timeout(5.0):
                    record = await self._queue.get()

                if record is None:
                    break

                try:
                    self._write_record_sync(record)
                    self._written += 1
                    retry_delay = 1.0

                except Exception:
                    self._write_errors += 1
                    logger.exception(
                        "Failed to write timing record: result_id=%s",
                        record.result_id,
                    )
                    if retry_delay <= max_retry_delay / 4:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, max_retry_delay)
                        try:
                            self._queue.put_nowait(record)
                        except asyncio.QueueFull:
                            self._dropped += 1
                            logger.error(
                                "Record dropped after write failure (queue full): result_id=%s",
                                record.result_id,
                            )
                    else:
                        logger.error(
                            "Record dropped after max retries: result_id=%s, retry_delay=%.1fs",
                            record.result_id,
                            retry_delay,
                        )
                        self._dropped += 1

            except asyncio.TimeoutError:
                continue
            except Exception:
                logger.exception("Unexpected error in _write_loop")
                await asyncio.sleep(1.0)

        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                if record is None:
                    continue
                try:
                    self._write_record_sync(record)
                    self._written += 1
                except Exception:
                    self._write_errors += 1
                    logger.exception(
                        "Failed to flush remaining record: result_id=%s",
                        record.result_id,
                    )
            except asyncio.QueueEmpty:
                break

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._writer_task = asyncio.create_task(self._write_loop())
        logger.info("TimingCollector started: data_dir=%s", self.data_dir)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._writer_task:
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "TimingCollector stopped: collected=%d, dropped=%d, written=%d, errors=%d",
            self._collected,
            self._dropped,
            self._written,
            self._write_errors,
        )

    def collect(
        self,
        timing: TimingTracker,
        result_id: str,
        model: str,
        task_id: str,
        run_id: str = "default",
    ) -> None:
        try:
            record = _TimingRecord(
                result_id=result_id,
                run_id=run_id,
                model=model,
                task_id=task_id,
                timing=timing,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._queue.put_nowait(record)
            self._collected += 1
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning(
                "TimingCollector queue full, record dropped: result_id=%s", result_id
            )

    def get_stats(self) -> dict[str, Any]:
        return {
            "collected": self._collected,
            "dropped": self._dropped,
            "written": self._written,
            "write_errors": self._write_errors,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }


_timing_collector: Optional[TimingCollector] = None
_collector_lock = asyncio.Lock()


def get_timing_collector() -> TimingCollector:
    if _timing_collector is None:
        raise RuntimeError(
            "TimingCollector not initialized. Call init_timing_collector() first."
        )
    return _timing_collector


def init_timing_collector(data_dir: str) -> TimingCollector:
    global _timing_collector
    collector = TimingCollector(data_dir)
    _timing_collector = collector
    return collector


async def start_timing_collection(data_dir: str) -> TimingCollector:
    global _timing_collector
    async with _collector_lock:
        if _timing_collector is None:
            _timing_collector = TimingCollector(data_dir)
        _timing_collector.start()
        return _timing_collector


async def stop_timing_collection() -> None:
    global _timing_collector
    if _timing_collector is not None:
        await _timing_collector.stop()
