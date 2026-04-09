"""耗时追踪模块：支持 phase 级别的细粒度耗时统计和异步持久化."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

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
            str, list[tuple[float, str]]
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

    def record_wait_start(self, name: str, task_id: str) -> None:
        """记录等待开始（不阻塞主流程）."""
        now = time.monotonic()
        if name not in self._wait_stacks:
            self._wait_stacks[name] = []
        self._wait_stacks[name].append((now, task_id))

    def record_wait_end(self, name: str, task_id: str) -> None:
        """记录等待结束，累加 wait_time 到对应 phase."""
        if name not in self._wait_stacks or not self._wait_stacks[name]:
            logger.warning(
                "record_wait_end called without matching record_wait_start: %s, %s",
                name,
                task_id,
            )
            return

        now = time.monotonic()
        # 匹配最新的等待记录
        stack = self._wait_stacks[name]
        for i in range(len(stack) - 1, -1, -1):
            start_time, stacked_task_id = stack[i]
            if stacked_task_id == task_id:
                wait_duration = now - start_time
                if name in self._phases:
                    self._phases[name].wait_time += wait_duration
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
    """异步收集器：将 TimingTracker 数据非阻塞写入 SQLite."""

    def __init__(self, db_path: str, max_queue_size: int = 10000) -> None:
        self.db_path = db_path
        self._queue: asyncio.Queue[_TimingRecord | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._writer_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # 统计
        self._collected = 0
        self._dropped = 0
        self._written = 0
        self._write_errors = 0

    async def _init_db(self, db: aiosqlite.Connection) -> None:
        """初始化数据库表结构."""
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS timing_phases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                model TEXT NOT NULL,
                task_id TEXT NOT NULL,
                phase_name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                wait_time REAL NOT NULL,
                active_time REAL NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_timing_result ON timing_phases(result_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_timing_task ON timing_phases(task_id)"
        )
        await db.commit()

    async def _write_loop(self) -> None:
        """写入循环：持续从队列消费并写入数据库."""
        retry_delay = 1.0
        max_retry_delay = 60.0

        while self._running:
            try:
                async with asyncio.timeout(5.0):
                    record = await self._queue.get()

                if record is None:
                    # 收到停止信号，处理完剩余数据
                    break

                try:
                    async with aiosqlite.connect(self.db_path) as db:
                        await self._init_db(db)
                        phases = record.timing._phases
                        for phase in phases.values():
                            await db.execute(
                                """
                                INSERT INTO timing_phases (
                                    result_id, run_id, model, task_id, phase_name,
                                    start_time, end_time, duration, wait_time, active_time,
                                    metadata, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    record.result_id,
                                    record.run_id,
                                    record.model,
                                    record.task_id,
                                    phase.phase_name,
                                    phase.start_time,
                                    phase.end_time,
                                    phase.duration,
                                    phase.wait_time,
                                    phase.active_time,
                                    json.dumps(phase.metadata, ensure_ascii=False),
                                    record.created_at,
                                ),
                            )
                        await db.commit()
                    self._written += 1
                    retry_delay = 1.0  # 成功后重置延迟

                except Exception:
                    self._write_errors += 1
                    logger.exception(
                        "Failed to write timing record: result_id=%s", record.result_id
                    )
                    # 指数退避，最多重试3次
                    if retry_delay <= max_retry_delay / 4:  # 最多重试3次 (1s, 2s, 4s)
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, max_retry_delay)
                        # 重新入队
                        try:
                            self._queue.put_nowait(record)
                        except asyncio.QueueFull:
                            self._dropped += 1
                            logger.error(
                                "Record dropped after write failure (queue full): result_id=%s",
                                record.result_id,
                            )
                    else:
                        # 超过重试次数，记录到错误日志
                        logger.error(
                            "Record dropped after max retries: result_id=%s, retry_delay=%.1fs",
                            record.result_id,
                            retry_delay,
                        )
                        self._dropped += 1

            except asyncio.TimeoutError:
                # 队列等待超时，继续循环检查 _running
                continue
            except Exception:
                logger.exception("Unexpected error in _write_loop")
                await asyncio.sleep(1.0)

        # 处理队列中剩余的记录
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                if record is None:
                    continue
                try:
                    async with aiosqlite.connect(self.db_path) as db:
                        await self._init_db(db)
                        for phase in record.timing._phases.values():
                            await db.execute(
                                """
                                INSERT INTO timing_phases (
                                    result_id, run_id, model, task_id, phase_name,
                                    start_time, end_time, duration, wait_time, active_time,
                                    metadata, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    record.result_id,
                                    record.run_id,
                                    record.model,
                                    record.task_id,
                                    phase.phase_name,
                                    phase.start_time,
                                    phase.end_time,
                                    phase.duration,
                                    phase.wait_time,
                                    phase.active_time,
                                    json.dumps(phase.metadata, ensure_ascii=False),
                                    record.created_at,
                                ),
                            )
                        await db.commit()
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
        """启动异步写入任务."""
        if self._running:
            return
        self._running = True
        self._writer_task = asyncio.create_task(self._write_loop())
        logger.info("TimingCollector started: db=%s", self.db_path)

    async def stop(self) -> None:
        """停止写入任务并等待队列完成."""
        if not self._running:
            return
        self._running = False
        # 发送停止信号
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
        """非阻塞收集数据，队列满时丢弃并计数."""
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
        """返回统计信息."""
        return {
            "collected": self._collected,
            "dropped": self._dropped,
            "written": self._written,
            "write_errors": self._write_errors,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }


# ============================================================================
# 全局单例管理
# ============================================================================

_timing_collector: Optional[TimingCollector] = None
_collector_lock = asyncio.Lock()


def get_timing_collector() -> TimingCollector:
    """获取已初始化的 TimingCollector 实例."""
    if _timing_collector is None:
        raise RuntimeError(
            "TimingCollector not initialized. Call init_timing_collector() first."
        )
    return _timing_collector


def init_timing_collector(db_path: str) -> TimingCollector:
    """同步初始化（不启动写入任务）."""
    global _timing_collector
    collector = TimingCollector(db_path)
    _timing_collector = collector
    return collector


async def start_timing_collection(db_path: str) -> TimingCollector:
    """异步启动收集器."""
    global _timing_collector
    async with _collector_lock:
        if _timing_collector is None:
            _timing_collector = TimingCollector(db_path)
        _timing_collector.start()
        return _timing_collector


async def stop_timing_collection() -> None:
    """异步停止收集器."""
    global _timing_collector
    if _timing_collector is not None:
        await _timing_collector.stop()
