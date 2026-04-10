"""异步评分 Worker。

独立进程运行，轮询 pending_scoring_tasks 表，调用评分后端进行评分。
启动方式: python -m benchmark.workers.scoring_worker [--once]

模式说明:
- 守护模式（默认）: 长期运行，定期轮询新任务
- --once模式: 处理完所有当前待评分任务后退出

锁机制:
- 使用文件锁防止多个worker实例同时运行
- 锁文件路径: /tmp/benchmark_scoring_worker.lock
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.models.database import Database
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.llm_scorer import create_scorer_backend

logger = logging.getLogger(__name__)

LOCK_FILE_PATH = Path("/tmp/benchmark_scoring_worker.lock")


@contextmanager
def worker_lock(timeout: float = 5.0):
    """Worker进程锁，防止多个实例同时运行。

    Args:
        timeout: 获取锁的超时时间（秒）

    Raises:
        RuntimeError: 无法获取锁（其他worker正在运行）
    """
    lock_file = None
    try:
        lock_file = open(LOCK_FILE_PATH, "w")
        # 尝试非阻塞获取锁
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            # 锁被占用，等待一段时间
            logger.info("Another worker is running, waiting for lock...")
            start = time.time()
            while time.time() - start < timeout:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (IOError, OSError):
                    time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"Cannot acquire worker lock at {LOCK_FILE_PATH}. "
                    "Another worker instance is already running."
                )

        # 写入PID便于调试
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        yield lock_file
    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                LOCK_FILE_PATH.unlink(missing_ok=True)
            except Exception:
                pass


class ScoringWorker:
    """异步评分 Worker。

    轮询 pending_scoring_tasks 表，拉取待处理任务，
    调用评分后端进行评分，保存结果到数据库。
    """

    def __init__(self) -> None:
        self.db = Database()
        self.backend = create_scorer_backend()
        self._running = False
        self._current_tasks: list[int] = []

    async def start_once(
        self,
        max_concurrency: int = 3,
        completion_delay: int = 10,
    ) -> None:
        """单次运行模式：处理完所有待评分任务后退出。

        逻辑：
        1. 批量处理所有pending任务（并发控制）
        2. 完成后延迟10秒
        3. 再次检查是否有新pending任务
        4. 如有则继续处理，如无则退出

        Args:
            max_concurrency: 最大并发评分数
            completion_delay: 完成后延迟检查时间（秒）
        """
        logger.info(
            "ScoringWorker (once mode) started | backend=%s | max_concurrency=%d",
            type(self.backend).__name__,
            max_concurrency,
        )

        healthy = await self.backend.health_check()
        if not healthy:
            logger.warning("Backend health check failed, will retry on each task")

        self._running = True
        self._recover_stale_tasks()

        total_processed = 0
        while self._running:
            batch_count = await self._process_all_pending(max_concurrency)
            total_processed += batch_count

            if batch_count == 0:
                # 没有任务了，延迟后再次检查
                logger.info(
                    "No pending tasks, waiting %ds before final check...",
                    completion_delay,
                )
                await asyncio.sleep(completion_delay)

                # 再次检查
                final_count = await self._process_all_pending(max_concurrency)
                total_processed += final_count

                if final_count == 0:
                    logger.info(
                        "ScoringWorker (once mode) finished. Total tasks processed: %d",
                        total_processed,
                    )
                    break

        self._running = False

    async def start_daemon(
        self,
        poll_interval: int = 5,
        batch_size: int = 10,
        max_retries: int = 3,
    ) -> None:
        """守护模式：长期运行，定期轮询新任务。

        Args:
            poll_interval: 轮询间隔（秒）
            batch_size: 每批处理任务数
            max_retries: 单个任务最大重试次数
        """
        logger.info(
            "ScoringWorker (daemon mode) started | backend=%s | poll_interval=%ds | batch_size=%d",
            type(self.backend).__name__,
            poll_interval,
            batch_size,
        )

        healthy = await self.backend.health_check()
        if not healthy:
            logger.warning("Backend health check failed, will retry on each task")

        self._running = True
        self._recover_stale_tasks()

        while self._running:
            try:
                await self._process_batch(batch_size, max_retries)
            except Exception as e:
                logger.error("Worker batch error: %s", e, exc_info=True)

            # 等待下一次轮询
            for _ in range(poll_interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

        logger.info("ScoringWorker (daemon mode) stopped")

    async def _process_all_pending(self, max_concurrency: int) -> int:
        """处理所有待评分任务（并发控制）。

        Args:
            max_concurrency: 最大并发数

        Returns:
            处理的任务数量
        """
        tasks = self.db.fetch_pending_scoring_tasks(limit=1000)

        if not tasks:
            return 0

        logger.info(
            "Processing %d tasks with max_concurrency=%d", len(tasks), max_concurrency
        )

        semaphore = asyncio.Semaphore(max_concurrency)
        processed = 0

        async def process_with_semaphore(task_dict: dict) -> bool:
            async with semaphore:
                if not self._running:
                    return False
                try:
                    await self._process_single_task(task_dict, 3)
                    return True
                except Exception as e:
                    logger.error("Task %d failed: %s", task_dict["id"], e)
                    self.db.fail_scoring_task(task_dict["id"], str(e))
                    return True

        # 并发执行所有任务
        results = await asyncio.gather(*[process_with_semaphore(t) for t in tasks])
        processed = sum(1 for r in results if r)

        logger.info("Batch completed: %d/%d tasks processed", processed, len(tasks))
        return processed

    async def _process_batch(self, batch_size: int, max_retries: int) -> None:
        """处理一批任务（串行，用于守护模式）。"""
        tasks = self.db.fetch_pending_scoring_tasks(limit=batch_size)

        if not tasks:
            return

        logger.info("Processing %d tasks", len(tasks))

        for task_dict in tasks:
            if not self._running:
                logger.info("Stop requested, breaking batch processing")
                break

            task_id = task_dict["id"]
            self._current_tasks.append(task_id)

            try:
                await self._process_single_task(task_dict, max_retries)
            except Exception as e:
                logger.error("Task %d failed: %s", task_id, e, exc_info=True)
                self.db.fail_scoring_task(task_id, str(e))
            finally:
                self._current_tasks.remove(task_id)

    async def _process_single_task(self, task_dict: dict, max_retries: int) -> None:
        """处理单个评分任务。"""
        task_id = task_dict["id"]
        retry_count = task_dict["retry_count"]

        logger.info(
            "Processing task %d | task_id=%s | dimension=%s | retry=%d/%d",
            task_id,
            task_dict["task_id"],
            task_dict["dimension"],
            retry_count,
            max_retries,
        )

        context = self._build_scoring_context(task_dict)
        dimensions = json.loads(task_dict["scoring_dimensions"])

        score_results = await self.backend.score(context, dimensions)

        result_data: dict[str, Any] = {}
        for dim, result in score_results.items():
            result_data[dim] = {
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
                "reasoning": result.reasoning,
            }

        self.db.complete_scoring_task(task_id, result_data)
        self._update_eval_result(task_dict["result_id"], result_data)

        logger.info(
            "Task %d completed | dimensions=%s", task_id, list(score_results.keys())
        )

    def _build_scoring_context(self, task_dict: dict) -> ScoringContext:
        """从数据库行构建 ScoringContext。"""
        test_cases = json.loads(task_dict.get("test_cases", "[]"))
        metadata = json.loads(task_dict.get("metadata", "{}"))

        return ScoringContext(
            model_answer=task_dict["model_answer"],
            raw_output=task_dict["model_output"],
            expected=task_dict["expected_output"],
            task=TaskDefinition(
                task_id=task_dict["task_id"],
                dimension=task_dict["dimension"],
                dataset=task_dict["dataset"],
                prompt=task_dict["prompt"],
                expected_output=task_dict["expected_output"],
                test_cases=test_cases,
                metadata=metadata,
            ),
            reasoning_content=task_dict.get("reasoning_content", ""),
        )

    def _update_eval_result(self, result_id: str, score_data: dict) -> None:
        """更新 eval_results 表的评分数据。"""
        if not score_data:
            return

        conn = self.db._get_conn()

        scores = [
            v["score"]
            for v in score_data.values()
            if isinstance(v.get("score"), (int, float))
        ]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        passed = avg_score >= 60

        composite_scores = {
            k: v["score"]
            for k, v in score_data.items()
            if isinstance(v.get("score"), (int, float))
        }
        report_details = {
            "composite": {"scores": composite_scores},
            "async_scoring": False,
        }
        for k, v in score_data.items():
            report_details[k] = {
                "score": v.get("score", 0),
                "passed": v.get("passed", False),
                "reasoning": v.get("reasoning", ""),
            }

        conn.execute(
            """UPDATE eval_results
               SET functional_score = ?, quality_score = ?, final_score = ?, passed = ?, details = ?
               WHERE result_id = ?
            """,
            (
                avg_score,
                avg_score,
                avg_score,
                int(passed),
                json.dumps(report_details, ensure_ascii=False),
                result_id,
            ),
        )
        conn.commit()

    def _recover_stale_tasks(self) -> None:
        """恢复遗留的 processing 状态任务。"""
        conn = self.db._get_conn()
        cursor = conn.execute(
            """UPDATE pending_scoring_tasks
               SET status = 'pending', processing_started_at = NULL, updated_at = ?
               WHERE status = 'processing'
               AND processing_started_at < datetime('now', '-5 minutes')
            """,
            (datetime.now().isoformat(),),
        )
        recovered = cursor.rowcount
        if recovered > 0:
            conn.commit()
            logger.info("Recovered %d stale tasks from processing state", recovered)

    def stop(self) -> None:
        """请求停止 Worker。"""
        self._running = False
        logger.info("Stop requested, waiting for current tasks to finish...")


async def main() -> None:
    """Worker 入口函数。"""
    parser = argparse.ArgumentParser(description="异步评分 Worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="单次运行模式：处理完所有待评分任务后退出",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("SCORING_WORKER_MAX_CONCURRENCY", "3")),
        help="最大并发评分数（仅--once模式有效，默认3）",
    )
    parser.add_argument(
        "--completion-delay",
        type=int,
        default=int(os.getenv("SCORING_WORKER_COMPLETION_DELAY", "10")),
        help="完成后延迟检查时间（秒，默认10）",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=int(os.getenv("SCORING_WORKER_POLL_INTERVAL", "5")),
        help="守护模式轮询间隔（秒，默认5）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("SCORING_WORKER_BATCH_SIZE", "10")),
        help="守护模式每批处理任务数（默认10）",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="禁用锁文件（不推荐，仅用于调试）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 锁文件机制
    if not args.no_lock:
        try:
            with worker_lock(timeout=5.0):
                worker = ScoringWorker()

                # 注册信号处理
                loop = asyncio.get_event_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, worker.stop)

                try:
                    if args.once:
                        await worker.start_once(
                            max_concurrency=args.max_concurrency,
                            completion_delay=args.completion_delay,
                        )
                    else:
                        await worker.start_daemon(
                            poll_interval=args.poll_interval,
                            batch_size=args.batch_size,
                        )
                except KeyboardInterrupt:
                    worker.stop()
                finally:
                    worker.db.close()
        except RuntimeError as e:
            logger.error("Failed to start worker: %s", e)
            sys.exit(1)
    else:
        # 无锁模式（仅调试）
        worker = ScoringWorker()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, worker.stop)

        try:
            if args.once:
                await worker.start_once(
                    max_concurrency=args.max_concurrency,
                    completion_delay=args.completion_delay,
                )
            else:
                await worker.start_daemon(
                    poll_interval=args.poll_interval,
                    batch_size=args.batch_size,
                )
        except KeyboardInterrupt:
            worker.stop()
        finally:
            worker.db.close()


if __name__ == "__main__":
    asyncio.run(main())
