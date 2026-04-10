"""异步评分 Worker。

独立进程运行，轮询 pending_scoring_tasks 表，调用评分后端进行评分。
启动方式: python -m benchmark.workers.scoring_worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import os
from datetime import datetime
from typing import Any

from benchmark.models.database import Database
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.llm_scorer import create_scorer_backend

logger = logging.getLogger(__name__)


class ScoringWorker:
    """异步评分 Worker。

    轮询 pending_scoring_tasks 表，拉取待处理任务，
    调用评分后端进行评分，保存结果到数据库。
    """

    def __init__(self) -> None:
        self.db = Database()
        self.backend = create_scorer_backend()
        self._running = False
        self._current_tasks: list[int] = []  # 正在处理的任务ID

    async def start(self) -> None:
        """启动 Worker 主循环。"""
        poll_interval = int(os.getenv("SCORING_WORKER_POLL_INTERVAL", "5"))
        batch_size = int(os.getenv("SCORING_WORKER_BATCH_SIZE", "10"))
        max_retries = int(os.getenv("SCORING_WORKER_MAX_RETRIES", "3"))

        logger.info(
            "ScoringWorker started | backend=%s | poll_interval=%ds | batch_size=%d | max_retries=%d",
            type(self.backend).__name__,
            poll_interval,
            batch_size,
            max_retries,
        )

        # 健康检查
        healthy = await self.backend.health_check()
        if not healthy:
            logger.warning(
                "Backend health check failed, worker will still start and retry on each task"
            )

        self._running = True

        # 恢复遗留的 processing 状态任务（Worker崩溃后可能残留）
        self._recover_stale_tasks()

        while self._running:
            try:
                await self._process_batch(batch_size, max_retries)
            except Exception as e:
                logger.error("Worker batch error: %s", e, exc_info=True)

            # 等待下一次轮询，但响应停止信号
            for _ in range(poll_interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

        logger.info("ScoringWorker stopped")

    def stop(self) -> None:
        """请求停止 Worker（优雅关闭）。"""
        self._running = False
        logger.info("Stop requested, waiting for current tasks to finish...")

    def _recover_stale_tasks(self) -> None:
        """恢复遗留的 processing 状态任务。

        如果 Worker 崩溃，可能有任务卡在 processing 状态。
        将超过5分钟的 processing 任务重置为 pending。
        """
        conn = self.db._get_conn()
        # 使用 SQLite 的 datetime 计算
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

    async def _process_batch(self, batch_size: int, max_retries: int) -> None:
        """处理一批任务。"""
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

        # 构建评分上下文
        context = self._build_scoring_context(task_dict)
        dimensions = json.loads(task_dict["scoring_dimensions"])

        # 调用评分后端
        score_results = await self.backend.score(context, dimensions)

        # 转换为可序列化格式
        result_data: dict[str, Any] = {}
        for dim, result in score_results.items():
            result_data[dim] = {
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
                "reasoning": result.reasoning,
            }

        # 保存结果
        self.db.complete_scoring_task(task_id, result_data)

        # 同时更新 eval_results 表的分数
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
        """更新 eval_results 表的评分数据。

        计算加权总分，更新 functional_score, quality_score, final_score, passed, details。
        """
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

        # 将 score_data 包装成 composite.scores 格式，兼容报告渲染器
        # 报告的 _build_dimension_score_table 和 _extract_dimension_scores
        # 期望 details.composite.scores 结构
        composite_scores = {
            k: v["score"]
            for k, v in score_data.items()
            if isinstance(v.get("score"), (int, float))
        }
        report_details = {
            "composite": {"scores": composite_scores},
            "async_scoring": False,
        }
        # 保留原始评分数据的详细信息
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


async def main() -> None:
    """Worker 入口函数。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    worker = ScoringWorker()

    # 注册信号处理
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)

    try:
        await worker.start()
    except KeyboardInterrupt:
        worker.stop()
    finally:
        worker.db.close()


if __name__ == "__main__":
    asyncio.run(main())
