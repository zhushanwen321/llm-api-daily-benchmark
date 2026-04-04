"""定时调度器：基于 APScheduler 实现定时评测。"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BenchmarkScheduler:
    """定时评测调度器。

    从环境变量读取配置，使用 APScheduler 按 cron 表达式定时触发评测。
    调度器在后台线程中运行，主进程（如 Streamlit）不受阻塞。
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
        self.cron = os.getenv("SCHEDULER_CRON", "0 2 * * *")
        self.models = [
            m.strip()
            for m in os.getenv("SCHEDULER_MODELS", "").split(",")
            if m.strip()
        ]
        dimensions_raw = os.getenv("SCHEDULER_DIMENSIONS", "all")
        self.dimensions = [
            d.strip()
            for d in dimensions_raw.split(",")
            if d.strip()
        ]
        self.samples = int(os.getenv("SCHEDULER_SAMPLES", "15"))
        self._scheduler: BackgroundScheduler | None = None

    def start(self) -> None:
        """启动调度器。如果未启用则跳过。"""
        if not self.enabled:
            logger.info("调度器未启用 (SCHEDULER_ENABLED != true)")
            return

        if not self.models:
            logger.warning("调度器启用但未配置 SCHEDULER_MODELS，跳过启动")
            return

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._run_scheduled_evaluation,
            CronTrigger.from_crontab(self.cron),
            id="daily_evaluation",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            f"调度器已启动: cron='{self.cron}', "
            f"models={self.models}, dimensions={self.dimensions}, "
            f"samples={self.samples}"
        )

    def stop(self) -> None:
        """停止调度器。"""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("调度器已停止")

    def _run_scheduled_evaluation(self) -> None:
        """调度触发时执行的全量评测。"""
        logger.info("定时评测触发: models=%s, dimensions=%s", self.models, self.dimensions)
        from benchmark.cli import _run_multi_evaluation

        dimensions = self.dimensions
        if dimensions == ["all"]:
            from benchmark.cli import DIMENSION_REGISTRY
            dimensions = list(DIMENSION_REGISTRY.keys())

        asyncio.run(_run_multi_evaluation(self.models, dimensions, self.samples, debug=False))
        logger.info("定时评测完成")
