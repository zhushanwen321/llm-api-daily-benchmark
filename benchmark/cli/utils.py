"""CLI 辅助工具函数。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup_proxy() -> None:
    """从 .env 加载代理配置，用于 HuggingFace 数据集下载."""
    load_dotenv()

    dataset_flag = Path("benchmark/datasets/.download-complete")
    if dataset_flag.exists():
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        logger.info("检测到 .download-complete 标志，启用离线模式")
    else:
        logger.debug("未检测到 .download-complete 标志，使用网络下载数据集")

    proxy = os.getenv("HF_PROXY")
    if proxy:
        os.environ.setdefault("http_proxy", proxy)
        os.environ.setdefault("https_proxy", proxy)
        os.environ.setdefault("all_proxy", proxy)


def get_provider_concurrency(model: str) -> int:
    """获取 provider 的最大并发数。"""
    try:
        from benchmark.config import get_model_config

        cfg = get_model_config(model)
        return cfg.get("max_concurrency", cfg.get("rate_limit", 2))
    except Exception:
        return 2
