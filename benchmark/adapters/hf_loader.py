"""HuggingFace 数据集加载器。

通过 HuggingFace datasets-server API 下载数据，替代 datasets.load_dataset()。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HF_SERVER = "https://datasets-server.huggingface.co"
_PAGE_SIZE = 100
_TIMEOUT = 30
_MAX_RETRIES = 3


def _cache_path(cache_dir: str, repo: str, config: str | None, split: str) -> str:
    """生成缓存文件路径: {cache_dir}/{safe_repo}/{config}/{split}.json"""
    safe_repo = repo.replace("/", "--")
    parts = [cache_dir, safe_repo]
    if config:
        parts.append(config.replace("/", "--"))
    parts.append(f"{split}.json")
    return os.path.join(*parts)


def _fetch_all_rows(repo: str, config: str | None, split: str) -> list[dict[str, Any]]:
    """分页获取数据集的所有行。"""
    all_rows: list[dict[str, Any]] = []
    offset = 0
    params: dict[str, str] = {"dataset": repo, "split": split, "length": str(_PAGE_SIZE)}
    if config:
        params["config"] = config

    while True:
        params["offset"] = str(offset)
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(f"{_HF_SERVER}/rows", params=params, timeout=_TIMEOUT)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    logger.warning("HF API request failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
        else:
            raise RuntimeError(f"Failed to fetch {repo}/{config}/{split} after {_MAX_RETRIES} retries: {last_error}")

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"HF datasets-server error: {data['error']}")

        rows = data.get("rows", [])
        for item in rows:
            all_rows.append(item["row"])

        num_total = data.get("num_rows_total", 0)
        offset += len(rows)
        if offset >= num_total or not rows:
            break

    logger.info("Loaded %d rows from %s (config=%s, split=%s)", len(all_rows), repo, config, split)
    return all_rows


def load_hf_dataset(
    repo: str,
    split: str,
    cache_dir: str,
    *,
    config_name: str | None = None,
) -> list[dict]:
    """从 HuggingFace 下载数据集，带本地 JSON 缓存。

    Args:
        repo: HuggingFace 数据集仓库 ID，如 "openai/gsm8k"。
        split: 数据集分片名，如 "test"。
        cache_dir: 本地缓存目录。
        config_name: 数据集配置名（可选）。
    """
    cache_file = _cache_path(cache_dir, repo, config_name, split)
    if os.path.exists(cache_file):
        logger.debug("Reading cache: %s", cache_file)
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Cache miss for %s (config=%s, split=%s), fetching...", repo, config_name, split)
    rows = _fetch_all_rows(repo, config_name, split)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logger.info("Cached %d rows to %s", len(rows), cache_file)
    return rows
