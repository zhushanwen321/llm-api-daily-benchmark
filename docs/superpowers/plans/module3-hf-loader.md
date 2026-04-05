# 模块 3：数据集加载替换（Task 7-8）

### Task 7: 创建 hf_loader.py + 测试

**Files:**
- Create: `benchmark/adapters/hf_loader.py`
- Create: `tests/test_hf_loader.py`

---

- [ ] **Step 1: 编写 hf_loader 测试**

```python
# tests/test_hf_loader.py
"""hf_loader 单元测试。"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import requests

from benchmark.adapters.hf_loader import _cache_path, _fetch_all_rows, load_hf_dataset


def _make_rows_response(rows: list[dict], num_total: int, offset: int = 0) -> dict:
    wrapped = [{"row_idx": offset + i, "row": r, "truncated_cells": []} for i, r in enumerate(rows)]
    return {
        "features": [],
        "rows": wrapped,
        "num_rows_total": num_total,
        "num_rows_per_page": 100,
        "partial": False,
    }


class TestCachePath:
    def test_basic(self):
        path = _cache_path("/tmp/cache", "openai/gsm8k", None, "test")
        assert path == "/tmp/cache/openai--gsm8k/test.json"

    def test_with_config(self):
        path = _cache_path("/tmp/cache", "openai/gsm8k", "main", "test")
        assert path == "/tmp/cache/openai--gsm8k/main/test.json"


class TestFetchAllRows:
    def test_single_page(self, requests_mock):
        rows = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
        resp = _make_rows_response(rows, num_total=2)
        requests_mock.get(
            "https://datasets-server.huggingface.co/rows",
            json=resp,
        )
        result = _fetch_all_rows("test/repo", None, "test")
        assert len(result) == 2

    def test_multi_page(self, requests_mock):
        page1_rows = [{"id": i} for i in range(100)]
        page2_rows = [{"id": i} for i in range(100, 150)]
        resp1 = _make_rows_response(page1_rows, num_total=150, offset=0)
        resp2 = _make_rows_response(page2_rows, num_total=150, offset=100)
        requests_mock.get(
            "https://datasets-server.huggingface.co/rows",
            [{"json": resp1}, {"json": resp2}],
        )
        result = _fetch_all_rows("test/repo", None, "test")
        assert len(result) == 150

    def test_with_config_param(self, requests_mock):
        resp = _make_rows_response([], num_total=0)
        m = requests_mock.get("https://datasets-server.huggingface.co/rows", json=resp)
        _fetch_all_rows("test/repo", "my_config", "test")
        assert m.last_request.qs["config"] == ["my_config"]

    def test_retry_on_network_error(self, requests_mock):
        resp = _make_rows_response([{"id": 1}], num_total=1)
        requests_mock.get(
            "https://datasets-server.huggingface.co/rows",
            [{"exc": requests.ConnectionError("timeout")}, {"json": resp}],
        )
        result = _fetch_all_rows("test/repo", None, "test")
        assert len(result) == 1


class TestLoadHfDataset:
    def test_cache_miss_then_write(self, requests_mock):
        rows = [{"q": "what?", "a": "42"}]
        resp = _make_rows_response(rows, num_total=1)
        requests_mock.get("https://datasets-server.huggingface.co/rows", json=resp)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_hf_dataset("test/repo", "test", tmpdir)
            assert len(result) == 1
            cache_file = _cache_path(tmpdir, "test/repo", None, "test")
            assert os.path.exists(cache_file)

    def test_cache_hit_no_request(self, requests_mock):
        rows = [{"q": "cached?", "a": "yes"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = _cache_path(tmpdir, "test/repo", None, "test")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(rows, f)
            result = load_hf_dataset("test/repo", "test", tmpdir)
            assert len(result) == 1
            assert not requests_mock.called
```

- [ ] **Step 2: 验证测试失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_hf_loader.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: 实现 hf_loader.py**

```python
# benchmark/adapters/hf_loader.py
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
```

- [ ] **Step 4: 添加 requests-mock 测试依赖**

在 `pyproject.toml` 的 `dev` 依赖中添加 `"requests-mock>=1.11"`：

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "requests-mock>=1.11"]
```

安装：`uv pip install requests-mock`

- [ ] **Step 5: 验证测试通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_hf_loader.py -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```
git add benchmark/adapters/hf_loader.py tests/test_hf_loader.py pyproject.toml
git commit -m "feat(adapters): 新增 hf_loader，用 HF API 替代 datasets 库"
```

---

### Task 8: 更新 5 个 adapter 使用 hf_loader

**Files:**
- Modify: `benchmark/adapters/math_adapter.py`
- Modify: `benchmark/adapters/gsm8k_adapter.py`
- Modify: `benchmark/adapters/mmlu_adapter.py`
- Modify: `benchmark/adapters/mmlu_pro_adapter.py`
- Modify: `benchmark/adapters/bigcodebench_adapter.py`

---

- [ ] **Step 1: 更新 math_adapter.py**

替换 import 和 load_dataset 调用：

```python
# 删除
from datasets import load_dataset

# 新增
from benchmark.adapters.hf_loader import load_hf_dataset
```

```python
# 替换 load_dataset() 为 load_hf_dataset()
dataset = load_hf_dataset(
    "nlile/hendrycks-MATH-benchmark",
    split="test",
    cache_dir=cache_dir,
)
```

- [ ] **Step 2: 更新 gsm8k_adapter.py**

```python
# 删除
from datasets import load_dataset

# 新增
from benchmark.adapters.hf_loader import load_hf_dataset
```

```python
# 替换
dataset = load_hf_dataset(
    "openai/gsm8k",
    split="test",
    cache_dir=cache_dir,
    config_name="main",
)
```

- [ ] **Step 3: 更新 mmlu_adapter.py**

```python
# 删除
from datasets import load_dataset

# 新增
from benchmark.adapters.hf_loader import load_hf_dataset
```

替换两层 try/except 中的 load_dataset 调用：

```python
try:
    dataset = load_hf_dataset("cais/mmlu", split="test", cache_dir=cache_dir, config_name=subject)
except Exception as e:
    try:
        dataset = load_hf_dataset("cais/mmlu", split="test", cache_dir=cache_dir, config_name=f"mmlu_{subject}")
    except Exception:
        raise ValueError(f"Failed to load MMLU subject '{subject}'") from e
```

- [ ] **Step 4: 更新 mmlu_pro_adapter.py**

```python
# 删除
from datasets import load_dataset

# 新增
from benchmark.adapters.hf_loader import load_hf_dataset
```

```python
dataset = load_hf_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=cache_dir)
```

- [ ] **Step 5: 更新 bigcodebench_adapter.py**

```python
# 删除
from datasets import load_dataset

# 新增
from benchmark.adapters.hf_loader import load_hf_dataset
```

```python
dataset = load_hf_dataset("bigcode/bigcodebench-hard", split="v0.1.0_hf", cache_dir=cache_dir)
```

- [ ] **Step 6: 验证没有残留的 datasets import**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && grep -r "from datasets import" --include="*.py" benchmark/`
Expected: 无输出

- [ ] **Step 7: 提交**

```
git add benchmark/adapters/
git commit -m "refactor(adapters): 所有 adapter 改用 hf_loader 加载数据"
```
