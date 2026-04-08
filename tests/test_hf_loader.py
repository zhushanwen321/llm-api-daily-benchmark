"""hf_loader 单元测试。"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import requests

from benchmark.adapters.hf_loader import _cache_path, _fetch_all_rows, load_hf_dataset


@pytest.fixture(autouse=True)
def _ensure_online():
    """确保测试期间离线模式关闭，避免其他测试的副作用。"""
    old = os.environ.get("HF_DATASETS_OFFLINE")
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    yield
    if old is not None:
        os.environ["HF_DATASETS_OFFLINE"] = old


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
        # config_name=None 时 load_hf_dataset 会先调 _resolve_config
        requests_mock.get(
            "https://datasets-server.huggingface.co/configs",
            json={"configs": [{"config": "default"}]},
        )

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
