# tests/test_mmlu_adapter.py
import json
import os
import tempfile

import pytest
from benchmark.adapters.mmlu_adapter import MMLUAdapter
from benchmark.models.schemas import TaskDefinition
from benchmark.adapters.hf_loader import _cache_path


def _write_cache(tmpdir: str, config: str, rows: list[dict]) -> None:
    """写入 mock HF 缓存文件。"""
    cache_file = _cache_path(tmpdir, "cais/mmlu", config, "test")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(rows, f)


def _make_rows(n: int) -> list[dict]:
    return [
        {
            "question": f"Question {i}?",
            "choices": ["A", "B", "C", "D"],
            "answer": i % 4,
        }
        for i in range(n)
    ]


def test_load_returns_tasks():
    """加载应返回题目任务."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_cache(tmpdir, "college_computer_science", _make_rows(3))
        _write_cache(tmpdir, "abstract_algebra", _make_rows(2))
        adapter = MMLUAdapter(subjects=["college_computer_science", "abstract_algebra"])
        tasks = adapter.load(tmpdir)
        assert len(tasks) == 5


def test_all_tasks_have_required_fields():
    """所有任务应包含必需字段."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_cache(tmpdir, "college_computer_science", _make_rows(3))
        _write_cache(tmpdir, "abstract_algebra", _make_rows(2))
        adapter = MMLUAdapter(subjects=["college_computer_science", "abstract_algebra"])
        tasks = adapter.load(tmpdir)
        for task in tasks:
            assert task.task_id
            assert task.dimension == "system-architecture"
            assert task.dataset == "mmlu"
            assert task.prompt
            assert task.expected_output


def test_validate_valid_task():
    """验证有效任务应返回True."""
    adapter = MMLUAdapter(subjects=["college_computer_science", "abstract_algebra"])
    task = TaskDefinition(
        task_id="mmlu_test_1",
        dimension="system-architecture",
        dataset="mmlu",
        prompt="Test question",
        expected_output="A",
        metadata={}
    )
    assert adapter.validate(task) is True


def test_get_dimension():
    """get_dimension应返回system-architecture."""
    adapter = MMLUAdapter(subjects=["college_computer_science", "abstract_algebra"])
    assert adapter.get_dimension() == "system-architecture"
