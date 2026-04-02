"""FrontCode 适配器测试."""

import pytest

from benchmark.adapters.frontcode_adapter import FrontCodeAdapter
from benchmark.models.schemas import TaskDefinition


def test_load_returns_5_tasks():
    """加载应返回5个任务."""
    adapter = FrontCodeAdapter()
    tasks = adapter.load("benchmark/datasets/frontcode")
    assert len(tasks) == 5


def test_tasks_have_correct_types():
    """任务应包含不同类型的前端题目."""
    adapter = FrontCodeAdapter()
    tasks = adapter.load("benchmark/datasets/frontcode")
    task_types = [task.metadata.get("type") for task in tasks]
    expected_types = ["html", "css", "javascript", "react", "complex"]
    assert set(task_types) == set(expected_types)


def test_validate_valid_task():
    """验证有效任务应返回True."""
    adapter = FrontCodeAdapter()
    task = TaskDefinition(
        task_id="frontcode_1",
        dimension="frontend-dev",
        dataset="frontcode",
        prompt="Create a button",
        expected_output="",
        metadata={"type": "html", "keywords": ["button"]},
    )
    assert adapter.validate(task) is True


def test_get_dimension():
    """get_dimension应返回frontend-dev."""
    adapter = FrontCodeAdapter()
    assert adapter.get_dimension() == "frontend-dev"
