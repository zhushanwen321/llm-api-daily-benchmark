"""FrontCode 适配器测试."""

import json
import os
import tempfile

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


class TestFrontCodeAdapterNewFeatures:
    """test_cases 和 difficulty 透传相关测试."""

    def _write_tasks(self, tasks: list[dict]) -> str:
        tmpdir = tempfile.mkdtemp()
        tasks_file = os.path.join(tmpdir, "tasks.json")
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump({"tasks": tasks}, f, ensure_ascii=False)
        return tmpdir

    def test_load_with_test_cases(self):
        tasks_data = [
            {
                "id": "fc_test_1",
                "type": "javascript",
                "prompt": "实现 debounce 函数",
                "keywords": ["setTimeout", "clearTimeout"],
                "test_cases": [
                    "typeof debounce === 'function'",
                    "debounce 返回一个函数",
                ],
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert len(result) == 1
        assert result[0].test_cases == tasks_data[0]["test_cases"]

    def test_load_without_test_cases(self):
        tasks_data = [
            {
                "id": "fc_test_2",
                "type": "html",
                "prompt": "创建 header 元素",
                "keywords": ["header", "nav"],
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert len(result) == 1
        assert result[0].test_cases == []

    def test_load_with_difficulty_in_metadata(self):
        tasks_data = [
            {
                "id": "fc_test_3",
                "type": "css",
                "prompt": "实现响应式布局",
                "keywords": ["media", "flex"],
                "test_cases": ["存在 @media 查询"],
                "difficulty": "medium",
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert result[0].metadata["difficulty"] == "medium"

    def test_validate_with_test_cases(self):
        tasks_data = [
            {
                "id": "fc_test_4",
                "type": "javascript",
                "prompt": "实现 throttle",
                "keywords": ["setTimeout", "Date"],
                "test_cases": ["typeof throttle === 'function'"],
            }
        ]
        path = self._write_tasks(tasks_data)
        adapter = FrontCodeAdapter()
        result = adapter.load(path=path)
        assert adapter.validate(result[0]) is True
