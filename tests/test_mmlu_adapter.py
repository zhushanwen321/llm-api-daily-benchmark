# tests/test_mmlu_adapter.py
import pytest
from benchmark.adapters.mmlu_adapter import MMLUAdapter

def test_load_returns_5_tasks():
    """加载应返回5个任务."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    tasks = adapter.load("benchmark/datasets/mmlu")
    assert len(tasks) == 5

def test_all_tasks_have_required_fields():
    """所有任务应包含必需字段."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    tasks = adapter.load("benchmark/datasets/mmlu")
    for task in tasks:
        assert task.task_id
        assert task.dimension == "system-architecture"
        assert task.dataset == "mmlu"
        assert task.prompt
        assert task.expected_output  # MMLU是选择题，expected_output是正确选项

def test_validate_valid_task():
    """验证有效任务应返回True."""
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    from benchmark.models.schemas import TaskDefinition
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
    adapter = MMLUAdapter(subjects=["computer_science", "abstract_algebra"])
    assert adapter.get_dimension() == "system-architecture"
