"""Probe 数据集加载和字段完整性测试."""

import json
import os
import tempfile
from collections import Counter

import pytest

from benchmark.adapters.probe_adapter import ProbeAdapter
from benchmark.models.schemas import TaskDefinition


TASKS_FILE = "benchmark/datasets/probe/tasks.json"

REQUIRED_FIELDS = [
    "id",
    "type",
    "prompt",
    "expected_answer",
    "expected_format",
    "instruction_constraints",
    "difficulty",
    "source",
]

VALID_TYPES = {"format", "reasoning", "consistency", "instruction", "known_answer"}
VALID_FORMATS = {"json", "boxed", "text"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

EXPECTED_TYPE_COUNTS = {
    "format": 5,
    "reasoning": 5,
    "consistency": 3,
    "instruction": 4,
    "known_answer": 3,
}


@pytest.fixture(scope="module")
def tasks_data():
    with open(TASKS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tasks(tasks_data):
    return tasks_data["tasks"]


class TestProbeStructure:
    """数据集整体结构测试."""

    def test_is_valid_json(self, tasks_data):
        """文件应该是合法的 JSON."""
        assert "tasks" in tasks_data
        assert isinstance(tasks_data["tasks"], list)

    def test_total_count_is_20(self, tasks):
        """总共 20 题."""
        assert len(tasks) == 20

    def test_all_ids_unique(self, tasks):
        """ID 不能重复."""
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids))

    def test_id_prefix(self, tasks):
        """ID 应以 probe_ 开头."""
        for t in tasks:
            assert t["id"].startswith("probe_"), f"Bad ID: {t['id']}"

    def test_type_counts(self, tasks):
        """各类型题目数量正确."""
        type_counts = Counter(t["type"] for t in tasks)
        for type_name, expected in EXPECTED_TYPE_COUNTS.items():
            assert type_counts[type_name] == expected, (
                f"Type '{type_name}': expected {expected}, got {type_counts[type_name]}"
            )


class TestProbeFields:
    """每题字段完整性测试."""

    def test_required_fields_present(self, tasks):
        """每题包含所有必需字段."""
        for t in tasks:
            missing = [f for f in REQUIRED_FIELDS if f not in t]
            assert not missing, f"Task {t['id']} missing fields: {missing}"

    def test_type_is_valid(self, tasks):
        """type 字段值合法."""
        for t in tasks:
            assert t["type"] in VALID_TYPES, f"Task {t['id']}: invalid type '{t['type']}'"

    def test_expected_format_is_valid(self, tasks):
        """expected_format 字段值合法."""
        for t in tasks:
            assert t["expected_format"] in VALID_FORMATS, (
                f"Task {t['id']}: invalid format '{t['expected_format']}'"
            )

    def test_difficulty_is_valid(self, tasks):
        """difficulty 字段值合法."""
        for t in tasks:
            assert t["difficulty"] in VALID_DIFFICULTIES, (
                f"Task {t['id']}: invalid difficulty '{t['difficulty']}'"
            )

    def test_source_is_probe(self, tasks):
        """source 固定为 probe."""
        for t in tasks:
            assert t["source"] == "probe", f"Task {t['id']}: source should be 'probe'"

    def test_instruction_constraints_is_list(self, tasks):
        """instruction_constraints 应为列表."""
        for t in tasks:
            assert isinstance(t["instruction_constraints"], list), (
                f"Task {t['id']}: instruction_constraints must be a list"
            )

    def test_prompt_is_non_empty(self, tasks):
        """prompt 不能为空."""
        for t in tasks:
            assert t["prompt"].strip(), f"Task {t['id']}: prompt is empty"

    def test_expected_answer_is_string(self, tasks):
        """expected_answer 应为字符串（可为空字符串）."""
        for t in tasks:
            assert isinstance(t["expected_answer"], str), (
                f"Task {t['id']}: expected_answer must be a string"
            )


class TestProbeContent:
    """题目内容合理性测试."""

    def test_format_tasks_require_json_output(self, tasks):
        """格式遵从题的 expected_format 应为 json."""
        for t in tasks:
            if t["type"] == "format":
                assert t["expected_format"] == "json", (
                    f"Task {t['id']}: format task should have expected_format='json'"
                )

    def test_format_tasks_have_format_instructions(self, tasks):
        """格式遵从题的 prompt 应包含 JSON 格式要求."""
        for t in tasks:
            if t["type"] == "format":
                assert "JSON" in t["prompt"], (
                    f"Task {t['id']}: format task prompt should mention JSON"
                )

    def test_reasoning_tasks_have_numeric_answer(self, tasks):
        """推理题应有确定的数值答案（非空字符串）."""
        for t in tasks:
            if t["type"] == "reasoning":
                assert t["expected_answer"].strip(), (
                    f"Task {t['id']}: reasoning task should have non-empty expected_answer"
                )

    def test_known_answer_tasks_are_simple(self, tasks):
        """已知答案题应为 text 格式且不需要 JSON."""
        for t in tasks:
            if t["type"] == "known_answer":
                assert t["expected_format"] == "text", (
                    f"Task {t['id']}: known_answer task should be text format"
                )

    def test_instruction_tasks_have_constraints(self, tasks):
        """指令遵从题应有 instruction_constraints."""
        for t in tasks:
            if t["type"] == "instruction":
                assert len(t["instruction_constraints"]) > 0, (
                    f"Task {t['id']}: instruction task should have constraints"
                )

    def test_consistency_tasks_have_long_context(self, tasks):
        """长上下文一致性题的 prompt 应较长（>100字符）."""
        for t in tasks:
            if t["type"] == "consistency":
                assert len(t["prompt"]) > 100, (
                    f"Task {t['id']}: consistency task prompt too short ({len(t['prompt'])} chars)"
                )


# ── ProbeAdapter 测试 ──────────────────────────────────────────────────────────


@pytest.fixture
def adapter():
    return ProbeAdapter()


@pytest.fixture(scope="module")
def loaded_tasks():
    return ProbeAdapter().load()


class TestProbeAdapterLoad:
    """ProbeAdapter.load 测试."""

    def test_load_returns_20_tasks(self, loaded_tasks):
        assert len(loaded_tasks) == 20

    def test_load_task_ids_unique(self, loaded_tasks):
        ids = [t.task_id for t in loaded_tasks]
        assert len(ids) == len(set(ids))

    def test_load_default_path(self, adapter):
        tasks = adapter.load()
        assert len(tasks) == 20

    def test_load_custom_path(self, adapter):
        tasks = adapter.load(path="benchmark/datasets/probe")
        assert len(tasks) == 20

    def test_load_missing_file(self, adapter):
        with pytest.raises(FileNotFoundError, match="Probe tasks file not found"):
            adapter.load(path="/nonexistent/path")

    def test_load_invalid_json(self, adapter, tmp_path):
        bad_file = tmp_path / "tasks.json"
        bad_file.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            adapter.load(path=str(tmp_path))

    def test_load_missing_required_fields(self, adapter, tmp_path):
        bad_data = {"tasks": [{"id": "probe_bad_1", "prompt": "test"}]}
        bad_file = tmp_path / "tasks.json"
        bad_file.write_text(json.dumps(bad_data), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            adapter.load(path=str(tmp_path))

    def test_task_definition_fields(self, loaded_tasks):
        task = loaded_tasks[0]
        assert task.dimension == "probe"
        assert task.dataset == "probe"
        assert task.prompt
        assert isinstance(task.metadata, dict)

    def test_metadata_contains_probe_fields(self, loaded_tasks):
        for task in loaded_tasks:
            meta = task.metadata
            assert "type" in meta
            assert "expected_format" in meta
            assert "instruction_constraints" in meta
            assert "difficulty" in meta
            assert "source" in meta

    def test_expected_output_matches_expected_answer(self, loaded_tasks, tasks):
        """expected_output 应等于 tasks.json 中的 expected_answer."""
        by_id = {t["id"]: t for t in tasks}
        for task in loaded_tasks:
            assert task.expected_output == by_id[task.task_id]["expected_answer"]

    def test_prompt_not_modified(self, loaded_tasks, tasks):
        """prompt 应原样保留，不做额外包装."""
        by_id = {t["id"]: t for t in tasks}
        for task in loaded_tasks:
            assert task.prompt == by_id[task.task_id]["prompt"]


class TestProbeAdapterValidate:
    """ProbeAdapter.validate 测试."""

    def test_validate_valid_task(self, adapter):
        task = TaskDefinition(
            task_id="probe_test",
            dimension="probe",
            dataset="probe",
            prompt="test prompt",
            expected_output="test answer",
        )
        assert adapter.validate(task) is True

    def test_validate_empty_task_id(self, adapter):
        task = TaskDefinition(
            task_id="",
            dimension="probe",
            dataset="probe",
            prompt="test",
            expected_output="answer",
        )
        assert adapter.validate(task) is False

    def test_validate_empty_prompt(self, adapter):
        task = TaskDefinition(
            task_id="probe_test",
            dimension="probe",
            dataset="probe",
            prompt="",
            expected_output="answer",
        )
        assert adapter.validate(task) is False

    def test_validate_empty_expected_output(self, adapter):
        """expected_output 为空字符串仍应通过验证（instruction 类型题目允许空答案）."""
        task = TaskDefinition(
            task_id="probe_test",
            dimension="probe",
            dataset="probe",
            prompt="test",
            expected_output="",
        )
        assert adapter.validate(task) is True

    def test_validate_all_loaded_tasks(self, adapter, loaded_tasks):
        """所有从文件加载的任务都应通过验证."""
        for task in loaded_tasks:
            assert adapter.validate(task) is True, f"Failed for {task.task_id}"


class TestProbeAdapterGetDimension:
    """ProbeAdapter.get_dimension 测试."""

    def test_get_dimension(self, adapter):
        assert adapter.get_dimension() == "probe"
