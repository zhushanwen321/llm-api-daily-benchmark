"""Probe 数据集加载和字段完整性测试."""

import json
from collections import Counter

import pytest


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
