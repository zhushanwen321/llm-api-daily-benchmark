"""指纹探针单元测试."""

from __future__ import annotations

import pytest
from datetime import datetime

from benchmark.probes.fingerprint.fingerprint_probe import FingerprintProbe
from benchmark.models.schemas import TaskDefinition, EvalResult


class TestFingerprintProbe:
    """指纹探针测试类."""

    def test_load_probes(self):
        """测试加载指纹探针."""
        probe = FingerprintProbe()
        tasks = probe.load_probes()

        assert len(tasks) == 7
        assert all(isinstance(t, TaskDefinition) for t in tasks)
        assert all(t.dimension == "fingerprint" for t in tasks)

        categories = {t.dataset for t in tasks}
        assert "formatting" in categories
        assert "json_handling" in categories

    def test_frequency_property(self):
        """测试频率属性."""
        probe = FingerprintProbe()
        assert probe.frequency == "slow"

    def test_extract_response_features_basic(self):
        """测试基本特征提取."""
        probe = FingerprintProbe()

        response = "Hello world\nThis is a test."
        features = probe._extract_response_features(response, {})

        assert features["length"] == len(response)
        assert features["word_count"] == 6
        assert features["line_count"] == 2

    def test_extract_response_features_markdown(self):
        """测试Markdown特征提取."""
        probe = FingerprintProbe()

        response = "# Header\n- Item 1\n- Item 2\n1. First\n2. Second"
        assertions = {"checks": ["bullet_list", "numbered_list", "headers"]}
        features = probe._extract_response_features(response, assertions)

        assert features["has_headers"] is True
        assert features["has_bullet_list"] is True
        assert features["has_numbered_list"] is True

    def test_extract_response_features_json(self):
        """测试JSON特征提取."""
        probe = FingerprintProbe()

        valid_json = '{"name": "test", "value": 123}'
        features = probe._extract_response_features(valid_json, {"valid_json": True})
        assert features["valid_json"] is True

        invalid_json = "not json"
        features = probe._extract_response_features(invalid_json, {"valid_json": True})
        assert features["valid_json"] is False

    def test_extract_response_features_word_count(self):
        """测试字数特征提取."""
        probe = FingerprintProbe()

        response = "今天天气很好"
        features = probe._extract_response_features(response, {"exact_word_count": 10})
        assert features["actual_word_count"] == 1

    def test_extract_response_features_code_comments(self):
        """测试代码注释特征提取."""
        probe = FingerprintProbe()

        code_with_comments = """# This is a comment
def test():
    \"\"\"Docstring\"\"\"
    pass
"""
        features = probe._extract_response_features(
            code_with_comments, {"has_comments": True}
        )
        assert features["has_hash_comments"] is True
        assert features["has_docstring_double"] is True

    def test_extract_response_features_step_by_step(self):
        """测试逐步思考特征提取."""
        probe = FingerprintProbe()

        response = "首先，我们需要计算。然后，得出结果。最后，验证答案。"
        features = probe._extract_response_features(
            response, {"has_step_by_step": True}
        )
        assert features["has_steps"] is True

        response_simple = "答案是42。"
        features = probe._extract_response_features(
            response_simple, {"has_step_by_step": True}
        )
        assert features["has_steps"] is False

    def test_calculate_score_valid_json(self):
        """测试JSON有效时的得分."""
        probe = FingerprintProbe()

        features = {"valid_json": True}
        assertions = {"valid_json": True}
        score = probe._calculate_score(features, assertions)

        assert score == 70.0

    def test_calculate_score_exact_word_count(self):
        """测试精确字数得分."""
        probe = FingerprintProbe()

        features = {"actual_word_count": 10}
        assertions = {"exact_word_count": 10}
        score = probe._calculate_score(features, assertions)

        assert score == 80.0

    def test_calculate_score_close_word_count(self):
        """测试接近字数得分."""
        probe = FingerprintProbe()

        features = {"actual_word_count": 9}
        assertions = {"exact_word_count": 10}
        score = probe._calculate_score(features, assertions)

        assert score == 65.0

    def test_calculate_score_has_comments(self):
        """测试代码注释得分."""
        probe = FingerprintProbe()

        features = {"has_hash_comments": True}
        assertions = {"has_comments": True}
        score = probe._calculate_score(features, assertions)

        assert score == 70.0

    def test_calculate_score_has_steps(self):
        """测试逐步思考得分."""
        probe = FingerprintProbe()

        features = {"has_steps": True}
        assertions = {"has_step_by_step": True}
        score = probe._calculate_score(features, assertions)

        assert score == 70.0

    def test_extract_features(self):
        """测试特征提取方法."""
        probe = FingerprintProbe()

        result = EvalResult(
            result_id="test_1",
            run_id="run_1",
            task_id="fp_formatting_1",
            task_content="test prompt",
            model_output="test output",
            functional_score=85.0,
            final_score=85.0,
            passed=True,
            execution_time=1.0,
            created_at=datetime.now(),
            details={
                "category": "formatting",
                "features": {"has_headers": True},
            },
        )

        features = probe.extract_features(result)

        assert features["fingerprint_score"] == 85.0
        assert features["category"] == "formatting"
        assert features["features"]["has_headers"] is True
        assert features["response_length"] == 11

    @pytest.mark.asyncio
    async def test_execute_probe(self):
        """测试执行探针."""
        from unittest.mock import AsyncMock
        from benchmark.models.schemas import GenerateResponse

        mock_adapter = AsyncMock()
        mock_adapter.agenerate.return_value = GenerateResponse(
            content="# Header\n- Item 1\n- Item 2",
            duration=1.0,
        )

        probe = FingerprintProbe()

        task = TaskDefinition(
            task_id="fp_formatting_1",
            dimension="fingerprint",
            dataset="formatting",
            prompt="用Markdown格式列出3种水果",
            expected_output="",
            metadata={
                "category": "formatting",
                "assertions": {"checks": ["bullet_list", "headers"]},
            },
        )

        result = await probe.execute_probe(task, "test_model", mock_adapter)

        assert result.task_id == "fp_formatting_1"
        assert result.details["category"] == "formatting"
        assert "features" in result.details
        assert result.details["features"]["has_bullet_list"] is True
        assert result.details["features"]["has_headers"] is True
        mock_adapter.agenerate.assert_called_once()
