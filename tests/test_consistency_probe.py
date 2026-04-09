"""一致性探针单元测试."""

from __future__ import annotations

import pytest
from datetime import datetime

from benchmark.probes.consistency.consistency_probe import ConsistencyProbe
from benchmark.models.schemas import TaskDefinition, EvalResult


class TestConsistencyProbe:
    """一致性探针测试类."""

    def test_load_probes(self):
        """测试加载一致性探针."""
        probe = ConsistencyProbe()
        tasks = probe.load_probes()

        assert len(tasks) == 9
        assert all(isinstance(t, TaskDefinition) for t in tasks)
        assert all(t.dimension == "consistency" for t in tasks)

        categories = {t.dataset for t in tasks}
        assert "paraphrase_consistency" in categories
        assert "knowledge_consistency" in categories
        assert "logical_consistency" in categories

    def test_frequency_property(self):
        """测试频率属性."""
        probe = ConsistencyProbe()
        assert probe.frequency == "slow"

    def test_calculate_similarity_exact_match(self):
        """测试精确匹配相似度."""
        probe = ConsistencyProbe()

        similarity = probe._calculate_similarity("答案是42", "42")
        assert similarity == 100.0

    def test_calculate_similarity_case_insensitive(self):
        """测试大小写不敏感匹配."""
        probe = ConsistencyProbe()

        similarity = probe._calculate_similarity("Beijing", "beijing")
        assert similarity == 100.0

    def test_calculate_similarity_partial_match(self):
        """测试部分匹配相似度."""
        probe = ConsistencyProbe()

        similarity = probe._calculate_similarity("the quick brown fox", "quick fox")
        assert similarity > 0.0
        assert similarity <= 100.0

    def test_calculate_similarity_no_match(self):
        """测试无匹配相似度."""
        probe = ConsistencyProbe()

        similarity = probe._calculate_similarity("上海", "北京")
        assert similarity == 0.0

    def test_calculate_similarity_empty_expected(self):
        """测试空期望值的相似度."""
        probe = ConsistencyProbe()

        similarity = probe._calculate_similarity("任何回答", "")
        assert similarity == 0.0

    def test_calculate_group_consistency(self):
        """测试组一致性计算."""
        probe = ConsistencyProbe()

        results = [
            EvalResult(
                result_id="1",
                run_id="r1",
                task_id="t1",
                task_content="prompt",
                model_output="output",
                functional_score=100.0,
                final_score=100.0,
                passed=True,
                execution_time=1.0,
                created_at=datetime.now(),
                details={},
            ),
            EvalResult(
                result_id="2",
                run_id="r1",
                task_id="t2",
                task_content="prompt",
                model_output="output",
                functional_score=80.0,
                final_score=80.0,
                passed=True,
                execution_time=1.0,
                created_at=datetime.now(),
                details={},
            ),
        ]

        consistency = probe.calculate_group_consistency(results)

        assert consistency["consistency_score"] == 90.0
        assert consistency["variance"] == 100.0
        assert consistency["all_passed"] is True
        assert consistency["sample_count"] == 2

    def test_calculate_group_consistency_empty(self):
        """测试空结果的一致性计算."""
        probe = ConsistencyProbe()

        consistency = probe.calculate_group_consistency([])

        assert consistency["consistency_score"] == 0.0
        assert consistency["variance"] == 0.0

    def test_calculate_group_consistency_single_result(self):
        """测试单结果的一致性计算."""
        probe = ConsistencyProbe()

        results = [
            EvalResult(
                result_id="1",
                run_id="r1",
                task_id="t1",
                task_content="prompt",
                model_output="output",
                functional_score=85.0,
                final_score=85.0,
                passed=True,
                execution_time=1.0,
                created_at=datetime.now(),
                details={},
            ),
        ]

        consistency = probe.calculate_group_consistency(results)

        assert consistency["consistency_score"] == 85.0
        assert consistency["variance"] == 0.0
        assert consistency["all_passed"] is True

    def test_extract_features(self):
        """测试特征提取."""
        probe = ConsistencyProbe()

        result = EvalResult(
            result_id="test_1",
            run_id="run_1",
            task_id="cons_math_1a",
            task_content="test prompt",
            model_output="42",
            functional_score=100.0,
            final_score=100.0,
            passed=True,
            execution_time=1.0,
            created_at=datetime.now(),
            details={
                "group_id": "math_basic",
                "category": "paraphrase_consistency",
            },
        )

        features = probe.extract_features(result)

        assert features["consistency_score"] == 100.0
        assert features["group_id"] == "math_basic"
        assert features["category"] == "paraphrase_consistency"
        assert features["passed"] is True

    @pytest.mark.asyncio
    async def test_execute_probe(self):
        """测试执行探针."""
        from unittest.mock import AsyncMock
        from benchmark.models.schemas import GenerateResponse

        mock_adapter = AsyncMock()
        mock_adapter.agenerate.return_value = GenerateResponse(
            content="答案是42。",
            duration=1.0,
        )

        probe = ConsistencyProbe()

        task = TaskDefinition(
            task_id="cons_math_1a",
            dimension="consistency",
            dataset="paraphrase_consistency",
            prompt="15加27等于多少？",
            expected_output="42",
            metadata={
                "group_id": "math_basic",
                "category": "paraphrase_consistency",
            },
        )

        result = await probe.execute_probe(task, "test_model", mock_adapter)

        assert result.task_id == "cons_math_1a"
        assert result.passed is True
        assert result.functional_score == 100.0
        assert result.details["expected"] == "42"
        mock_adapter.agenerate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_probe_low_similarity(self):
        """测试低相似度的探针执行."""
        from unittest.mock import AsyncMock
        from benchmark.models.schemas import GenerateResponse

        mock_adapter = AsyncMock()
        mock_adapter.agenerate.return_value = GenerateResponse(
            content="我不知道答案。",
            duration=1.0,
        )

        probe = ConsistencyProbe()

        task = TaskDefinition(
            task_id="cons_math_1a",
            dimension="consistency",
            dataset="paraphrase_consistency",
            prompt="15加27等于多少？",
            expected_output="42",
            metadata={
                "group_id": "math_basic",
                "category": "paraphrase_consistency",
            },
        )

        result = await probe.execute_probe(task, "test_model", mock_adapter)

        assert result.task_id == "cons_math_1a"
        assert result.passed is False
        assert result.functional_score < 70.0
