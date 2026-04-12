"""评分系统集成测试 - 适配 FileRepository."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.repository import FileRepository
from benchmark.scorers.llm_scorer.base import LLMScorerBackend
from benchmark.scorers.llm_scorer.factory import create_scorer_backend
from benchmark.models.schemas import ScoreResult


# ── Repository Scoring 操作测试 ──


class TestRepositoryScoringOperations:
    """测试 FileRepository 的评分任务管理功能."""

    @pytest.fixture(autouse=True)
    def setup_repo(self, tmp_path):
        self.data_dir = tmp_path / "test_data"
        self.repo = FileRepository(data_root=self.data_dir)
        # 创建测试用的 benchmark run
        self.benchmark_id = self.repo.create_benchmark_run(
            model="test-model",
            dimension="reasoning",
            dataset="test",
            questions=["q1", "q2", "q3"],
        )
        yield

    def test_create_and_fetch_task(self):
        """测试创建评分任务并获取待处理任务."""
        task_id = self.repo.create_scoring_task(
            result_id="r1",
            run_id=self.benchmark_id,
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        assert task_id is not None
        assert isinstance(task_id, int)
        assert task_id > 0

        # 获取待评分任务 - 检查 scoring_status 字段
        tasks = self.repo.fetch_pending_scoring_tasks(limit=1)
        assert len(tasks) == 1
        assert tasks[0]["scoring_status"] == "pending"
        assert tasks[0]["task_id"] == "t1"

    def test_complete_task(self):
        """测试完成任务评分."""
        task_id = self.repo.create_scoring_task(
            result_id="r1",
            run_id=self.benchmark_id,
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )

        # 完成任务 - 传入完整的 scoring 数据
        self.repo.complete_scoring_task(
            task_id,
            {
                "functional_score": 100.0,
                "quality_score": 100.0,
                "final_score": 100.0,
                "passed": True,
                "details": {"answer_correctness": {"score": 100, "passed": True}},
            },
        )

        # 验证完成后没有 pending 任务
        tasks = self.repo.fetch_pending_scoring_tasks(limit=10)
        assert len(tasks) == 0

    def test_fail_task(self):
        """测试标记任务失败."""
        task_id = self.repo.create_scoring_task(
            result_id="r1",
            run_id=self.benchmark_id,
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )

        # 标记失败 - fail_scoring_task 返回 None，通过副作用验证
        self.repo.fail_scoring_task(task_id, "API timeout")

        # 验证失败后没有 pending 任务
        tasks = self.repo.fetch_pending_scoring_tasks(limit=10)
        assert len(tasks) == 0

    def test_retry_task(self):
        """测试重试失败任务."""
        task_id = self.repo.create_scoring_task(
            result_id="r1",
            run_id=self.benchmark_id,
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )

        # 先失败
        self.repo.fail_scoring_task(task_id, "API timeout")

        # 重试 - retry_scoring_task 返回 None，通过副作用验证
        self.repo.retry_scoring_task(task_id)

        # 验证任务重新变为 pending
        tasks = self.repo.fetch_pending_scoring_tasks(limit=10)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"
        assert tasks[0]["scoring_status"] == "pending"

    def test_pending_count(self):
        """测试获取待评分任务计数."""
        count_before = self.repo.get_pending_task_count()
        assert count_before == 0

        # 创建任务
        self.repo.create_scoring_task(
            result_id="r1",
            run_id=self.benchmark_id,
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )

        count_after = self.repo.get_pending_task_count()
        assert count_after == 1

        # 创建第二个任务
        self.repo.create_scoring_task(
            result_id="r2",
            run_id=self.benchmark_id,
            task_id="t2",
            dimension="reasoning",
            dataset="test",
            prompt="2+2",
            expected_output="4",
            model_output="4",
            model_answer="4",
            scoring_dimensions=["answer_correctness"],
        )

        count_final = self.repo.get_pending_task_count()
        assert count_final == 2

    def test_fetch_limit(self):
        """测试获取任务数量限制."""
        # 创建 5 个任务
        for i in range(5):
            self.repo.create_scoring_task(
                result_id=f"r{i}",
                run_id=self.benchmark_id,
                task_id=f"t{i}",
                dimension="reasoning",
                dataset="test",
                prompt=f"{i}+{i}",
                expected_output=f"{i * 2}",
                model_output=f"{i * 2}",
                model_answer=f"{i * 2}",
                scoring_dimensions=["answer_correctness"],
            )

        # 限制获取 3 个
        tasks = self.repo.fetch_pending_scoring_tasks(limit=3)
        assert len(tasks) == 3


# ── 工厂函数测试 ──


class TestFactory:
    def test_default_backend(self):
        backend = create_scorer_backend("qwen_cli")
        assert type(backend).__name__ == "QwenCLIBackend"

    def test_api_backend(self):
        backend = create_scorer_backend("llm_api")
        assert type(backend).__name__ == "LLMAPIScorerBackend"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown scoring backend"):
            create_scorer_backend("unknown")

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"SCORING_BACKEND_TYPE": "llm_api"}):
            backend = create_scorer_backend()
            assert type(backend).__name__ == "LLMAPIScorerBackend"


# ── 抽象基类测试 ──


class TestAbstractBackend:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMScorerBackend()

    def test_concrete_implementation(self):
        class MockBackend(LLMScorerBackend):
            async def score(self, context, dimensions):
                return {
                    "test": ScoreResult(
                        score=100, passed=True, details={}, reasoning="ok"
                    )
                }

            async def health_check(self):
                return True

        backend = MockBackend()
        assert backend is not None
