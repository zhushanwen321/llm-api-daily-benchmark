"""异步评分系统集成测试。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import patch

import pytest

from benchmark.models.database import Database
from benchmark.models.schemas import (
    ScoreResult,
)
from benchmark.models.scoring_task import PendingScoringTask, ScoringTaskStatus
from benchmark.scorers.llm_scorer.base import LLMScorerBackend
from benchmark.scorers.llm_scorer.factory import create_scorer_backend


# ── Pydantic 模型测试 ──


class TestPendingScoringTask:
    def test_default_status_is_pending(self):
        task = PendingScoringTask(
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
            created_at=datetime.now(),
            result_id="r1",
            run_id="run1",
        )
        assert task.status == ScoringTaskStatus.PENDING

    def test_all_fields_serializable(self):
        task = PendingScoringTask(
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            reasoning_content="think...",
            test_cases=["case1"],
            metadata={"key": "value"},
            scoring_dimensions=["answer_correctness"],
            created_at=datetime.now(),
            result_id="r1",
            run_id="run1",
        )
        data = task.model_dump()
        assert data["task_id"] == "t1"
        assert data["reasoning_content"] == "think..."
        assert data["test_cases"] == ["case1"]


class TestScoringTaskStatus:
    def test_all_values(self):
        assert ScoringTaskStatus.PENDING.value == "pending"
        assert ScoringTaskStatus.PROCESSING.value == "processing"
        assert ScoringTaskStatus.COMPLETED.value == "completed"
        assert ScoringTaskStatus.FAILED.value == "failed"


# ── 数据库操作测试 ──


class TestDatabaseScoringOperations:
    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db_path = tmp_path / "test.db"
        self.db = Database(db_path=self.db_path)
        yield
        self.db.close()

    def test_create_and_fetch_task(self):
        tid = self.db.create_scoring_task(
            result_id="r1",
            run_id="run1",
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        assert tid > 0

        tasks = self.db.fetch_pending_scoring_tasks(limit=1)
        assert len(tasks) == 1
        assert tasks[0]["status"] == "processing"

    def test_complete_task(self):
        tid = self.db.create_scoring_task(
            result_id="r1",
            run_id="run1",
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        self.db.complete_scoring_task(tid, {"answer_correctness": {"score": 100}})

        conn = self.db._get_conn()
        cursor = conn.execute(
            "SELECT status, score_result FROM pending_scoring_tasks WHERE id=?", (tid,)
        )
        row = cursor.fetchone()
        assert row[0] == "completed"
        assert json.loads(row[1])["answer_correctness"]["score"] == 100

    def test_fail_task(self):
        tid = self.db.create_scoring_task(
            result_id="r1",
            run_id="run1",
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        self.db.fail_scoring_task(tid, "API timeout")

        conn = self.db._get_conn()
        cursor = conn.execute(
            "SELECT status, error_message FROM pending_scoring_tasks WHERE id=?", (tid,)
        )
        row = cursor.fetchone()
        assert row[0] == "failed"
        assert "timeout" in row[1]

    def test_retry_task(self):
        tid = self.db.create_scoring_task(
            result_id="r1",
            run_id="run1",
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        # 先锁定
        self.db.fetch_pending_scoring_tasks(limit=1)
        # 重试
        self.db.retry_scoring_task(tid)

        conn = self.db._get_conn()
        cursor = conn.execute(
            "SELECT status, retry_count FROM pending_scoring_tasks WHERE id=?", (tid,)
        )
        row = cursor.fetchone()
        assert row[0] == "pending"
        assert row[1] == 1

    def test_pending_count(self):
        count_before = self.db.get_pending_task_count()
        assert count_before == 0

        self.db.create_scoring_task(
            result_id="r1",
            run_id="run1",
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        count_after = self.db.get_pending_task_count()
        assert count_after == 1

    def test_atomic_lock_prevents_double_fetch(self):
        self.db.create_scoring_task(
            result_id="r1",
            run_id="run1",
            task_id="t1",
            dimension="reasoning",
            dataset="test",
            prompt="1+1",
            expected_output="2",
            model_output="2",
            model_answer="2",
            scoring_dimensions=["answer_correctness"],
        )
        # 第一次拉取应锁定任务
        tasks1 = self.db.fetch_pending_scoring_tasks(limit=1)
        assert len(tasks1) == 1

        # 第二次拉取应为空（任务已被锁定）
        tasks2 = self.db.fetch_pending_scoring_tasks(limit=1)
        assert len(tasks2) == 0


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
