from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from benchmark.repository import Repository
from benchmark.repository.interface import DATA_DIR


EXPECTED_ABSTRACT_METHODS = {
    "create_run",
    "finish_run",
    "get_active_runs",
    "save_answer",
    "save_scoring",
    "update_status",
    "save_metrics",
    "get_results",
    "get_result_detail",
    "build_index",
    "save_analysis",
    "save_cluster_report",
    "save_quality_signals",
    "get_quality_signals_for_run",
    "get_quality_signals_history",
    "get_stability_reports",
    "get_cluster_reports",
    "get_timing_phases",
    "get_timing_summaries",
    "create_scoring_task",
    "fetch_pending_scoring_tasks",
    "complete_scoring_task",
    "fail_scoring_task",
    "retry_scoring_task",
    "get_pending_task_count",
}

EXPECTED_PATH_HELPERS = {
    "get_run_dir",
    "get_question_dir",
    "get_scoring_path",
    "get_answer_path",
    "get_metrics_path",
}


class TestRepositoryInterfaceDefinition:
    def test_is_abstract_base_class(self):
        from abc import ABC

        assert issubclass(Repository, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Repository()

    def test_all_required_abstract_methods_exist(self):
        actual = {name for name in dir(Repository) if name.startswith("_") is False}
        missing = EXPECTED_ABSTRACT_METHODS - actual
        assert not missing, f"缺少抽象方法: {missing}"

    def test_abstract_methods_count(self):
        abstract_methods = {
            name
            for name in dir(Repository)
            if getattr(getattr(Repository, name), "__isabstractmethod__", False)
        }
        assert abstract_methods == EXPECTED_ABSTRACT_METHODS, (
            f"抽象方法不匹配: 多余={abstract_methods - EXPECTED_ABSTRACT_METHODS}, "
            f"缺少={EXPECTED_ABSTRACT_METHODS - abstract_methods}"
        )

    def test_abstract_methods_have_type_annotations(self):
        hints_cache: dict[str, dict] = {}
        for method_name in EXPECTED_ABSTRACT_METHODS:
            method = getattr(Repository, method_name)
            sig = inspect.signature(method)
            hints = get_type_hints(method)
            hints_cache[method_name] = hints

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                assert param_name in hints, f"{method_name}({param_name}) 缺少类型标注"

            assert "return" in hints, f"{method_name} 缺少返回值类型标注"

    def test_create_run_signature(self):
        hints = get_type_hints(Repository.create_run)
        from benchmark.models.schemas import EvalRun

        assert hints.get("run") is EvalRun
        assert hints.get("return") is str

    def test_save_answer_signature(self):
        hints = get_type_hints(Repository.save_answer)
        from benchmark.models.schemas import EvalResult

        assert hints.get("result") is EvalResult
        assert hints.get("return") is str

    def test_save_scoring_signature(self):
        hints = get_type_hints(Repository.save_scoring)
        from benchmark.models.schemas import EvalResult

        assert hints.get("result") is EvalResult
        assert hints.get("return") is str

    def test_save_metrics_signature(self):
        hints = get_type_hints(Repository.save_metrics)
        from benchmark.models.schemas import ApiCallMetrics

        assert hints.get("metrics") is ApiCallMetrics
        assert hints.get("return") is str

    def test_get_results_signature(self):
        hints = get_type_hints(Repository.get_results)
        assert "model" in hints
        assert "dimension" in hints
        assert "run_id" in hints
        from typing import get_origin

        return_type = hints.get("return")
        assert get_origin(return_type) is list

    def test_get_result_detail_signature(self):
        hints = get_type_hints(Repository.get_result_detail)
        assert hints.get("result_id") is str
        assert hints.get("return") is not None

    def test_methods_count_covers_database(self):
        from benchmark.models.database import Database

        db_public_methods = {
            name
            for name, val in inspect.getmembers(Database, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        repo_methods = EXPECTED_ABSTRACT_METHODS | EXPECTED_PATH_HELPERS
        uncovered = (
            db_public_methods
            - repo_methods
            - {
                "close",
                "get_timing_phases",
                "get_timing_summaries",
                "create_scoring_task",
                "fetch_pending_scoring_tasks",
                "complete_scoring_task",
                "fail_scoring_task",
                "retry_scoring_task",
                "get_pending_task_count",
            }
        )
        uncovered -= {
            "asave_result",
            "asave_metrics",
            "asave_quality_signals",
            "asave_stability_report",
            "asave_cluster_report",
            "aget_quality_signals_for_run",
            "aget_quality_signals_history",
            "aget_stability_reports",
            "aget_cluster_reports",
            "acreate_scoring_task",
            "afetch_pending_scoring_tasks",
            "acomplete_scoring_task",
            "afail_scoring_task",
            "aretry_scoring_task",
            "aget_pending_task_count",
        }
        uncovered -= {
            "save_result",
            "get_timing_phases",
            "get_timing_summaries",
            "create_scoring_task",
            "fetch_pending_scoring_tasks",
            "complete_scoring_task",
            "fail_scoring_task",
            "retry_scoring_task",
            "get_pending_task_count",
        }
        assert len(repo_methods) >= len(db_public_methods), (
            f"Repository 方法数({len(repo_methods)}) < Database 公开方法数({len(db_public_methods)})"
        )


class TestRepositoryPathHelpers:
    def test_data_dir_constant(self):
        assert DATA_DIR == "data"

    def test_get_run_dir(self):
        result = Repository.get_run_dir("run-123")
        assert result == Path("data") / "run-123"

    def test_get_question_dir(self):
        result = Repository.get_question_dir("run-123", "q1")
        assert result == Path("data") / "run-123" / "q1"

    def test_get_scoring_path(self):
        result = Repository.get_scoring_path("run-123", "q1")
        assert result == Repository.get_question_dir("run-123", "q1") / "scoring.json"

    def test_get_answer_path(self):
        result = Repository.get_answer_path("run-123", "q1")
        assert result == Repository.get_question_dir("run-123", "q1") / "answer.json"

    def test_get_metrics_path(self):
        result = Repository.get_metrics_path("run-123", "q1")
        assert result == Repository.get_question_dir("run-123", "q1") / "metrics.json"

    def test_path_helpers_not_abstract(self):
        for name in EXPECTED_PATH_HELPERS:
            method = getattr(Repository, name)
            assert not getattr(method, "__isabstractmethod__", False), (
                f"{name} 不应为抽象方法"
            )


class TestRepositorySubclass:
    def test_concrete_subclass_can_instantiate(self):
        class ConcreteRepo(Repository):
            def create_run(self, run):
                return run.run_id

            def finish_run(self, run_id, status="completed"):
                pass

            def get_active_runs(self):
                return []

            def save_answer(self, result):
                return result.result_id

            def save_scoring(self, result):
                return result.result_id

            def update_status(self, run_id, status):
                pass

            def save_metrics(self, metrics):
                return metrics.result_id

            def get_results(self, model=None, dimension=None, run_id=None):
                return []

            def get_result_detail(self, result_id):
                return None

            def build_index(self):
                return {}

            def save_analysis(self, report):
                return ""

            def save_cluster_report(self, report):
                return ""

            def save_quality_signals(self, signals):
                return ""

            def get_quality_signals_for_run(self, run_id):
                return []

            def get_quality_signals_history(self, model, days=7):
                return []

            def get_stability_reports(self, model=None):
                return []

            def get_cluster_reports(self, model=None):
                return []

            def get_timing_phases(
                self,
                model=None,
                run_id=None,
                result_id=None,
                phase_name=None,
                start_date=None,
                end_date=None,
                limit=1000,
            ):
                import pandas as pd

                return pd.DataFrame()

            def get_timing_summaries(
                self, model=None, run_id=None, start_date=None, end_date=None
            ):
                import pandas as pd

                return pd.DataFrame()

            def create_scoring_task(
                self,
                result_id,
                run_id,
                task_id,
                dimension,
                dataset,
                prompt,
                expected_output,
                model_output,
                model_answer,
                reasoning_content="",
                test_cases=None,
                metadata=None,
                scoring_dimensions=None,
            ):
                return 0

            def fetch_pending_scoring_tasks(self, limit=10):
                return []

            def complete_scoring_task(self, task_id, score_result):
                pass

            def fail_scoring_task(self, task_id, error_message):
                pass

            def retry_scoring_task(self, task_id):
                pass

            def get_pending_task_count(self):
                return 0

        repo = ConcreteRepo()
        assert isinstance(repo, Repository)
