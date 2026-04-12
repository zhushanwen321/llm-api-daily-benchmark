from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.models.schemas import ApiCallMetrics, EvalResult
from benchmark.repository.handlers.answer_handler import AnswerHandler


def _make_result(
    result_id: str = "r1",
    run_id: str = "run1",
    task_id: str = "q1",
    model_output: str = "output",
    model_answer: str = "42",
    expected_output: str = "42",
    execution_time: float = 1.0,
) -> EvalResult:
    return EvalResult(
        result_id=result_id,
        run_id=run_id,
        task_id=task_id,
        task_content="test prompt",
        model_output=model_output,
        model_think="thinking",
        model_answer=model_answer,
        expected_output=expected_output,
        functional_score=100.0,
        quality_score=90.0,
        final_score=95.0,
        passed=True,
        execution_time=execution_time,
        created_at=datetime.now(timezone.utc),
    )


def _make_metrics(
    result_id: str = "r1",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int = 20,
    duration: float = 2.0,
    tokens_per_second: float = 25.0,
    ttft: float = 0.3,
    ttft_content: float = 0.5,
) -> ApiCallMetrics:
    return ApiCallMetrics(
        result_id=result_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        duration=duration,
        tokens_per_second=tokens_per_second,
        ttft=ttft,
        ttft_content=ttft_content,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def handler(tmp_root: Path) -> AnswerHandler:
    return AnswerHandler(data_root=tmp_root)


class TestSaveAnswer:
    def test_creates_answer_jsonl(self, handler: AnswerHandler, tmp_root: Path) -> None:
        result = _make_result()
        metrics = _make_metrics()
        handler.save_answer(result, metrics)
        assert (tmp_root / "run1" / "q1" / "answer.jsonl").exists()

    def test_each_line_is_valid_json(
        self, handler: AnswerHandler, tmp_root: Path
    ) -> None:
        result = _make_result()
        metrics = _make_metrics()
        handler.save_answer(result, metrics)
        lines = (
            (tmp_root / "run1" / "q1" / "answer.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert isinstance(parsed, dict)

    def test_contains_answer_and_api_metrics_top_level_keys(
        self, handler: AnswerHandler, tmp_root: Path
    ) -> None:
        result = _make_result()
        metrics = _make_metrics()
        handler.save_answer(result, metrics)
        line = (
            (tmp_root / "run1" / "q1" / "answer.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        parsed = json.loads(line)
        assert "answer" in parsed
        assert "api_metrics" in parsed

    def test_answer_fields(self, handler: AnswerHandler, tmp_root: Path) -> None:
        result = _make_result()
        metrics = _make_metrics()
        handler.save_answer(result, metrics)
        line = (
            (tmp_root / "run1" / "q1" / "answer.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        parsed = json.loads(line)
        answer = parsed["answer"]
        assert answer["result_id"] == "r1"
        assert answer["run_id"] == "run1"
        assert answer["task_id"] == "q1"
        assert answer["model_output"] == "output"
        assert answer["model_think"] == "thinking"
        assert answer["model_answer"] == "42"
        assert answer["expected_output"] == "42"
        assert answer["execution_time"] == 1.0

    def test_api_metrics_fields(self, handler: AnswerHandler, tmp_root: Path) -> None:
        result = _make_result()
        metrics = _make_metrics()
        handler.save_answer(result, metrics)
        line = (
            (tmp_root / "run1" / "q1" / "answer.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        parsed = json.loads(line)
        m = parsed["api_metrics"]
        assert m["prompt_tokens"] == 100
        assert m["completion_tokens"] == 50
        assert m["reasoning_tokens"] == 20
        assert m["duration"] == 2.0
        assert m["tokens_per_second"] == 25.0
        assert m["ttft"] == 0.3
        assert m["ttft_content"] == 0.5

    def test_multiple_saves_append_lines(
        self, handler: AnswerHandler, tmp_root: Path
    ) -> None:
        result1 = _make_result(result_id="r1", task_id="q1")
        metrics1 = _make_metrics(result_id="r1")
        result2 = _make_result(result_id="r2", task_id="q1")
        metrics2 = _make_metrics(result_id="r2")
        handler.save_answer(result1, metrics1)
        handler.save_answer(result2, metrics2)
        lines = (
            (tmp_root / "run1" / "q1" / "answer.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2
        assert json.loads(lines[0])["answer"]["result_id"] == "r1"
        assert json.loads(lines[1])["answer"]["result_id"] == "r2"

    def test_without_metrics(self, handler: AnswerHandler, tmp_root: Path) -> None:
        result = _make_result()
        handler.save_answer(result, metrics=None)
        line = (
            (tmp_root / "run1" / "q1" / "answer.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        parsed = json.loads(line)
        assert parsed["answer"]["result_id"] == "r1"
        assert parsed["api_metrics"] is None


class TestGetAnswer:
    def test_returns_latest_answer(self, handler: AnswerHandler) -> None:
        result1 = _make_result(result_id="r1", model_answer="first")
        metrics1 = _make_metrics(result_id="r1")
        result2 = _make_result(result_id="r2", model_answer="second")
        metrics2 = _make_metrics(result_id="r2")
        handler.save_answer(result1, metrics1)
        handler.save_answer(result2, metrics2)
        data = handler.get_answer(run_id="run1", task_id="q1")
        assert data["answer"]["model_answer"] == "second"

    def test_returns_dict_with_answer_and_api_metrics(
        self, handler: AnswerHandler
    ) -> None:
        result = _make_result()
        metrics = _make_metrics()
        handler.save_answer(result, metrics)
        data = handler.get_answer(run_id="run1", task_id="q1")
        assert "answer" in data
        assert "api_metrics" in data

    def test_nonexistent_raises_file_not_found(self, handler: AnswerHandler) -> None:
        with pytest.raises(FileNotFoundError):
            handler.get_answer(run_id="nonexistent", task_id="q1")


class TestGetAnswersByRun:
    def test_returns_all_answers_in_run(self, handler: AnswerHandler) -> None:
        result1 = _make_result(result_id="r1", task_id="q1")
        metrics1 = _make_metrics(result_id="r1")
        result2 = _make_result(result_id="r2", task_id="q2")
        metrics2 = _make_metrics(result_id="r2")
        handler.save_answer(result1, metrics1)
        handler.save_answer(result2, metrics2)
        answers = handler.get_answers_by_run(run_id="run1")
        task_ids = {a["answer"]["task_id"] for a in answers}
        assert task_ids == {"q1", "q2"}

    def test_returns_empty_for_nonexistent_run(self, handler: AnswerHandler) -> None:
        answers = handler.get_answers_by_run(run_id="nonexistent")
        assert answers == []

    def test_only_returns_latest_per_task(self, handler: AnswerHandler) -> None:
        result1 = _make_result(result_id="r1", task_id="q1", model_answer="first")
        metrics1 = _make_metrics(result_id="r1")
        result2 = _make_result(result_id="r2", task_id="q1", model_answer="second")
        metrics2 = _make_metrics(result_id="r2")
        handler.save_answer(result1, metrics1)
        handler.save_answer(result2, metrics2)
        answers = handler.get_answers_by_run(run_id="run1")
        assert len(answers) == 1
        assert answers[0]["answer"]["model_answer"] == "second"


class TestRoundtrip:
    def test_write_and_read_preserves_data(self, handler: AnswerHandler) -> None:
        result = _make_result(
            result_id="r-rt",
            run_id="run-rt",
            task_id="q-rt",
            model_output="the output",
            model_answer="99",
            expected_output="99",
            execution_time=3.14,
        )
        metrics = _make_metrics(
            result_id="r-rt",
            prompt_tokens=200,
            completion_tokens=100,
            reasoning_tokens=30,
            duration=5.0,
            tokens_per_second=20.0,
            ttft=0.8,
            ttft_content=1.2,
        )
        handler.save_answer(result, metrics)
        data = handler.get_answer(run_id="run-rt", task_id="q-rt")
        assert data["answer"]["result_id"] == "r-rt"
        assert data["answer"]["model_output"] == "the output"
        assert data["answer"]["model_answer"] == "99"
        assert data["answer"]["execution_time"] == 3.14
        assert data["api_metrics"]["prompt_tokens"] == 200
        assert data["api_metrics"]["ttft"] == 0.8
