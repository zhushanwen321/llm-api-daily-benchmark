"""测试 QualitySignalCollector 的 13 个质量信号提取。"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from benchmark.analysis.quality_signals import QualitySignalCollector
from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun, TaskDefinition


# ── fixtures ──


def _test_db() -> Database:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = Database(db_path=Path(tmp.name))
    return db


def _make_task(task_id: str = "t1", dimension: str = "reasoning") -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        dimension=dimension,
        dataset="gsm8k",
        prompt="What is 1+1?",
        expected_output="2",
    )


def _insert_history(
    db: Database,
    model: str = "test/model",
    dimension: str = "reasoning",
    task_id: str = "t1",
    output_lengths: list[int] | None = None,
    prompt_tokens: list[int] | None = None,
    tps_values: list[float] | None = None,
    ttft_values: list[float] | None = None,
) -> None:
    """向 db 插入历史评测数据，用于 z-score 计算。"""
    now = datetime.now()
    n = max(
        len(output_lengths or []),
        len(prompt_tokens or []),
        len(tps_values or []),
        len(ttft_values or []),
        1,
    )
    for i in range(n):
        run_id = f"hist-run-{i}"
        result_id = f"hist-res-{i}"
        run = EvalRun(
            run_id=run_id,
            model=model,
            dimension=dimension,
            dataset="gsm8k",
            started_at=now - timedelta(days=i),
            finished_at=now - timedelta(days=i),
            status="completed",
        )
        db.create_run(run)
        out_len = (output_lengths or [100])[min(i, len(output_lengths or [100]) - 1)]
        result = EvalResult(
            result_id=result_id,
            run_id=run_id,
            task_id=task_id,
            task_content="test",
            model_output="x" * out_len,
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now - timedelta(days=i),
        )
        db.save_result(result)
        metrics = ApiCallMetrics(
            result_id=result_id,
            prompt_tokens=(prompt_tokens or [50])[
                min(i, len(prompt_tokens or [50]) - 1)
            ],
            completion_tokens=100,
            reasoning_tokens=10,
            duration=1.0,
            tokens_per_second=(tps_values or [50.0])[
                min(i, len(tps_values or [50.0]) - 1)
            ],
            ttft_content=(ttft_values or [0.5])[min(i, len(ttft_values or [0.5]) - 1)],
            created_at=now - timedelta(days=i),
        )
        db.save_metrics(metrics)


# ── 信号 1: format_compliance ──


class TestFormatCompliance:
    def test_reasoning_with_boxed(self):
        assert (
            QualitySignalCollector._check_format(r"答案 \boxed{42}", "reasoning") == 1.0
        )

    def test_reasoning_without_boxed(self):
        assert QualitySignalCollector._check_format("答案是 42", "reasoning") == 0.0

    def test_backend_dev_valid_json(self):
        output = 'The result is {"key": "value"}'
        assert QualitySignalCollector._check_format(output, "backend-dev") == 1.0

    def test_backend_dev_no_json(self):
        assert (
            QualitySignalCollector._check_format("no json here", "backend-dev") == 0.0
        )

    def test_frontend_dev_json_with_code(self):
        output = '{"code": "<div>Hello</div>", "explanation": "..."}'
        assert QualitySignalCollector._check_format(output, "frontend-dev") == 1.0

    def test_frontend_dev_json_without_code(self):
        output = '{"explanation": "..."}'
        assert QualitySignalCollector._check_format(output, "frontend-dev") == 0.0

    def test_other_dimension(self):
        assert QualitySignalCollector._check_format("anything", "unknown-dim") == 1.0

    def test_empty_output(self):
        assert QualitySignalCollector._check_format("", "reasoning") == 0.0


# ── 信号 2: repetition_ratio ──


class TestRepetitionRatio:
    def test_no_repetition(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert QualitySignalCollector._calc_repetition(text) == 0.0

    def test_full_repetition(self):
        text = "abc abc abc abc"
        # trigrams: (abc,abc,abc) x2, all same => 1 dup out of 2 total
        ratio = QualitySignalCollector._calc_repetition(text)
        assert ratio == 0.5  # 2 trigrams, 1 unique, (2-1)/2 = 0.5

    def test_short_text(self):
        assert QualitySignalCollector._calc_repetition("hi") == 0.0

    def test_empty_text(self):
        assert QualitySignalCollector._calc_repetition("") == 0.0

    def test_two_words(self):
        assert QualitySignalCollector._calc_repetition("hello world") == 0.0

    def test_partial_repetition(self):
        text = "a b c a b c a b"
        # words: a b c a b c a b → 8 words, 6 trigrams
        # (a,b,c), (b,c,a), (c,a,b), (a,b,c), (b,c,a), (c,a,b)
        # unique: (a,b,c), (b,c,a), (c,a,b) → 3 unique
        # (6-3)/6 = 0.5
        ratio = QualitySignalCollector._calc_repetition(text)
        assert ratio == pytest.approx(0.5)


# ── 信号 3: garbled_text_ratio ──


class TestGarbledTextRatio:
    def test_clean_text(self):
        assert QualitySignalCollector._calc_garbled("hello world") == 0.0

    def test_text_with_newlines(self):
        # \n \t \r 不算乱码
        assert QualitySignalCollector._calc_garbled("line1\nline2\ttab\r") == 0.0

    def test_text_with_control_chars(self):
        text = "hello\x00world\x01"
        ratio = QualitySignalCollector._calc_garbled(text)
        assert ratio == pytest.approx(2 / len(text))

    def test_private_use_area(self):
        text = "normal\ue000garbled\uf8ff"
        ratio = QualitySignalCollector._calc_garbled(text)
        assert ratio == pytest.approx(2 / len(text))

    def test_empty_string(self):
        assert QualitySignalCollector._calc_garbled("") == 0.0

    def test_all_garbled(self):
        text = "\x00\x01\x02"
        assert QualitySignalCollector._calc_garbled(text) == 1.0


# ── 信号 4: refusal_detected ──


class TestRefusalDetected:
    def test_chinese_refusal(self):
        assert (
            QualitySignalCollector._check_refusal("作为一个AI，我无法回答这个问题") == 1
        )

    def test_english_refusal(self):
        assert QualitySignalCollector._check_refusal("I cannot help with that") == 1

    def test_sorry_cant(self):
        assert QualitySignalCollector._check_refusal("Sorry, I can't do that") == 1

    def test_no_refusal(self):
        assert QualitySignalCollector._check_refusal("The answer is 42") == 0

    def test_case_insensitive(self):
        assert QualitySignalCollector._check_refusal("I CANNOT do this") == 1

    def test_empty_string(self):
        assert QualitySignalCollector._check_refusal("") == 0

    def test_im_unable(self):
        assert QualitySignalCollector._check_refusal("I'm unable to assist") == 1

    def test_i_am_unable(self):
        assert QualitySignalCollector._check_refusal("I am unable to help") == 1


# ── 信号 5: language_consistency ──


class TestLanguageConsistency:
    def test_pure_chinese(self):
        assert (
            QualitySignalCollector._calc_language_consistency("这是一段纯中文文本")
            == 1.0
        )

    def test_pure_english(self):
        assert (
            QualitySignalCollector._calc_language_consistency(
                "This is pure English text"
            )
            == 1.0
        )

    def test_mixed_dominant_chinese(self):
        # 大量中文 + 少量英文，比例 >= 0.1 → 1.0
        # "这是一个很长的中文文本" = 11 个 CJK 字符，重复 10 次 = 110 个 CJK
        # 需要 >= 11 个英文单词才能 ratio >= 0.1
        text = (
            "这是一个很长的中文文本" * 10
            + " "
            + " ".join([f"word{i}" for i in range(12)])
        )
        assert QualitySignalCollector._calc_language_consistency(text) == 1.0

    def test_heavily_mixed(self):
        # 大量中文 + 大量英文，比例 > 0.1 → 1.0
        text = "Hello 你好 World 世界"
        result = QualitySignalCollector._calc_language_consistency(text)
        # 2 CJK chars vs 2 EN words → ratio = 1.0 → not < 0.1
        assert result == 1.0

    def test_extreme_imbalance(self):
        # 极端不平衡：1 个中文 vs 100 个英文单词
        text = "中 " + "word " * 100
        result = QualitySignalCollector._calc_language_consistency(text)
        # 1 CJK / 100 EN words = 0.01 < 0.1 → 返回 0.01
        assert result == pytest.approx(0.01)

    def test_empty_string(self):
        assert QualitySignalCollector._calc_language_consistency("") == 1.0

    def test_only_numbers(self):
        assert QualitySignalCollector._calc_language_consistency("12345 67890") == 1.0


# ── 信号 6: output_length_zscore ──


class TestOutputLengthZscore:
    def test_no_history(self):
        db = _test_db()
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(
            collector._calc_length_zscore("short", "reasoning", _make_task())
        )
        assert result == 0.0
        db.close()

    def test_with_history(self):
        db = _test_db()
        _insert_history(db, output_lengths=[100, 110, 90])
        collector = QualitySignalCollector(db, "test/model")
        # mean=100, pstdev≈7.07, len("short")=5 → z = (5-100)/7.07
        result = asyncio.run(
            collector._calc_length_zscore("short", "reasoning", _make_task())
        )
        assert result < -10  # 远低于均值
        db.close()

    def test_only_one_history(self):
        """少于 2 条历史记录时返回 0.0。"""
        db = _test_db()
        _insert_history(db, output_lengths=[100])
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(
            collector._calc_length_zscore("x" * 100, "reasoning", _make_task())
        )
        assert result == 0.0
        db.close()


# ── 信号 7: thinking_ratio ──


class TestThinkingRatio:
    def test_normal_ratio(self):
        metrics = {"reasoning_tokens": 300, "completion_tokens": 500}
        assert QualitySignalCollector._calc_thinking_ratio(metrics) == pytest.approx(
            0.6
        )

    def test_zero_completion(self):
        metrics = {"reasoning_tokens": 100, "completion_tokens": 0}
        assert QualitySignalCollector._calc_thinking_ratio(metrics) == 0.0

    def test_clamped_above_one(self):
        metrics = {"reasoning_tokens": 200, "completion_tokens": 100}
        assert QualitySignalCollector._calc_thinking_ratio(metrics) == 1.0

    def test_zero_reasoning(self):
        metrics = {"reasoning_tokens": 0, "completion_tokens": 100}
        assert QualitySignalCollector._calc_thinking_ratio(metrics) == 0.0

    def test_missing_keys(self):
        metrics = {}
        # completion_tokens defaults to 1, reasoning_tokens defaults to 0
        assert QualitySignalCollector._calc_thinking_ratio(metrics) == 0.0


# ── 信号 8: empty_reasoning ──


class TestEmptyReasoning:
    def test_with_content(self):
        assert QualitySignalCollector._check_empty_reasoning("some reasoning", {}) == 0

    def test_none_content_no_tokens(self):
        assert (
            QualitySignalCollector._check_empty_reasoning(None, {"reasoning_tokens": 0})
            == 0
        )

    def test_empty_content_no_tokens(self):
        assert (
            QualitySignalCollector._check_empty_reasoning("  ", {"reasoning_tokens": 0})
            == 0
        )

    def test_empty_content_with_tokens(self):
        """有 reasoning_tokens 但内容为空 → 内容丢失 → 1。"""
        assert (
            QualitySignalCollector._check_empty_reasoning(
                None, {"reasoning_tokens": 100}
            )
            == 1
        )

    def test_whitespace_content_with_tokens(self):
        assert (
            QualitySignalCollector._check_empty_reasoning(
                "  ", {"reasoning_tokens": 50}
            )
            == 1
        )


# ── 信号 9: truncated (inline test) ──


class TestTruncated:
    def test_truncated_in_collect(self):
        # truncated 信号在 collect_and_save 中直接计算，这里单独验证逻辑
        finish_reason = "length"
        assert (1 if finish_reason == "length" else 0) == 1

    def test_not_truncated(self):
        finish_reason = "stop"
        assert (1 if finish_reason == "length" else 0) == 0


# ── 信号 10: token_efficiency_zscore ──


class TestTokenEfficiencyZscore:
    def test_no_history(self):
        db = _test_db()
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(
            collector._calc_token_efficiency_zscore({"prompt_tokens": 50}, _make_task())
        )
        assert result == 0.0
        db.close()

    def test_with_history(self):
        db = _test_db()
        _insert_history(db, prompt_tokens=[100, 100, 100])
        collector = QualitySignalCollector(db, "test/model")
        # mean=100, std=0 → 返回 0.0
        result = asyncio.run(
            collector._calc_token_efficiency_zscore(
                {"prompt_tokens": 200}, _make_task()
            )
        )
        assert result == 0.0
        db.close()

    def test_with_varying_history(self):
        db = _test_db()
        _insert_history(db, prompt_tokens=[80, 100, 120])
        collector = QualitySignalCollector(db, "test/model")
        # mean=100, pstdev≈14.14
        result = asyncio.run(
            collector._calc_token_efficiency_zscore(
                {"prompt_tokens": 100}, _make_task()
            )
        )
        assert abs(result) < 0.01
        db.close()


# ── 信号 11: tps_zscore ──


class TestTpsZscore:
    def test_no_history(self):
        db = _test_db()
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(collector._calc_tps_zscore({"tokens_per_second": 50.0}))
        assert result == 0.0
        db.close()

    def test_with_history(self):
        db = _test_db()
        _insert_history(db, tps_values=[40.0, 50.0, 60.0])
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(collector._calc_tps_zscore({"tokens_per_second": 50.0}))
        # mean=50, pstdev≈7.07, (50-50)/7.07 = 0
        assert abs(result) < 0.01
        db.close()

    def test_anomalous_tps(self):
        db = _test_db()
        _insert_history(db, tps_values=[50.0, 50.0, 50.0, 50.0])
        collector = QualitySignalCollector(db, "test/model")
        # mean=50, std=0 → 返回 0.0
        result = asyncio.run(collector._calc_tps_zscore({"tokens_per_second": 10.0}))
        assert result == 0.0
        db.close()


# ── 信号 12: ttft_zscore ──


class TestTtftZscore:
    def test_no_history(self):
        db = _test_db()
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(collector._calc_ttft_zscore({"ttft_content": 0.5}))
        assert result == 0.0
        db.close()

    def test_with_history(self):
        db = _test_db()
        _insert_history(db, ttft_values=[0.4, 0.5, 0.6])
        collector = QualitySignalCollector(db, "test/model")
        result = asyncio.run(collector._calc_ttft_zscore({"ttft_content": 0.5}))
        # mean=0.5, pstdev≈0.07, (0.5-0.5)/0.07 ≈ 0
        assert abs(result) < 0.01
        db.close()


# ── 信号 13: answer_entropy ──


class TestAnswerEntropy:
    def test_always_zero(self):
        """answer_entropy 由 StabilityAnalyzer 批次级计算，此处始终为 0.0。"""
        db = _test_db()
        collector = QualitySignalCollector(db, "test/model")
        # 通过 collect_and_save 验证
        now = datetime.now()
        run = EvalRun(
            run_id="run-entropy",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)
        result = EvalResult(
            result_id="res-entropy",
            run_id="run-entropy",
            task_id="t1",
            task_content="test",
            model_output="output",
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        signals = asyncio.run(
            collector.collect_and_save(
                result_id="res-entropy",
                raw_output="output",
                reasoning_content="thinking...",
                gen_metrics={"completion_tokens": 100, "reasoning_tokens": 10},
                finish_reason="stop",
                task=_make_task(),
                dimension="reasoning",
            )
        )
        assert signals["answer_entropy"] == 0.0
        db.close()


# ── collect_and_save 集成测试 ──


class TestCollectAndSave:
    def test_full_flow(self):
        db = _test_db()
        now = datetime.now()
        run = EvalRun(
            run_id="run-full",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)
        result = EvalResult(
            result_id="res-full",
            run_id="run-full",
            task_id="t1",
            task_content="1+1",
            model_output=r"答案是 \boxed{2}",
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        collector = QualitySignalCollector(db, "test/model")
        signals = asyncio.run(
            collector.collect_and_save(
                result_id="res-full",
                raw_output=r"答案是 \boxed{2}",
                reasoning_content="Let me calculate 1+1",
                gen_metrics={
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "reasoning_tokens": 30,
                    "tokens_per_second": 50.0,
                    "ttft_content": 0.5,
                },
                finish_reason="stop",
                task=_make_task(),
                dimension="reasoning",
            )
        )

        assert signals["format_compliance"] == 1.0
        assert signals["repetition_ratio"] == 0.0
        assert signals["garbled_text_ratio"] == 0.0
        assert signals["refusal_detected"] == 0
        assert signals["thinking_ratio"] == pytest.approx(0.3)
        assert signals["empty_reasoning"] == 0
        assert signals["truncated"] == 0
        assert signals["answer_entropy"] == 0.0
        assert signals["raw_output_length"] > 0

        # 验证已写入 DB
        rows = db._get_quality_signals_for_run("run-full")
        assert len(rows) == 1
        assert rows[0]["format_compliance"] == 1.0
        assert rows[0]["thinking_ratio"] == pytest.approx(0.3)
        db.close()

    def test_truncated_output(self):
        db = _test_db()
        now = datetime.now()
        run = EvalRun(
            run_id="run-trunc",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)
        result = EvalResult(
            result_id="res-trunc",
            run_id="run-trunc",
            task_id="t1",
            task_content="test",
            model_output="truncated output",
            functional_score=0.0,
            quality_score=0.0,
            final_score=0.0,
            passed=False,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        collector = QualitySignalCollector(db, "test/model")
        signals = asyncio.run(
            collector.collect_and_save(
                result_id="res-trunc",
                raw_output="truncated output",
                reasoning_content=None,
                gen_metrics={"completion_tokens": 0, "reasoning_tokens": 0},
                finish_reason="length",
                task=_make_task(),
                dimension="reasoning",
            )
        )
        assert signals["truncated"] == 1
        db.close()

    def test_raw_output_length(self):
        db = _test_db()
        now = datetime.now()
        run = EvalRun(
            run_id="run-len",
            model="test/model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)
        result = EvalResult(
            result_id="res-len",
            run_id="run-len",
            task_id="t1",
            task_content="test",
            model_output="hello",
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        collector = QualitySignalCollector(db, "test/model")
        raw = "Hello World, this is a test output"
        signals = asyncio.run(
            collector.collect_and_save(
                result_id="res-len",
                raw_output=raw,
                reasoning_content=None,
                gen_metrics={},
                finish_reason="stop",
                task=_make_task(),
                dimension="reasoning",
            )
        )
        assert signals["raw_output_length"] == len(raw)
        db.close()
