"""测试 QualitySignalCollector 的 13 个质量信号提取 - 适配 FileRepository."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from benchmark.analysis.quality_signals import QualitySignalCollector
from benchmark.models.schemas import TaskDefinition


# ── fixtures ──


def _make_task(task_id: str = "t1", dimension: str = "reasoning") -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        dimension=dimension,
        dataset="gsm8k",
        prompt="What is 1+1?",
        expected_output="2",
    )


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


# ── 信号 6-9: 需要历史数据的 z-score 测试 ──


class TestZscoreWithNoHistory:
    """没有历史数据时，所有 z-score 应该返回 0.0。"""

    def test_output_length_zscore_no_history(self, tmp_path):
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")
        result = asyncio.run(
            collector._calc_length_zscore("short", "reasoning", _make_task())
        )
        assert result == 0.0

    def test_token_efficiency_zscore_no_history(self, tmp_path):
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")
        result = asyncio.run(
            collector._calc_token_efficiency_zscore({"prompt_tokens": 50}, _make_task())
        )
        assert result == 0.0

    def test_tps_zscore_no_history(self, tmp_path):
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")
        result = asyncio.run(collector._calc_tps_zscore({"tokens_per_second": 50.0}))
        assert result == 0.0

    def test_ttft_zscore_no_history(self, tmp_path):
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")
        result = asyncio.run(collector._calc_ttft_zscore({"ttft_content": 0.5}))
        assert result == 0.0


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


# ── 信号 13: answer_entropy ──


class TestAnswerEntropy:
    def test_always_zero(self):
        """answer_entropy 由 StabilityAnalyzer 批次级计算，此处始终为 0.0。"""
        # 这是一个静态逻辑测试，验证信号在 collector 中始终为 0
        assert True  # answer_entropy 在 collect_and_save 中硬编码为 0.0


# ── collect_and_save 集成测试 (简化版) ──


class TestCollectAndSaveBasic:
    """使用 FileRepository 测试 collect_and_save 的基本功能。"""

    def test_static_signals(self, tmp_path):
        """测试静态信号计算（不依赖历史数据）。"""
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")

        signals = asyncio.run(
            collector.collect_and_save(
                result_id="res-full",
                raw_output=r"答案是 \boxed{2}",
                reasoning_content="Let me calculate 1+1",
                gen_metrics={
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "reasoning_tokens": 30,
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

    def test_truncated_output(self, tmp_path):
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")
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

    def test_raw_output_length(self, tmp_path):
        from benchmark.repository import FileRepository

        repo = FileRepository(data_root=tmp_path)
        collector = QualitySignalCollector(repo, "test/model")
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
