# tests/test_trends.py
"""趋势图组件测试 - 适配 FileRepository."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from benchmark.repository import FileRepository
from benchmark.visualization.components.trends import (
    get_trend_data,
    create_trend_figure,
)


def test_get_trend_data_returns_correct_structure():
    """获取趋势数据应返回正确的结构."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        # 创建 FileRepository 并生成测试数据
        repo = FileRepository(data_root=tmpdir)

        # 创建测试 runs
        now = datetime.now()
        for i in range(3):
            benchmark_id = repo.create_benchmark_run(
                model="test-model",
                dimension="reasoning",
                dataset="test",
                questions=[f"q{i}"],
            )
            # 保存结果
            repo.save_question_result(
                benchmark_id=benchmark_id,
                question_id=f"q{i}",
                answer_data={
                    "result_id": f"r{i}",
                    "task_content": "test",
                    "model_output": "answer",
                    "model_answer": "answer",
                    "expected_output": "answer",
                    "functional_score": 80.0 + i * 5,
                    "quality_score": 80.0 + i * 5,
                    "final_score": 80.0 + i * 5,
                    "passed": True,
                    "execution_time": 1.0,
                },
            )
            # 保存评分以完成 run
            repo.save_question_scoring(
                benchmark_id=benchmark_id,
                question_id=f"q{i}",
                scoring_data={
                    "task_id": f"q{i}",
                    "functional_score": 80.0 + i * 5,
                    "quality_score": 80.0 + i * 5,
                    "final_score": 80.0 + i * 5,
                    "passed": True,
                    "details": {},
                },
            )

        # 使用 FileRepository 获取趋势数据
        data = get_trend_data(repo, "test-model", "reasoning", 30)
        assert "dates" in data
        assert "scores" in data
        assert len(data["dates"]) == len(data["scores"])


def test_create_trend_figure():
    """创建趋势图应返回 matplotlib Figure."""
    data = {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "scores": [80.0, 85.0, 82.0],
    }
    fig = create_trend_figure(data, title="Test Trend")
    assert fig is not None
