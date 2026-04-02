# tests/test_trends.py
import pytest
from datetime import datetime, timedelta
from benchmark.visualization.components.trends import (
    get_trend_data,
    create_trend_figure
)

def test_get_trend_data_returns_correct_structure():
    """获取趋势数据应返回正确的结构."""
    # 使用 mock connection 进行测试
    import sqlite3
    import tempfile
    import os

    # 创建临时数据库
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        conn = sqlite3.connect(db_path)
        # 创建测试表和数据
        conn.execute("""
            CREATE TABLE eval_runs (
                run_id TEXT PRIMARY KEY,
                model TEXT,
                dimension TEXT,
                started_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE eval_results (
                result_id TEXT PRIMARY KEY,
                run_id TEXT,
                final_score REAL,
                created_at TIMESTAMP
            )
        """)

        # 插入测试数据
        from datetime import datetime, timedelta
        now = datetime.now()
        conn.execute("INSERT INTO eval_runs VALUES (?, ?, ?, ?)", ("run1", "glm-4.7", "reasoning", now))
        conn.execute("INSERT INTO eval_results VALUES (?, ?, ?, ?)", ("res1", "run1", 80.0, now))

        data = get_trend_data(conn, "glm-4.7", "reasoning", 30)
        assert "dates" in data
        assert "scores" in data
        assert len(data["dates"]) == len(data["scores"])
    finally:
        os.unlink(db_path)

def test_create_trend_figure():
    """创建趋势图应返回 matplotlib Figure."""
    data = {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "scores": [80.0, 85.0, 82.0]
    }
    fig = create_trend_figure(data, title="Test Trend")
    assert fig is not None
