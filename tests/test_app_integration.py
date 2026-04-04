"""Streamlit 应用集成测试."""

import pytest


def test_app_imports_statistics_module():
    """验证 app 可以导入统计模块."""
    from benchmark.visualization import app
    from benchmark.core import statistics
    assert statistics is not None


def test_app_imports_trends_component():
    """验证 app 可以导入趋势图组件."""
    from benchmark.visualization import app
    from benchmark.visualization.components import trends
    assert trends is not None
