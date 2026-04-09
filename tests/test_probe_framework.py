"""测试新的探针框架."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from benchmark.probes.fast.capability_probe import CapabilityProbe
from benchmark.models.schemas import TaskDefinition, EvalResult


@pytest.mark.asyncio
async def test_capability_probe_load():
    """测试探针加载."""
    probe = CapabilityProbe()

    # 验证频率
    assert probe.frequency == "fast"


def test_capability_probe_features():
    """测试特征提取."""
    from datetime import datetime

    probe = CapabilityProbe()

    # 创建mock结果
    result = EvalResult(
        result_id="test_001",
        run_id="run_001",
        task_id="task_001",
        task_content="test prompt",
        model_output="test output",
        functional_score=100.0,
        final_score=100.0,
        passed=True,
        execution_time=1.5,
        created_at=datetime.now(),
    )

    features = probe.extract_features(result)

    assert features["passed"] is True
    assert features["score"] == 100.0
    assert features["execution_time"] == 1.5
    assert features["output_length"] == 11  # "test output"
