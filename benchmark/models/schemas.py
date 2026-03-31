"""LLM Benchmark 数据模型定义。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TaskDefinition(BaseModel):
    """评测题目定义。"""

    task_id: str
    dimension: str  # reasoning, backend-dev
    dataset: str  # gsm8k, bigcodebench
    prompt: str
    expected_output: str
    test_cases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """单题评分结果。"""

    score: float  # 0-100
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class EvalRun(BaseModel):
    """一次评测运行的记录。"""

    run_id: str
    model: str
    dimension: str
    dataset: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str  # running, completed, failed
    config_snapshot: str = "{}"


class EvalResult(BaseModel):
    """单题评测结果。"""

    result_id: str
    run_id: str
    task_id: str
    task_content: str
    model_output: str
    functional_score: float
    quality_score: float = 0.0
    final_score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    created_at: datetime
