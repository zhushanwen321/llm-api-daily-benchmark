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


class GenerateResponse(BaseModel):
    """LLM API 调用响应，包含文本和 token 用量。"""

    content: str
    reasoning_content: str = ""   # 推理过程（从 API reasoning_content 字段获取）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0     # 推理 token 数
    duration: float = 0.0
    tokens_per_second: float = 0.0
    ttft: float = 0.0             # 首 token 延迟（TTFT-R）
    ttft_content: float = 0.0     # 首 content token 延迟（TTFT-C）
    truncated: bool = False
    finish_reason: str = ""


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
    model_think: str = ""
    model_answer: str = ""
    functional_score: float
    quality_score: float = 0.0
    final_score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    created_at: datetime


class ApiCallMetrics(BaseModel):
    """单次 API 调用的 token 指标。"""

    result_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_content: str = ""
    duration: float = 0.0
    tokens_per_second: float = 0.0
    ttft_content: float = 0.0
    created_at: datetime
