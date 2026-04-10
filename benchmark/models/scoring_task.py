"""待评分任务数据模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class ScoringTaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PendingScoringTask(BaseModel):
    """待评分任务，存储在 pending_scoring_tasks 表中。"""

    model_config = ConfigDict(protected_namespaces=())

    id: int | None = None
    task_id: str
    dimension: str
    dataset: str
    prompt: str
    expected_output: str
    model_output: str
    model_answer: str
    reasoning_content: str = ""
    test_cases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scoring_dimensions: list[str] = Field(default_factory=list)
    status: ScoringTaskStatus = ScoringTaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    score_result: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    processing_started_at: datetime | None = None
    processing_finished_at: datetime | None = None
    result_id: str
    run_id: str
