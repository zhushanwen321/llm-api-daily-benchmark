"""质量信号与稳定性分析的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class AnomalyDetail:
    """单个异常信号的具体数值。"""

    signal_name: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float


@dataclass
class ChangePoint:
    """检测到的突变点。"""

    signal_name: str
    detected_at: datetime
    direction: Literal["increase", "decrease"]
    magnitude: float


@dataclass
class StabilityReport:
    """一次稳定性分析的完整报告。"""

    model: str
    run_id: str
    overall_status: Literal["stable", "degraded", "suspicious"]
    anomalies: list[AnomalyDetail] = field(default_factory=list)
    change_points: list[ChangePoint] = field(default_factory=list)
    stat_tests: list[dict] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterInfo:
    """单个聚类簇的信息。"""

    cluster_id: int
    size: int
    time_range: tuple[str, str]  # (最早时间戳, 最晚时间戳)
    centroid: list[float]
    avg_score: float


@dataclass
class ClusterReport:
    """聚类分析报告。"""

    model: str
    n_clusters: int
    n_noise: int
    clusters: list[ClusterInfo] = field(default_factory=list)
    suspected_changes: list[dict] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
