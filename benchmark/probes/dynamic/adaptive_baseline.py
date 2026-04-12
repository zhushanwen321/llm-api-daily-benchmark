"""自适应基线管理器 - 根据历史数据动态调整评分基线."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from benchmark.core.tz import now


@dataclass
class BaselineConfig:
    """基线配置."""

    window_days: int = 30  # 历史数据窗口（天）
    min_samples: int = 10  # 最小样本数
    outlier_std_threshold: float = 3.0  # 异常值标准差阈值
    adjustment_factor: float = 0.1  # 基线调整因子
    enable_trend_detection: bool = True  # 启用趋势检测


@dataclass
class ScoreBaseline:
    """评分基线."""

    metric_name: str
    expected_mean: float
    expected_std: float
    min_acceptable: float
    max_acceptable: float
    sample_count: int
    last_updated: datetime
    trend: str = "stable"  # stable, increasing, decreasing
    confidence: float = 0.0  # 置信度 0-1


@dataclass
class AnomalyRecord:
    """异常记录."""

    metric_name: str
    observed_value: float
    expected_range: tuple[float, float]
    deviation_score: float  # 偏离程度
    timestamp: datetime
    severity: str  # low, medium, high


class HistoricalDataAnalyzer:
    """历史数据分析器."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()

    def extract_metric_series(
        self,
        historical_data: list[dict[str, Any]],
        metric_path: str,
    ) -> list[float]:
        """从历史数据中提取指标时间序列.

        Args:
            historical_data: 历史数据列表
            metric_path: 指标路径，如 "scores.functional"

        Returns:
            指标值列表
        """
        values = []
        for record in historical_data:
            value = self._get_nested_value(record, metric_path)
            if value is not None and isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """获取嵌套字典值."""
        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def calculate_statistics(self, values: list[float]) -> dict[str, float]:
        """计算统计指标."""
        if len(values) < self.config.min_samples:
            return {
                "mean": sum(values) / len(values) if values else 0.0,
                "std": 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "median": statistics.median(values) if values else 0.0,
            }

        # 去除异常值
        filtered = self._remove_outliers(values)

        return {
            "mean": statistics.mean(filtered),
            "std": statistics.stdev(filtered) if len(filtered) > 1 else 0.0,
            "min": min(filtered),
            "max": max(filtered),
            "median": statistics.median(filtered),
        }

    def _remove_outliers(self, values: list[float]) -> list[float]:
        """使用IQR方法去除异常值."""
        if len(values) < 4:
            return values

        sorted_values = sorted(values)
        q1_idx = len(sorted_values) // 4
        q3_idx = q1_idx * 3
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return [v for v in values if lower_bound <= v <= upper_bound]

    def detect_trend(self, values: list[float]) -> str:
        """检测趋势."""
        if len(values) < 5:
            return "stable"

        # 简单线性回归斜率
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # 根据斜率判断趋势
        avg_value = sum(abs(v) for v in values) / len(values) if values else 1
        normalized_slope = slope / avg_value if avg_value > 0 else 0

        if normalized_slope > 0.05:
            return "increasing"
        elif normalized_slope < -0.05:
            return "decreasing"
        return "stable"


class AdaptiveBaselineManager:
    """自适应基线管理器."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self._analyzer = HistoricalDataAnalyzer(self.config)
        self._baselines: dict[str, ScoreBaseline] = {}
        self._anomaly_history: list[AnomalyRecord] = []

    def compute_baseline(
        self,
        metric_name: str,
        historical_data: list[dict[str, Any]],
        metric_path: str | None = None,
    ) -> ScoreBaseline:
        """计算指标基线.

        Args:
            metric_name: 指标名称
            historical_data: 历史数据
            metric_path: 数据中的指标路径，默认为metric_name

        Returns:
            计算得到的基线
        """
        path = metric_path or metric_name
        values = self._analyzer.extract_metric_series(historical_data, path)

        if not values:
            # 无历史数据时使用默认值
            baseline = ScoreBaseline(
                metric_name=metric_name,
                expected_mean=0.0,
                expected_std=0.0,
                min_acceptable=0.0,
                max_acceptable=100.0,
                sample_count=0,
                last_updated=now(),
                confidence=0.0,
            )
        else:
            stats = self._analyzer.calculate_statistics(values)
            trend = self._analyzer.detect_trend(values)

            # 计算可接受范围（均值 ± 3*标准差）
            mean = stats["mean"]
            std = stats["std"]
            min_acceptable = max(0, mean - self.config.outlier_std_threshold * std)
            max_acceptable = min(100, mean + self.config.outlier_std_threshold * std)

            # 置信度基于样本数量
            confidence = min(len(values) / self.config.min_samples, 1.0)

            baseline = ScoreBaseline(
                metric_name=metric_name,
                expected_mean=mean,
                expected_std=std,
                min_acceptable=min_acceptable,
                max_acceptable=max_acceptable,
                sample_count=len(values),
                last_updated=now(),
                trend=trend,
                confidence=confidence,
            )

        self._baselines[metric_name] = baseline
        return baseline

    def update_baseline(
        self,
        metric_name: str,
        new_value: float,
        timestamp: datetime | None = None,
    ) -> ScoreBaseline:
        """使用新观测值更新基线.

        使用指数移动平均法平滑更新基线.
        """
        ts = timestamp or now()

        if metric_name not in self._baselines:
            # 首次创建基线
            baseline = ScoreBaseline(
                metric_name=metric_name,
                expected_mean=new_value,
                expected_std=0.0,
                min_acceptable=new_value * 0.8,
                max_acceptable=new_value * 1.2,
                sample_count=1,
                last_updated=ts,
            )
        else:
            old = self._baselines[metric_name]
            alpha = self.config.adjustment_factor

            # 指数移动平均更新均值
            new_mean = (1 - alpha) * old.expected_mean + alpha * new_value

            # 更新标准差
            if old.sample_count > 1:
                new_std = (
                    (1 - alpha) * old.expected_std**2
                    + alpha * (new_value - new_mean) ** 2
                ) ** 0.5
            else:
                new_std = abs(new_value - new_mean)

            # 更新范围
            threshold = self.config.outlier_std_threshold
            new_min = max(0, new_mean - threshold * new_std)
            new_max = min(100, new_mean + threshold * new_std)

            baseline = ScoreBaseline(
                metric_name=metric_name,
                expected_mean=new_mean,
                expected_std=new_std,
                min_acceptable=new_min,
                max_acceptable=new_max,
                sample_count=old.sample_count + 1,
                last_updated=ts,
                confidence=min(old.confidence + 0.05, 1.0),
            )

        self._baselines[metric_name] = baseline
        return baseline

    def detect_anomaly(
        self,
        metric_name: str,
        observed_value: float,
        timestamp: datetime | None = None,
    ) -> AnomalyRecord | None:
        """检测异常值."""
        if metric_name not in self._baselines:
            return None

        baseline = self._baselines[metric_name]
        ts = timestamp or now()

        # 检查是否在可接受范围内
        if baseline.min_acceptable <= observed_value <= baseline.max_acceptable:
            return None

        # 计算偏离程度
        if baseline.expected_std > 0:
            deviation = (
                observed_value - baseline.expected_mean
            ) / baseline.expected_std
        else:
            deviation = 0.0 if observed_value == baseline.expected_mean else 10.0

        # 确定严重程度
        abs_deviation = abs(deviation)
        if abs_deviation < 4:
            severity = "low"
        elif abs_deviation < 6:
            severity = "medium"
        else:
            severity = "high"

        anomaly = AnomalyRecord(
            metric_name=metric_name,
            observed_value=observed_value,
            expected_range=(baseline.min_acceptable, baseline.max_acceptable),
            deviation_score=deviation,
            timestamp=ts,
            severity=severity,
        )

        self._anomaly_history.append(anomaly)
        return anomaly

    def get_baseline(self, metric_name: str) -> ScoreBaseline | None:
        """获取指定指标的基线."""
        return self._baselines.get(metric_name)

    def get_all_baselines(self) -> dict[str, ScoreBaseline]:
        """获取所有基线."""
        return self._baselines.copy()

    def get_anomaly_history(
        self,
        metric_name: str | None = None,
        since: datetime | None = None,
        severity: str | None = None,
    ) -> list[AnomalyRecord]:
        """获取异常历史."""
        records = self._anomaly_history

        if metric_name:
            records = [r for r in records if r.metric_name == metric_name]
        if since:
            records = [r for r in records if r.timestamp >= since]
        if severity:
            records = [r for r in records if r.severity == severity]

        return records

    def get_health_report(self) -> dict[str, Any]:
        """生成健康报告."""
        total_metrics = len(self._baselines)
        low_confidence = sum(1 for b in self._baselines.values() if b.confidence < 0.5)

        recent_anomalies = [
            a for a in self._anomaly_history if a.timestamp > now() - timedelta(days=7)
        ]

        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for a in recent_anomalies:
            severity_counts[a.severity] += 1

        return {
            "total_metrics": total_metrics,
            "low_confidence_metrics": low_confidence,
            "recent_anomalies": len(recent_anomalies),
            "severity_breakdown": severity_counts,
            "metrics_with_trends": {
                name: b.trend
                for name, b in self._baselines.items()
                if b.trend != "stable"
            },
        }

    def export_baselines(self) -> dict[str, Any]:
        """导出基线配置."""
        return {
            name: {
                "expected_mean": b.expected_mean,
                "expected_std": b.expected_std,
                "min_acceptable": b.min_acceptable,
                "max_acceptable": b.max_acceptable,
                "sample_count": b.sample_count,
                "last_updated": b.last_updated.isoformat(),
                "trend": b.trend,
                "confidence": b.confidence,
            }
            for name, b in self._baselines.items()
        }

    def import_baselines(self, data: dict[str, Any]) -> None:
        """导入基线配置."""
        for name, b_data in data.items():
            try:
                last_updated_str = b_data.get("last_updated", now().isoformat())
                last_updated = datetime.fromisoformat(last_updated_str)
            except (ValueError, TypeError):
                last_updated = now()

            self._baselines[name] = ScoreBaseline(
                metric_name=name,
                expected_mean=b_data.get("expected_mean", 0.0),
                expected_std=b_data.get("expected_std", 0.0),
                min_acceptable=b_data.get("min_acceptable", 0.0),
                max_acceptable=b_data.get("max_acceptable", 100.0),
                sample_count=b_data.get("sample_count", 0),
                last_updated=last_updated,
                trend=b_data.get("trend", "stable"),
                confidence=b_data.get("confidence", 0.0),
            )
