"""测试 analysis 数据模型。"""

from dataclasses import asdict
from datetime import datetime

from benchmark.analysis.models import AnomalyDetail, ChangePoint, StabilityReport


# ── dataclass 基本测试 ──


class TestAnomalyDetail:
    def test_creation(self):
        a = AnomalyDetail(
            signal_name="repetition_ratio",
            current_value=0.5,
            baseline_mean=0.1,
            baseline_std=0.05,
            z_score=8.0,
        )
        assert a.signal_name == "repetition_ratio"
        assert a.z_score == 8.0

    def test_asdict(self):
        a = AnomalyDetail("x", 1.0, 0.5, 0.1, 5.0)
        d = asdict(a)
        assert d["signal_name"] == "x"
        assert d["z_score"] == 5.0


class TestChangePoint:
    def test_creation(self):
        cp = ChangePoint(
            signal_name="tps_zscore",
            detected_at=datetime(2026, 1, 1),
            direction="decrease",
            magnitude=-3.2,
        )
        assert cp.direction == "decrease"
        assert cp.magnitude == -3.2

    def test_direction_literal(self):
        """direction 只接受 increase / decrease。"""
        ChangePoint("x", datetime.now(), "increase", 1.0)
        ChangePoint("x", datetime.now(), "decrease", -1.0)


class TestStabilityReport:
    def test_minimal_creation(self):
        r = StabilityReport(
            model="gpt-4o",
            run_id="run-001",
            overall_status="stable",
        )
        assert r.anomalies == []
        assert r.change_points == []
        assert r.stat_tests == []
        assert r.summary == ""
        assert isinstance(r.created_at, datetime)

    def test_full_creation(self):
        anomaly = AnomalyDetail("rep", 0.5, 0.1, 0.05, 8.0)
        cp = ChangePoint("tps", datetime.now(), "decrease", -3.0)
        r = StabilityReport(
            model="gpt-4o",
            run_id="run-002",
            overall_status="degraded",
            anomalies=[anomaly],
            change_points=[cp],
            stat_tests=[{"test": "ks", "p_value": 0.01}],
            summary="模型质量下降",
        )
        assert len(r.anomalies) == 1
        assert len(r.change_points) == 1
        assert r.overall_status == "degraded"
