"""测试 analysis 数据模型和 DB 持久化。"""

import asyncio
import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest

from benchmark.analysis.models import AnomalyDetail, ChangePoint, StabilityReport
from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics, EvalResult, EvalRun


def _test_db() -> Database:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = Database(db_path=Path(tmp.name))
    return db


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
        # Python dataclass 不强制 Literal，但类型检查器会报错
        # 这里只验证合法值能正常创建


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


# ── DB 持久化测试 ──


class TestQualitySignalsDB:
    def test_save_and_query_signals(self):
        db = _test_db()
        now = datetime.now()
        # 先插入 eval_run 和 eval_result 作为外键依赖
        run = EvalRun(
            run_id="run-sig-001",
            model="test-model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)

        result = EvalResult(
            result_id="res-sig-001",
            run_id="run-sig-001",
            task_id="t1",
            task_content="1+1",
            model_output="2",
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        signals = {
            "result_id": "res-sig-001",
            "format_compliance": 0.95,
            "repetition_ratio": 0.1,
            "thinking_ratio": 0.3,
            "truncated": 0,
            "raw_output_length": 500,
        }
        signal_id = db._save_quality_signals(signals)
        assert len(signal_id) == 12

        # 查询验证
        rows = db._get_quality_signals_for_run("run-sig-001")
        assert len(rows) == 1
        assert rows[0]["format_compliance"] == 0.95
        assert rows[0]["raw_output_length"] == 500
        db.close()

    def test_save_signals_default_values(self):
        """只传 result_id 时，其余字段应使用默认值。"""
        db = _test_db()
        now = datetime.now()
        run = EvalRun(
            run_id="run-default-001",
            model="test-model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)

        result = EvalResult(
            result_id="res-default-001",
            run_id="run-default-001",
            task_id="t1",
            task_content="test",
            model_output="out",
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        signal_id = db._save_quality_signals({"result_id": "res-default-001"})
        rows = db._get_quality_signals_for_run("run-default-001")
        assert len(rows) == 1
        assert rows[0]["format_compliance"] == 0
        assert rows[0]["language_consistency"] == 1.0
        assert rows[0]["refusal_detected"] == 0
        db.close()


class TestStabilityReportDB:
    def test_save_and_query_report(self):
        db = _test_db()
        report = StabilityReport(
            model="gpt-4o",
            run_id="run-rpt-001",
            overall_status="degraded",
            anomalies=[
                AnomalyDetail("rep", 0.5, 0.1, 0.05, 8.0),
            ],
            change_points=[
                ChangePoint("tps", datetime(2026, 1, 1), "decrease", -3.0),
            ],
            stat_tests=[{"test": "ks", "p_value": 0.01}],
            summary="质量下降",
        )
        report_id = db._save_stability_report(report)
        assert len(report_id) == 12

        # 查询
        rows = db._get_stability_reports(model="gpt-4o")
        assert len(rows) == 1
        assert rows[0]["overall_status"] == "degraded"
        assert rows[0]["summary"] == "质量下降"

        # 验证 JSON 序列化
        anomalies = json.loads(rows[0]["anomalies"])
        assert len(anomalies) == 1
        assert anomalies[0]["signal_name"] == "rep"
        assert anomalies[0]["z_score"] == 8.0

        change_points = json.loads(rows[0]["change_points"])
        assert len(change_points) == 1
        assert change_points[0]["direction"] == "decrease"

        stat_tests = json.loads(rows[0]["stat_tests"])
        assert stat_tests[0]["p_value"] == 0.01
        db.close()

    def test_query_all_reports(self):
        """不传 model 参数时返回全部报告。"""
        db = _test_db()
        for i in range(3):
            report = StabilityReport(
                model=f"model-{i}",
                run_id=f"run-{i}",
                overall_status="stable",
            )
            db._save_stability_report(report)

        rows = db._get_stability_reports()
        assert len(rows) == 3
        db.close()


class TestAsyncMethods:
    """验证异步方法正确代理到同步实现。"""

    def test_asave_quality_signals(self):
        db = _test_db()
        now = datetime.now()
        run = EvalRun(
            run_id="run-async-001",
            model="test-model",
            dimension="reasoning",
            dataset="gsm8k",
            started_at=now,
            status="completed",
        )
        db.create_run(run)
        result = EvalResult(
            result_id="res-async-001",
            run_id="run-async-001",
            task_id="t1",
            task_content="test",
            model_output="out",
            functional_score=1.0,
            quality_score=1.0,
            final_score=1.0,
            passed=True,
            execution_time=1.0,
            created_at=now,
        )
        db.save_result(result)

        signal_id = asyncio.get_event_loop().run_until_complete(
            db.asave_quality_signals({"result_id": "res-async-001", "format_compliance": 0.8})
        )
        assert len(signal_id) == 12
        db.close()

    def test_asave_stability_report(self):
        db = _test_db()
        report = StabilityReport(
            model="gpt-4o",
            run_id="run-async-rpt",
            overall_status="suspicious",
            summary="需要关注",
        )
        report_id = asyncio.get_event_loop().run_until_complete(
            db.asave_stability_report(report)
        )
        assert len(report_id) == 12
        db.close()

    def test_aget_stability_reports(self):
        db = _test_db()
        report = StabilityReport(
            model="async-model",
            run_id="run-aget",
            overall_status="stable",
        )
        db._save_stability_report(report)

        rows = asyncio.get_event_loop().run_until_complete(
            db.aget_stability_reports(model="async-model")
        )
        assert len(rows) == 1
        assert rows[0]["overall_status"] == "stable"
        db.close()
