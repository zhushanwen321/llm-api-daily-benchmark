"""测试自适应基线管理器."""

import unittest
from datetime import datetime, timedelta

from benchmark.probes.dynamic.adaptive_baseline import (
    AdaptiveBaselineManager,
    AnomalyRecord,
    BaselineConfig,
    HistoricalDataAnalyzer,
    ScoreBaseline,
)


class TestBaselineConfig(unittest.TestCase):
    """测试基线配置."""

    def test_default_config(self):
        config = BaselineConfig()
        self.assertEqual(config.window_days, 30)
        self.assertEqual(config.min_samples, 10)
        self.assertEqual(config.outlier_std_threshold, 3.0)
        self.assertEqual(config.adjustment_factor, 0.1)
        self.assertTrue(config.enable_trend_detection)

    def test_custom_config(self):
        config = BaselineConfig(
            window_days=7,
            min_samples=5,
            adjustment_factor=0.2,
        )
        self.assertEqual(config.window_days, 7)
        self.assertEqual(config.min_samples, 5)
        self.assertEqual(config.adjustment_factor, 0.2)


class TestHistoricalDataAnalyzer(unittest.TestCase):
    """测试历史数据分析器."""

    def setUp(self):
        self.analyzer = HistoricalDataAnalyzer()

    def test_extract_metric_series(self):
        data = [
            {"scores": {"functional": 80, "quality": 90}},
            {"scores": {"functional": 85, "quality": 88}},
            {"scores": {"functional": 82, "quality": 92}},
        ]

        values = self.analyzer.extract_metric_series(data, "scores.functional")
        self.assertEqual(values, [80.0, 85.0, 82.0])

    def test_extract_metric_series_missing_path(self):
        data = [{"scores": {"quality": 90}}]
        values = self.analyzer.extract_metric_series(data, "scores.functional")
        self.assertEqual(values, [])

    def test_calculate_statistics(self):
        values = [80, 85, 82, 88, 90, 83, 87]
        stats = self.analyzer.calculate_statistics(values)

        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("median", stats)

    def test_calculate_statistics_insufficient_samples(self):
        values = [80]
        stats = self.analyzer.calculate_statistics(values)
        self.assertEqual(stats["mean"], 80.0)

    def test_detect_trend_increasing(self):
        values = [50, 60, 75, 85, 95, 110]  # 更明显的上升趋势
        trend = self.analyzer.detect_trend(values)
        self.assertEqual(trend, "increasing")

    def test_detect_trend_decreasing(self):
        values = [110, 95, 85, 75, 60, 50]  # 更明显的下降趋势
        trend = self.analyzer.detect_trend(values)
        self.assertEqual(trend, "decreasing")

    def test_detect_trend_stable(self):
        values = [85, 86, 85, 84, 85, 86]  # 稳定波动
        trend = self.analyzer.detect_trend(values)
        self.assertEqual(trend, "stable")

    def test_detect_trend_insufficient_data(self):
        values = [80, 85]
        trend = self.analyzer.detect_trend(values)
        self.assertEqual(trend, "stable")


class TestAdaptiveBaselineManager(unittest.TestCase):
    """测试自适应基线管理器."""

    def setUp(self):
        self.manager = AdaptiveBaselineManager()

    def test_compute_baseline_with_data(self):
        historical_data = [
            {"scores": {"functional": 80}},
            {"scores": {"functional": 85}},
            {"scores": {"functional": 82}},
            {"scores": {"functional": 88}},
            {"scores": {"functional": 90}},
            {"scores": {"functional": 83}},
            {"scores": {"functional": 87}},
            {"scores": {"functional": 86}},
            {"scores": {"functional": 84}},
            {"scores": {"functional": 89}},
        ]

        baseline = self.manager.compute_baseline(
            "functional_score",
            historical_data,
            "scores.functional",
        )

        self.assertIsInstance(baseline, ScoreBaseline)
        self.assertEqual(baseline.metric_name, "functional_score")
        self.assertGreater(baseline.sample_count, 0)
        self.assertGreaterEqual(baseline.confidence, 0.0)
        self.assertLessEqual(baseline.confidence, 1.0)

    def test_compute_baseline_without_data(self):
        baseline = self.manager.compute_baseline("empty_metric", [], "scores.empty")

        self.assertEqual(baseline.sample_count, 0)
        self.assertEqual(baseline.confidence, 0.0)
        self.assertEqual(baseline.expected_mean, 0.0)

    def test_update_baseline_new_metric(self):
        baseline = self.manager.update_baseline("new_metric", 85.0)

        self.assertEqual(baseline.metric_name, "new_metric")
        self.assertEqual(baseline.expected_mean, 85.0)
        self.assertEqual(baseline.sample_count, 1)

    def test_update_baseline_existing_metric(self):
        # 先创建基线
        self.manager.update_baseline("metric1", 80.0)
        baseline1 = self.manager.get_baseline("metric1")

        # 更新基线
        baseline2 = self.manager.update_baseline("metric1", 90.0)

        # 新值应该在80和90之间（指数移动平均）
        self.assertGreater(baseline2.expected_mean, baseline1.expected_mean)
        self.assertLess(baseline2.expected_mean, 90.0)
        self.assertEqual(baseline2.sample_count, 2)

    def test_detect_anomaly_within_range(self):
        for i in range(10):
            self.manager.update_baseline("stable_metric", 85.0 + (i % 3 - 1) * 2)

        anomaly = self.manager.detect_anomaly("stable_metric", 86.0)
        self.assertIsNone(anomaly)

    def test_detect_anomaly_outside_range(self):
        # 先建立基线
        for _ in range(10):
            self.manager.update_baseline("variable_metric", 85.0)

        # 检测明显异常的值
        anomaly = self.manager.detect_anomaly("variable_metric", 50.0)

        self.assertIsNotNone(anomaly)
        self.assertIsInstance(anomaly, AnomalyRecord)
        self.assertEqual(anomaly.metric_name, "variable_metric")
        self.assertEqual(anomaly.observed_value, 50.0)
        self.assertIn(anomaly.severity, ["low", "medium", "high"])

    def test_detect_anomaly_unknown_metric(self):
        anomaly = self.manager.detect_anomaly("unknown_metric", 85.0)
        self.assertIsNone(anomaly)

    def test_get_baseline(self):
        self.manager.update_baseline("test_metric", 85.0)
        baseline = self.manager.get_baseline("test_metric")

        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.metric_name, "test_metric")

    def test_get_all_baselines(self):
        self.manager.update_baseline("metric1", 80.0)
        self.manager.update_baseline("metric2", 90.0)

        all_baselines = self.manager.get_all_baselines()
        self.assertEqual(len(all_baselines), 2)
        self.assertIn("metric1", all_baselines)
        self.assertIn("metric2", all_baselines)

    def test_get_anomaly_history(self):
        # 创建基线并检测异常
        for _ in range(10):
            self.manager.update_baseline("anomaly_metric", 85.0)

        self.manager.detect_anomaly("anomaly_metric", 50.0)
        self.manager.detect_anomaly("anomaly_metric", 100.0)

        # 获取所有异常
        history = self.manager.get_anomaly_history()
        self.assertEqual(len(history), 2)

        # 按指标名筛选
        filtered = self.manager.get_anomaly_history(metric_name="anomaly_metric")
        self.assertEqual(len(filtered), 2)

        # 按严重级别筛选
        high_severity = self.manager.get_anomaly_history(severity="high")
        # 可能有高严重级别的异常

    def test_get_health_report(self):
        # 创建一些基线
        for _ in range(5):
            self.manager.update_baseline("metric1", 85.0)
            self.manager.update_baseline("metric2", 90.0)

        report = self.manager.get_health_report()

        self.assertIn("total_metrics", report)
        self.assertIn("low_confidence_metrics", report)
        self.assertIn("recent_anomalies", report)
        self.assertIn("severity_breakdown", report)
        self.assertIn("metrics_with_trends", report)

        self.assertEqual(report["total_metrics"], 2)

    def test_export_import_baselines(self):
        # 创建基线
        for _ in range(5):
            self.manager.update_baseline("export_metric", 85.0)

        # 导出
        exported = self.manager.export_baselines()
        self.assertIn("export_metric", exported)
        self.assertIn("expected_mean", exported["export_metric"])

        # 创建新的管理器并导入
        new_manager = AdaptiveBaselineManager()
        new_manager.import_baselines(exported)

        imported_baseline = new_manager.get_baseline("export_metric")
        self.assertIsNotNone(imported_baseline)
        self.assertEqual(imported_baseline.expected_mean, 85.0)


if __name__ == "__main__":
    unittest.main()
