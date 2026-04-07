import json
import math

from benchmark.core.reporter import _build_radar_svg, _extract_dimension_scores, _build_dimension_score_table, _DIMENSION_AXES


class TestExtractDimensionScores:
    def test_backend_scores(self):
        rows = [
            {
                "model": "test-model",
                "dimension": "backend-dev",
                "details": json.dumps({
                    "composite": {
                        "weights": {"test_coverage": 0.4, "performance": 0.25},
                        "scores": {"test_coverage": 90.0, "performance": 80.0},
                    },
                }),
            }
        ]
        result = _extract_dimension_scores(rows, "backend-dev", "test-model")
        assert result["test_coverage"] == 90.0
        assert result["performance"] == 80.0

    def test_missing_details(self):
        rows = [{"model": "test-model", "dimension": "backend-dev", "details": ""}]
        result = _extract_dimension_scores(rows, "backend-dev", "test-model")
        assert result == {}

    def test_no_composite_key(self):
        rows = [{"model": "test-model", "dimension": "backend-dev", "details": json.dumps({"some_key": "some_val"})}]
        result = _extract_dimension_scores(rows, "backend-dev", "test-model")
        assert result == {}

    def test_averages_across_rows(self):
        rows = [
            {"model": "m", "dimension": "backend-dev", "details": json.dumps({"composite": {"scores": {"test_coverage": 100}}})},
            {"model": "m", "dimension": "backend-dev", "details": json.dumps({"composite": {"scores": {"test_coverage": 60}}})},
        ]
        result = _extract_dimension_scores(rows, "backend-dev", "m")
        assert result["test_coverage"] == 80.0


class TestBuildRadarSvg:
    def test_basic_svg_output(self):
        scores = {"test_coverage": 80, "performance": 60, "code_style": 90}
        axes = [("test_coverage", "测试覆盖"), ("performance", "性能"), ("code_style", "风格")]
        svg = _build_radar_svg(scores, axes)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "测试覆盖" in svg

    def test_empty_scores(self):
        svg = _build_radar_svg({}, [("a", "A"), ("b", "B"), ("c", "C")])
        assert "<svg" in svg

    def test_polygon_points(self):
        scores = {"a": 100, "b": 0, "c": 50}
        axes = [("a", "A"), ("b", "B"), ("c", "C")]
        svg = _build_radar_svg(scores, axes, width=200, height=200)
        assert "<polygon" in svg

    def test_single_axis(self):
        svg = _build_radar_svg({"a": 50}, [("a", "A")])
        assert "<svg" in svg
        assert "数据不足" in svg


class TestBuildDimensionScoreTable:
    def test_basic(self):
        rows = [
            {
                "model": "model-a",
                "dimension": "backend-dev",
                "final_score": 82.5,
                "passed": 1,
                "details": json.dumps({
                    "composite": {
                        "weights": {"test_coverage": 0.4, "performance": 0.25},
                        "scores": {"test_coverage": 90.0, "performance": 75.0},
                    },
                }),
            },
            {
                "model": "model-a",
                "dimension": "backend-dev",
                "final_score": 70.0,
                "passed": 0,
                "details": json.dumps({
                    "composite": {
                        "weights": {"test_coverage": 0.4, "performance": 0.25},
                        "scores": {"test_coverage": 80.0, "performance": 60.0},
                    },
                }),
            },
        ]
        result = _build_dimension_score_table(rows)
        assert "model-a" in result
        assert "backend-dev" in result["model-a"]
        sub = result["model-a"]["backend-dev"]
        assert abs(sub["test_coverage"] - 85.0) < 0.1

    def test_empty_rows(self):
        result = _build_dimension_score_table([])
        assert result == {}

    def test_missing_details(self):
        rows = [{"model": "m", "dimension": "backend-dev", "final_score": 80, "passed": 1, "details": ""}]
        result = _build_dimension_score_table(rows)
        # 空的 details 不会创建任何条目
        assert result == {}
