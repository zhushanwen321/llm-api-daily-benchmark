"""Backend development scorers."""
from benchmark.scorers.backend.architecture import ArchitectureScorer
from benchmark.scorers.backend.code_style import CodeStyleScorer
from benchmark.scorers.backend.extensibility import ExtensibilityScorer
from benchmark.scorers.backend.performance import PerformanceScorer
from benchmark.scorers.backend.robustness import RobustnessScorer
from benchmark.scorers.backend.security import SecurityScorer
from benchmark.scorers.backend.test_coverage import TestCoverageScorer
from benchmark.scorers.composite import CompositeScorer


def create_backend_composite() -> CompositeScorer:
    return CompositeScorer([
        (0.40, TestCoverageScorer()),
        (0.25, PerformanceScorer()),
        (0.15, CodeStyleScorer()),
        (0.10, RobustnessScorer()),
        (0.05, ArchitectureScorer()),
        (0.03, SecurityScorer()),
        (0.02, ExtensibilityScorer()),
    ])


__all__ = [
    "CodeStyleScorer",
    "RobustnessScorer",
    "ArchitectureScorer",
    "SecurityScorer",
    "ExtensibilityScorer",
    "TestCoverageScorer",
    "PerformanceScorer",
    "create_backend_composite",
]
