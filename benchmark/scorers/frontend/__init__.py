from __future__ import annotations
from benchmark.scorers.frontend.functionality import FunctionalityScorer
from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
from benchmark.scorers.frontend.accessibility import AccessibilityScorer
from benchmark.scorers.frontend.css_quality import CSSQualityScorer
from benchmark.scorers.frontend.code_organization import CodeOrganizationScorer
from benchmark.scorers.frontend.performance import PerformanceScorer
from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer

def create_frontend_composite() -> list[tuple[float, object]]:
    return [
        (0.30, FunctionalityScorer()),
        (0.20, HTMLSemanticScorer()),
        (0.15, AccessibilityScorer()),
        (0.15, CSSQualityScorer()),
        (0.10, CodeOrganizationScorer()),
        (0.05, PerformanceScorer()),
        (0.05, BrowserCompatScorer()),
    ]
