import pytest
from unittest.mock import patch
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.composite import CompositeScorer


def make_frontend_ctx(
    code: str = "<html><body>hello</body></html>",
    task_type: str = "html",
    keywords: list[str] | None = None,
    test_cases: list[str] | None = None,
) -> ScoringContext:
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="front_test",
            dimension="frontend-dev",
            dataset="frontcode",
            prompt="test",
            expected_output="",
            metadata={
                "type": task_type,
                "keywords": keywords or [],
                "source": "frontcode",
            },
            test_cases=test_cases or [],
        ),
    )


class TestFunctionalityScorer:
    def test_empty_test_cases_gives_100(self):
        from benchmark.scorers.frontend.functionality import FunctionalityScorer
        r = FunctionalityScorer().score(make_frontend_ctx(test_cases=[]))
        assert r.score == 100.0

    def test_no_test_cases_field_gives_100(self):
        from benchmark.scorers.frontend.functionality import FunctionalityScorer
        r = FunctionalityScorer().score(make_frontend_ctx())
        assert r.score == 100.0

    def test_node_not_available_gives_100(self):
        from benchmark.scorers.frontend.functionality import FunctionalityScorer
        with patch("benchmark.scorers.frontend.functionality.shutil.which", return_value=None):
            r = FunctionalityScorer().score(
                make_frontend_ctx(code="console.log('hi')", task_type="javascript", test_cases=["assert true"])
            )
            assert r.score == 100.0

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.functionality import FunctionalityScorer
        assert FunctionalityScorer().get_metric_name() == "functionality"


class TestHTMLSemanticScorer:
    def test_semantic_html(self):
        from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
        code = "<header><nav></nav></header><main><article><section></section></article></main><footer></footer>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.score >= 60.0

    def test_non_semantic_html(self):
        from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
        code = "<div><div><div><span>hello</span></div></div></div>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.score <= 60.0

    def test_non_html_type(self):
        from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
        r = HTMLSemanticScorer().score(make_frontend_ctx(code="console.log('hi')", task_type="javascript"))
        assert r.score == 100.0

    def test_heading_hierarchy(self):
        from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
        code = "<h1>Title</h1><h2>Sub</h2><h3>Subsub</h3>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.details.get("heading_ok") is True

    def test_heading_skip(self):
        from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
        code = "<h1>Title</h1><h3>Skip h2</h3>"
        r = HTMLSemanticScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.details.get("heading_ok") is False

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.html_semantic import HTMLSemanticScorer
        assert HTMLSemanticScorer().get_metric_name() == "html_semantic"


class TestAccessibilityScorer:
    def test_playwright_unavailable_gives_100(self):
        from benchmark.scorers.frontend.accessibility import AccessibilityScorer
        with patch("benchmark.scorers.frontend.accessibility.shutil.which", return_value=None):
            r = AccessibilityScorer().score(make_frontend_ctx())
            assert r.score == 100.0

    def test_non_html_type(self):
        from benchmark.scorers.frontend.accessibility import AccessibilityScorer
        r = AccessibilityScorer().score(make_frontend_ctx(code="console.log('hi')", task_type="javascript"))
        assert r.score == 100.0

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.accessibility import AccessibilityScorer
        assert AccessibilityScorer().get_metric_name() == "accessibility"


class TestCSSQualityScorer:
    def test_non_css_type(self):
        from benchmark.scorers.frontend.css_quality import CSSQualityScorer
        r = CSSQualityScorer().score(make_frontend_ctx(code="console.log('hi')", task_type="javascript"))
        assert r.score == 100.0

    def test_good_css(self):
        from benchmark.scorers.frontend.css_quality import CSSQualityScorer
        css = "body { display: flex; margin: 0 auto; font-size: 1rem; } @media (max-width: 768px) { body { font-size: 0.875rem; } }"
        r = CSSQualityScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score >= 80.0

    def test_poor_css(self):
        from benchmark.scorers.frontend.css_quality import CSSQualityScorer
        css = "body { color: red; margin-top: 10px; padding: 5px; }"
        r = CSSQualityScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score <= 80.0

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.css_quality import CSSQualityScorer
        assert CSSQualityScorer().get_metric_name() == "css_quality"


class TestCodeOrganizationScorer:
    def test_non_js_type(self):
        from benchmark.scorers.frontend.code_organization import CodeOrganizationScorer
        r = CodeOrganizationScorer().score(make_frontend_ctx(code="<html></html>", task_type="html"))
        assert r.score == 100.0

    def test_well_organized_react(self):
        from benchmark.scorers.frontend.code_organization import CodeOrganizationScorer
        code = """
function MyComponent() {
  return <div>Hello</div>;
}
function Helper() {
  return null;
}
"""
        r = CodeOrganizationScorer().score(make_frontend_ctx(code=code, task_type="react"))
        assert r.score >= 80.0

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.code_organization import CodeOrganizationScorer
        assert CodeOrganizationScorer().get_metric_name() == "code_organization"


class TestPerformanceScorer:
    def test_non_html_type(self):
        from benchmark.scorers.frontend.performance import PerformanceScorer
        r = PerformanceScorer().score(make_frontend_ctx(code="console.log('hi')", task_type="javascript"))
        assert r.score == 100.0

    def test_clean_html(self):
        from benchmark.scorers.frontend.performance import PerformanceScorer
        r = PerformanceScorer().score(
            make_frontend_ctx(code="<html><body><img src='a.png' width='100' height='100'></body></html>", task_type="html")
        )
        assert r.score >= 70.0

    def test_img_without_dimensions(self):
        from benchmark.scorers.frontend.performance import PerformanceScorer
        code = "<html><body><img src='a.png'><img src='b.png'></body></html>"
        r = PerformanceScorer().score(make_frontend_ctx(code=code, task_type="html"))
        assert r.score < 70.0

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.performance import PerformanceScorer
        assert PerformanceScorer().get_metric_name() == "performance"


class TestBrowserCompatScorer:
    def test_non_css_type(self):
        from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer
        r = BrowserCompatScorer().score(make_frontend_ctx(code="console.log('hi')", task_type="javascript"))
        assert r.score == 100.0

    def test_no_vendor_prefixes(self):
        from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer
        css = "body { display: flex; gap: 10px; }"
        r = BrowserCompatScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score == 100.0

    def test_prefixes_without_supports(self):
        from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer
        css = "body { -webkit-transform: rotate(5deg); transform: rotate(5deg); }"
        r = BrowserCompatScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score == 60.0

    def test_prefixes_with_supports(self):
        from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer
        css = "@supports (display: grid) { body { display: grid; } } body { -webkit-transform: rotate(5deg); transform: rotate(5deg); }"
        r = BrowserCompatScorer().score(make_frontend_ctx(code=css, task_type="css"))
        assert r.score == 80.0

    def test_get_metric_name(self):
        from benchmark.scorers.frontend.browser_compat import BrowserCompatScorer
        assert BrowserCompatScorer().get_metric_name() == "browser_compat"


class TestFrontendComposite:
    def test_composite_integration(self):
        from benchmark.scorers.frontend import create_frontend_composite
        subs = create_frontend_composite()
        scorer = CompositeScorer(subs)
        ctx = make_frontend_ctx()
        r = scorer.score(ctx)
        assert r.score > 0.0

    def test_composite_all_defaults(self):
        from benchmark.scorers.frontend import create_frontend_composite
        subs = create_frontend_composite()
        scorer = CompositeScorer(subs)
        ctx = make_frontend_ctx(code="const x = 1;", task_type="javascript")
        r = scorer.score(ctx)
        assert r.score > 80.0
