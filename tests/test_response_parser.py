from benchmark.core.response_parser import extract_boxed, parse_response


def test_extract_boxed_simple():
    assert extract_boxed(r"The answer is \boxed{42}") == "42"


def test_extract_boxed_fraction():
    assert extract_boxed(r"\boxed{\frac{14}{3}}") == r"\frac{14}{3}"


def test_extract_boxed_degree():
    assert extract_boxed(r"\boxed{90^\circ}") == r"90^\circ"


def test_extract_boxed_nested():
    assert extract_boxed(r"\boxed{\frac{3\sqrt{3}}{4}}") == r"\frac{3\sqrt{3}}{4}"


def test_extract_boxed_double_nested():
    assert extract_boxed(r"\boxed{\boxed{42}}") == "42"


def test_extract_boxed_none():
    assert extract_boxed("no boxed answer here") == ""


def test_parse_response_math_boxed():
    result = parse_response(
        r"Let me solve this... The answer is \boxed{\frac{14}{3}}",
        "reasoning",
    )
    assert result.answer == r"\frac{14}{3}"


def test_parse_response_reasoning_json_fallback():
    """如果无 \\boxed{}，仍回退到 JSON 提取."""
    result = parse_response('{"answer": "42"}', "reasoning")
    assert result.answer == "42"
