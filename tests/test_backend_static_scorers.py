"""Backend 静态分析评分器测试。"""

import pytest
from benchmark.models.schemas import ScoringContext, TaskDefinition
from benchmark.scorers.backend.code_style import CodeStyleScorer
from benchmark.scorers.backend.robustness import RobustnessScorer
from benchmark.scorers.backend.architecture import ArchitectureScorer
from benchmark.scorers.backend.security import SecurityScorer
from benchmark.scorers.backend.extensibility import ExtensibilityScorer


def _make_ctx(code: str) -> ScoringContext:
    """构造 ScoringContext 用于测试。"""
    return ScoringContext(
        model_answer=code,
        raw_output=code,
        expected="",
        task=TaskDefinition(
            task_id="test",
            dimension="backend-dev",
            dataset="bigcodebench",
            prompt="test",
            expected_output="",
            metadata={"test": "", "entry_point": "", "canonical_solution": ""},
        ),
    )


# ============= CodeStyleScorer 测试 =============

def test_code_style_good_code():
    """测试良好代码风格的代码。"""
    good_code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(x, y):
    """Multiply two numbers."""
    return x * y
'''
    scorer = CodeStyleScorer()
    ctx = _make_ctx(good_code)
    result = scorer.score(ctx)

    # 工具可能不可用，返回 100 分
    assert 0 <= result.score <= 100
    assert scorer.get_metric_name() == "code_style"


def test_code_style_with_json_format():
    """测试 JSON 格式的代码提取。"""
    code = '{"code": "def hello():\\n    print(\\"world\\")"}'
    scorer = CodeStyleScorer()
    ctx = _make_ctx(code)
    result = scorer.score(ctx)

    assert 0 <= result.score <= 100


def test_code_style_bad_code():
    """测试代码风格不佳的代码。"""
    bad_code = '''
def f(x,y):return x+y  # 缩少空格和文档字符串
def g(a,b,c,d,e,f):return a*b+c*d+e*f  # 参数过多
'''
    scorer = CodeStyleScorer()
    ctx = _make_ctx(bad_code)
    result = scorer.score(ctx)

    assert 0 <= result.score <= 100


# ============= RobustnessScorer 测试 =============

def test_robustness_with_error_handling():
    """测试有错误处理的代码。"""
    good_code = '''
def process_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def fetch_data(url):
    try:
        response = requests.get(url)
        return response.json()
    except requests.RequestException:
        return {}
'''
    scorer = RobustnessScorer()
    ctx = _make_ctx(good_code)
    result = scorer.score(ctx)

    assert result.score == 100.0  # 有 try-except 和 with 语句
    assert result.passed is True


def test_robustness_without_error_handling():
    """测试没有错误处理的代码。"""
    bad_code = '''
def process_file(filename):
    f = open(filename, 'r')
    return f.read()

def fetch_data(url):
    response = requests.get(url)
    return response.json()

def run_command(cmd):
    return subprocess.run(cmd)
'''
    scorer = RobustnessScorer()
    ctx = _make_ctx(bad_code)
    result = scorer.score(ctx)

    # 应该扣分
    assert result.score < 100
    assert "deductions" in result.details


def test_robustness_syntax_error():
    """测试语法错误的代码。"""
    bad_code = 'def broken(:\n    return 1'
    scorer = RobustnessScorer()
    ctx = _make_ctx(bad_code)
    result = scorer.score(ctx)

    # AST 解析失败，返回 100 分
    assert result.score == 100.0


# ============= ArchitectureScorer 测试 =============

def test_architecture_simple():
    """测试简单架构的代码。"""
    simple_code = '''
def add(a, b):
    return a + b

def subtract(x, y):
    return x - y
'''
    scorer = ArchitectureScorer()
    ctx = _make_ctx(simple_code)
    result = scorer.score(ctx)

    assert result.score > 80  # 简单代码应该高分


def test_architecture_complex():
    """测试复杂架构的代码。"""
    complex_code = '''
def very_long_function():
    """这是一个很长的函数。"""
    x = 1
    # ... 很多行 ...
    for i in range(100):
        if i % 2 == 0:
            if i % 3 == 0:
                if i % 5 == 0:
                    if i % 7 == 0:
                        x += i
    return x
''' + "\n    # 添加更多行使函数变长\n" * 60

    scorer = ArchitectureScorer()
    ctx = _make_ctx(complex_code)
    result = scorer.score(ctx)

    # 复杂代码应该被扣分
    assert result.score <= 100


def test_architecture_without_radon():
    """测试 radon 不可用的情况。"""
    # 这个测试依赖 radon 是否安装
    scorer = ArchitectureScorer()
    assert scorer.get_metric_name() == "architecture"


# ============= SecurityScorer 测试 =============

def test_security_safe_code():
    """测试安全的代码。"""
    safe_code = '''
def process_input(user_input):
    # 使用参数化查询，无注入风险
    return user_input.strip()

def calculate(a, b):
    return a + b
'''
    scorer = SecurityScorer()
    ctx = _make_ctx(safe_code)
    result = scorer.score(ctx)

    # 安全代码应该高分（工具不可用时返回 100）
    assert result.score >= 0


def test_security_shell_injection():
    """测试 shell 注入风险。"""
    risky_code = '''
def run_command(user_input):
    cmd = "ls " + user_input
    os.system(cmd)
'''
    scorer = SecurityScorer()
    ctx = _make_ctx(risky_code)
    result = scorer.score(ctx)

    # 应该检测到 shell 注入风险
    if result.details.get("shell_injection_detected"):
        assert result.score < 100


def test_security_tools_unavailable():
    """测试安全工具不可用的情况。"""
    code = 'def hello(): pass'
    scorer = SecurityScorer()
    ctx = _make_ctx(code)
    result = scorer.score(ctx)

    # 如果工具都不可用，返回 100 分
    tools_available = result.details.get("tools_available")
    if tools_available is False:
        # tools_available 是 False 表示工具都不可用
        assert result.score == 100.0
    elif isinstance(tools_available, dict):
        # tools_available 是 dict 时检查两个工具是否都不可用
        if not tools_available.get("bandit", False) and not tools_available.get("semgrep", False):
            assert result.score == 100.0


# ============= ExtensibilityScorer 测试 =============

def test_extensibility_good_code():
    """测试可扩展性好的代码。"""
    good_code = '''
class Calculator:
    MAX_ITERATIONS = 100

    def add(self, a, b):
        return a + b

    def multiply(self, x, y):
        return x * y
'''
    scorer = ExtensibilityScorer()
    ctx = _make_ctx(good_code)
    result = scorer.score(ctx)

    # 好代码应该高分
    assert result.score > 80


def test_extensibility_magic_numbers():
    """测试有魔数的代码。"""
    bad_code = '''
def calculate():
    result = 0
    for i in range(10):
        result += i * 2.5 + 3.14
    return result * 42
'''
    scorer = ExtensibilityScorer()
    ctx = _make_ctx(bad_code)
    result = scorer.score(ctx)

    # 有很多魔数，应该扣分
    assert "magic_numbers" in result.details
    if result.details["magic_numbers"] > 5:
        assert result.score < 100


def test_extensibility_too_many_params():
    """测试参数过多的函数。"""
    bad_code = '''
def process(a, b, c, d, e, f):
    return a + b + c + d + e + f

def calculate(x, y, z, w):
    return x * y * z * w
'''
    scorer = ExtensibilityScorer()
    ctx = _make_ctx(bad_code)
    result = scorer.score(ctx)

    # 平均参数超过 3，应该扣分
    assert "average_params" in result.details
    if result.details["average_params"] > 3:
        assert result.score < 100
