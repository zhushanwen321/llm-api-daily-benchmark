"""数学表达式评分器。支持数值比较."""

from __future__ import annotations

import math
import re

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer


def _extract_balanced_braces(text: str, start: int) -> tuple[str, int]:
    """从 start 位置（指向 { 后第一个字符）开始，平衡匹配花括号."""
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip(), pos


def _normalize_latex(expr: str) -> str:
    """将 LaTeX 表达式转换为可解析的 Python 表达式."""
    s = expr.strip()

    # 去掉 \left \right
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\left[", "[")
    s = s.replace("\\right]", "]")
    s = s.replace("\\left\\{", "{")
    s = s.replace("\\right\\}", "}")

    # \dfrac, \tfrac 统一为 \frac
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")

    # \frac{a}{b} -> (a)/(b) — 用花括号平衡而非正则
    while "\\frac" in s:
        idx = s.find("\\frac")
        pos = idx + len("\\frac")
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s) or s[pos] != '{':
            break
        arg1, pos = _extract_balanced_braces(s, pos + 1)
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s) or s[pos] != '{':
            break
        arg2, pos = _extract_balanced_braces(s, pos + 1)
        s = s[:idx] + f"({arg1})/({arg2})" + s[pos:]

    # \mbox{...}, \text{...} 等文本/单位标注 -> 整体移除
    for cmd in ("\\mbox", "\\text", "\\mathrm", "\\textbf", "\\textit"):
        while cmd in s:
            idx = s.find(cmd)
            pos = idx + len(cmd)
            while pos < len(s) and s[pos] in ' \t':
                pos += 1
            if pos >= len(s) or s[pos] != '{':
                break
            _, pos = _extract_balanced_braces(s, pos + 1)
            s = s[:idx] + s[pos:]

    # \sqrt{expr} -> sqrt(expr)
    while "\\sqrt{" in s:
        idx = s.find("\\sqrt{")
        pos = idx + len("\\sqrt{") - 1
        arg, pos = _extract_balanced_braces(s, pos + 1)
        s = s[:idx] + f"sqrt({arg})" + s[pos:]

    # \sqrt 后直接跟数字（无花括号）
    s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
    # \sqrt 后直接跟字母
    s = re.sub(r"\\sqrt([a-z])", r"sqrt(\1)", s)
    # 裸 \sqrt -> sqrt
    s = s.replace("\\sqrt", "sqrt")

    # ^\circ -> 去掉
    s = s.replace("^\\circ", "")

    # 移除尾随的 ^{...} 或 ^\S（剥离单位后的孤立上标，如 inches^2）
    s = re.sub(r"\s*\^\{[^}]*\}\s*$", "", s)
    s = re.sub(r"\s*\^\S\s*$", "", s)

    # \pi -> pi
    s = s.replace("\\pi", "pi")
    # \cdot -> *
    s = s.replace("\\cdot", "*")
    # 去掉多余反斜杠
    s = s.replace("\\", "")
    return s.strip()


def _strip_equals(s: str) -> str:
    """预处理含 = 的表达式: 尝试取等号右侧的值."""
    if "=" not in s:
        return s
    parts = s.split("=", 1)
    right = parts[1].strip()
    if right:
        return right
    return s


def _try_numeric_match(a: str, b: str) -> bool:
    """尝试将两个表达式解析为数值并比较."""
    a_norm = _normalize_latex(_strip_equals(a))
    b_norm = _normalize_latex(_strip_equals(b))
    safe_globals = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        val_a = float(eval(a_norm, safe_globals, {}))  # noqa: S307
        val_b = float(eval(b_norm, safe_globals, {}))  # noqa: S307
        return math.isclose(val_a, val_b, rel_tol=1e-6)
    except Exception:
        return False


class MathScorer(BaseScorer):
    """数学题评分器.

    支持三种匹配模式:
    1. 字符串精确匹配
    2. 空格归一化字符串匹配
    3. 数值比较（normalize 后安全 eval）
    """

    @staticmethod
    def _normalize_spaces(s: str) -> str:
        """归一化结构化表达式中的空格：逗号和括号周围."""
        s = re.sub(r"\s*,\s*", ",", s)
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        return s.strip()

    def score(self, ctx: ScoringContext) -> ScoreResult:
        predicted = ctx.model_answer.strip()
        expected = ctx.expected.strip()

        # 快速路径: 字符串精确匹配
        if predicted == expected:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted, "expected": expected, "method": "string"},
                reasoning=f"Correct: {predicted}",
            )

        # 空格归一化比较
        if self._normalize_spaces(predicted) == self._normalize_spaces(expected):
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted, "expected": expected, "method": "space_normalized"},
                reasoning=f"Correct (space_normalized): {predicted} == {expected}",
            )

        # 数值比较
        if _try_numeric_match(predicted, expected):
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"predicted": predicted, "expected": expected, "method": "numeric"},
                reasoning=f"Correct (numeric): {predicted} == {expected}",
            )

        return ScoreResult(
            score=0.0,
            passed=False,
            details={"predicted": predicted, "expected": expected},
            reasoning=f"Incorrect: predicted={predicted}, expected={expected}",
        )

    def get_metric_name(self) -> str:
        return "math_match"
