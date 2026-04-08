"""代码架构评分器。使用 radon 分析圈复杂度和函数长度。"""

from __future__ import annotations

import ast
from typing import cast

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.backend._utils import extract_code, safe_parse_ast

try:
    from radon.complexity import cc_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False


class ArchitectureScorer(BaseScorer):
    """代码架构评分器。

    使用 radon 分析:
    1. 圈复杂度（Cyclomatic Complexity）: max CC > 10 扣 15 分, > 20 扣 30 分
    2. 函数长度: > 50 行扣 10 分, > 100 行扣 25 分

    基础分 100 分，根据扣分项计算最终得分。
    radon 不可用时返回 100 分。
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行架构评分。"""
        code = extract_code(ctx.model_answer)

        if not RADON_AVAILABLE:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"radon_available": False},
                reasoning="radon 不可用，跳过架构检查"
            )

        tree = safe_parse_ast(code)
        if tree is None:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"ast_parse_failed": True},
                reasoning="代码解析失败，跳过架构检查"
            )

        deductions = {}

        # 1. 检查圈复杂度
        max_cc = self._get_max_complexity(code)
        if max_cc > 20:
            deductions["high_complexity"] = 30
        elif max_cc > 10:
            deductions["medium_complexity"] = 15

        # 2. 检查函数长度
        max_func_len = self._get_max_function_length(tree)
        if max_func_len > 100:
            deductions["very_long_function"] = 25
        elif max_func_len > 50:
            deductions["long_function"] = 10

        # 计算最终得分
        total_deduction = sum(deductions.values())
        final_score = max(0, 100 - total_deduction)

        return ScoreResult(
            score=float(final_score),
            passed=final_score >= 60.0,
            details={
                "max_complexity": max_cc,
                "max_function_length": max_func_len,
                "deductions": deductions,
                "radon_available": True
            },
            reasoning=f"架构得分: {final_score:.1f} (max CC: {max_cc}, 最长函数: {max_func_len}行)"
        )

    def _get_max_complexity(self, code: str) -> int:
        """获取最大圈复杂度。"""
        try:
            results = cc_visit(code)
            if not results:
                return 0

            max_cc = 0
            for result in results:
                # result.complexity 是圈复杂度
                cc = getattr(result, "complexity", 0)
                if cc > max_cc:
                    max_cc = cc

            return max_cc
        except Exception:
            return 0

    def _get_max_function_length(self, tree: ast.AST) -> int:
        """获取最长函数的行数。"""
        class FunctionLengthAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.max_length = 0

            def visit_FunctionDef(self, node):
                # 计算函数体行数（不包括装饰器和定义行）
                if node.body:
                    first_line = node.body[0].lineno
                    last_line = node.body[-1].end_lineno or first_line
                    length = last_line - first_line + 1
                    if length > self.max_length:
                        self.max_length = length
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                # 同样处理异步函数
                if node.body:
                    first_line = node.body[0].lineno
                    last_line = node.body[-1].end_lineno or first_line
                    length = last_line - first_line + 1
                    if length > self.max_length:
                        self.max_length = length
                self.generic_visit(node)

        analyzer = FunctionLengthAnalyzer()
        analyzer.visit(tree)
        return analyzer.max_length

    def get_metric_name(self) -> str:
        return "architecture"
