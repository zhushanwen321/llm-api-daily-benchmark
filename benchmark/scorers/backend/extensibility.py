"""代码可扩展性评分器。使用 AST 检查硬编码和函数参数。"""

from __future__ import annotations

import ast

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.backend._utils import extract_code, safe_parse_ast


class ExtensibilityScorer(BaseScorer):
    """代码可扩展性评分器。

    检查:
    1. 硬编码数字字面量（排除 0, 1, -1, import 参数中的数字）:
       magic_numbers > 5 → 扣 15 分
    2. 函数参数（平均参数数 > 3 → 扣 10 分）

    基础分 100 分，根据扣分项计算最终得分。
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行可扩展性评分。"""
        code = extract_code(ctx.model_answer)
        tree = safe_parse_ast(code)

        if tree is None:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"ast_parse_failed": True},
                reasoning="代码解析失败，跳过可扩展性检查"
            )

        deductions = {}

        # 1. 检查硬编码数字
        magic_numbers = self._count_magic_numbers(tree)
        if magic_numbers > 5:
            deductions["too_many_magic_numbers"] = 15

        # 2. 检查函数参数
        avg_params = self._get_average_function_params(tree)
        if avg_params > 3:
            deductions["too_many_params"] = 10

        # 计算最终得分
        total_deduction = sum(deductions.values())
        final_score = max(0, 100 - total_deduction)

        return ScoreResult(
            score=float(final_score),
            passed=final_score >= 60.0,
            details={
                "magic_numbers": magic_numbers,
                "average_params": avg_params,
                "deductions": deductions
            },
            reasoning=f"可扩展性得分: {final_score:.1f} (魔数: {magic_numbers}, 平均参数: {avg_params:.1f})"
        )

    def _count_magic_numbers(self, tree: ast.AST) -> int:
        """统计硬编码数字字面量。

        排除:
        - 0, 1, -1（常用值）
        - import 语句中的数字
        """
        class MagicNumberCounter(ast.NodeVisitor):
            def __init__(self):
                self.count = 0
                self.in_import = False

            def visit_Import(self, node):
                self.in_import = True
                self.generic_visit(node)
                self.in_import = False

            def visit_ImportFrom(self, node):
                self.in_import = True
                self.generic_visit(node)
                self.in_import = False

            def visit_Constant(self, node):
                # 检查数字常量
                if isinstance(node.value, (int, float)):
                    # 排除常用值
                    if node.value not in {0, 1, -1}:
                        # 排除 import 中的数字
                        if not self.in_import:
                            self.count += 1

            def visit_Num(self, node):
                # Python 3.7 兼容
                if node.n not in {0, 1, -1} and not self.in_import:
                    self.count += 1

        counter = MagicNumberCounter()
        counter.visit(tree)
        return counter.count

    def _get_average_function_params(self, tree: ast.AST) -> float:
        """计算函数平均参数数量。"""
        class ParamCounter(ast.NodeVisitor):
            def __init__(self):
                self.total_params = 0
                self.function_count = 0

            def visit_FunctionDef(self, node):
                # 只统计非特殊方法（不以 __ 开头）
                if not node.name.startswith("__"):
                    # 排除 self 参数
                    params = [
                        arg for arg in node.args.args
                        if arg.arg not in {"self", "cls"}
                    ]
                    self.total_params += len(params)
                    self.function_count += 1
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                # 同样处理异步函数
                if not node.name.startswith("__"):
                    params = [
                        arg for arg in node.args.args
                        if arg.arg not in {"self", "cls"}
                    ]
                    self.total_params += len(params)
                    self.function_count += 1
                self.generic_visit(node)

        counter = ParamCounter()
        counter.visit(tree)

        if counter.function_count == 0:
            return 0.0

        return counter.total_params / counter.function_count

    def get_metric_name(self) -> str:
        return "extensibility"
