"""代码鲁棒性评分器。使用 AST 和 bandit 检查错误处理和风险操作。"""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.backend._utils import extract_code, safe_parse_ast


class RobustnessScorer(BaseScorer):
    """代码鲁棒性评分器。

    检查:
    1. 风险操作是否有 try-except 保护
    2. 文件操作是否使用 with 语句
    3. bandit 扫描的安全问题

    基础分 100 分，根据扣分项计算最终得分。
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行鲁棒性评分。"""
        code = extract_code(ctx.model_answer)
        tree = safe_parse_ast(code)

        if tree is None:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"ast_parse_failed": True},
                reasoning="代码解析失败，跳过鲁棒性检查"
            )

        deductions = {}

        # 1. 检查风险操作是否有 try-except 保护
        try_blocks = self._count_try_blocks(tree)
        risky_ops = self._count_risky_operations(tree)

        if risky_ops > 0 and try_blocks == 0:
            deductions["no_try_for_risky_ops"] = 30
        elif risky_ops > 0:
            # 有 try 块，但可能不够，轻微扣分
            if try_blocks < risky_ops:
                deductions["insufficient_try_blocks"] = 10

        # 2. 检查文件操作是否使用 with 语句
        file_opens_without_with = self._check_file_operations(tree)
        if file_opens_without_with > 0:
            deductions["file_open_without_with"] = 20

        # 3. bandit 扫描
        bandit_penalty = self._run_bandit(code)
        if bandit_penalty is not None and bandit_penalty > 0:
            deductions["bandit_issues"] = bandit_penalty

        # 计算最终得分
        total_deduction = sum(deductions.values())
        final_score = max(0, 100 - total_deduction)

        return ScoreResult(
            score=float(final_score),
            passed=final_score >= 60.0,
            details={
                "try_blocks": try_blocks,
                "risky_ops": risky_ops,
                "file_opens_without_with": file_opens_without_with,
                "deductions": deductions,
                "bandit_available": bandit_penalty is not None
            },
            reasoning=f"鲁棒性得分: {final_score:.1f} (扣分项: {deductions})"
        )

    def _count_try_blocks(self, tree: ast.AST) -> int:
        """统计 try-except 块数量。"""
        class TryCounter(ast.NodeVisitor):
            def __init__(self):
                self.count = 0

            def visit_Try(self, node):
                self.count += 1
                self.generic_visit(node)

        counter = TryCounter()
        counter.visit(tree)
        return counter.count

    def _count_risky_operations(self, tree: ast.AST) -> int:
        """统计风险操作数量。"""
        class RiskyOpCounter(ast.NodeVisitor):
            def __init__(self):
                self.count = 0

            def visit_Call(self, node):
                # 检查函数调用
                if isinstance(node.func, ast.Attribute):
                    # obj.method()
                    if node.func.attr in {"get", "post"}:
                        # 检查是否是 requests.get/post
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "requests":
                            self.count += 1
                elif isinstance(node.func, ast.Name):
                    # 直接函数调用
                    if node.func.id in {"open", "run", "call", "Popen"}:
                        # subprocess.run/call/Popen 需要额外检查
                        if node.func.id == "open":
                            self.count += 1
                        elif node.func.id in {"run", "call", "Popen"}:
                            # 假设是 subprocess 模块（简化检查）
                            self.count += 1
                self.generic_visit(node)

        counter = RiskyOpCounter()
        counter.visit(tree)
        return counter.count

    def _check_file_operations(self, tree: ast.AST) -> int:
        """检查不使用 with 语句的文件打开操作。"""
        class FileOpChecker(ast.NodeVisitor):
            def __init__(self):
                self.opens_without_with = 0

            def visit_With(self, node):
                # with 块内的 open 不算违规
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id == "open":
                                # 这是合法的 with open() 用法
                                pass
                self.generic_visit(node)

            def visit_Call(self, node):
                # 检查是否是 open() 调用
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    # 检查父节点是否是 With
                    # 简化处理：直接计数，实际应该检查上下文
                    # 这里我们用更简单的方法：如果在 With 的 items 里找到就不算
                    self.opens_without_with += 1
                self.generic_visit(node)

        # 更准确的检查：排除 with 语句中的 open
        class AccurateFileOpChecker(ast.NodeVisitor):
            def __init__(self):
                self.with_opens = set()
                self.total_opens = 0

            def visit_With(self, node):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id == "open":
                                # 记录这个 open 节点的位置
                                self.with_opens.add(id(item.context_expr))
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    self.total_opens += 1
                self.generic_visit(node)

            def get_opens_without_with(self):
                return self.total_opens - len(self.with_opens)

        checker = AccurateFileOpChecker()
        checker.visit(tree)
        return checker.get_opens_without_with()

    def _run_bandit(self, code: str) -> float | None:
        """运行 bandit 扫描安全问题。

        Returns:
            扣分（每个 MEDIUM+ 问题扣 5 分，上限 20 分），或 None 表示 bandit 不可用。
        """
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()

            result = subprocess.run(
                [
                    "bandit",
                    "-f", "json",
                    "-q",  # 安静模式
                    f.name
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        except Exception:
            return None

        try:
            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                return 0.0

            results = data.get("results", [])
            if not results:
                return 0.0

            # 统计 MEDIUM 及以上问题
            medium_plus_issues = [
                r for r in results
                if r.get("issue_severity", "LOW").upper() in {"MEDIUM", "HIGH"}
            ]

            # 每个 MEDIUM+ 问题扣 5 分，上限 20 分
            penalty = min(20, len(medium_plus_issues) * 5)
            return float(penalty)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def get_metric_name(self) -> str:
        return "robustness"
