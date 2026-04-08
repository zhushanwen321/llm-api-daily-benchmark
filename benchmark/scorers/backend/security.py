"""代码安全性评分器。使用 bandit、semgrep 和 AST 检查安全问题。"""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.backend._utils import extract_code, safe_parse_ast


class SecurityScorer(BaseScorer):
    """代码安全性评分器。

    检查:
    1. bandit 扫描: HIGH 扣 20 分, MEDIUM 扣 10 分
    2. semgrep 扫描: 每个发现扣 5 分（上限 20 分）
    3. shell 注入检测: AST 检测 os.system + 字符串拼接 → 扣 30 分

    基础分 100 分，根据扣分项计算最终得分。
    两个工具都不可用时返回 100 分。
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行安全性评分。"""
        code = extract_code(ctx.model_answer)

        deductions = {}
        tool_available = {"bandit": False, "semgrep": False}

        # 1. bandit 扫描
        bandit_penalty = self._run_bandit(code)
        if bandit_penalty is not None:
            tool_available["bandit"] = True
            if bandit_penalty > 0:
                deductions["bandit_issues"] = bandit_penalty

        # 2. semgrep 扫描
        semgrep_penalty = self._run_semgrep(code)
        if semgrep_penalty is not None:
            tool_available["semgrep"] = True
            if semgrep_penalty > 0:
                deductions["semgrep_issues"] = semgrep_penalty

        # 3. AST 检查 shell 注入
        tree = safe_parse_ast(code)
        if tree is not None:
            shell_injection = self._check_shell_injection(tree)
            if shell_injection:
                deductions["shell_injection_risk"] = 30

        # 如果两个工具都不可用，返回 100 分
        if not tool_available["bandit"] and not tool_available["semgrep"]:
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"tools_available": False},
                reasoning="安全工具不可用，跳过安全检查"
            )

        # 计算最终得分
        total_deduction = sum(deductions.values())
        final_score = max(0, 100 - total_deduction)

        return ScoreResult(
            score=float(final_score),
            passed=final_score >= 60.0,
            details={
                "deductions": deductions,
                "tools_available": tool_available,
                "shell_injection_detected": shell_injection if tree else False
            },
            reasoning=f"安全性得分: {final_score:.1f} (扣分项: {list(deductions.keys())})"
        )

    def _run_bandit(self, code: str) -> float | None:
        """运行 bandit 扫描安全问题。

        Returns:
            扣分（HIGH 扣 20 分, MEDIUM 扣 10 分），或 None 表示 bandit 不可用。
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

            penalty = 0
            for r in results:
                severity = r.get("issue_severity", "LOW").upper()
                if severity == "HIGH":
                    penalty += 20
                elif severity == "MEDIUM":
                    penalty += 10

            return float(penalty)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _run_semgrep(self, code: str) -> float | None:
        """运行 semgrep 扫描安全问题。

        Returns:
            扣分（每个发现扣 5 分，上限 20 分），或 None 表示 semgrep 不可用。
        """
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()

            result = subprocess.run(
                [
                    "semgrep",
                    "--config", "auto",
                    "--json",
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

            # 每个发现扣 5 分，上限 20 分
            penalty = min(20, len(results) * 5)
            return float(penalty)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _check_shell_injection(self, tree: ast.AST) -> bool:
        """检查是否存在 shell 注入风险。

        检测 os.system + 字符串拼接的模式。
        """
        class ShellInjectionChecker(ast.NodeVisitor):
            def __init__(self):
                self.has_injection_risk = False

            def visit_Call(self, node):
                # 检查 os.system 调用
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "system":
                        # 检查是否是 os.system
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                            # 检查参数是否包含字符串拼接
                            if node.args:
                                arg = node.args[0]
                                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                                    # 字符串拼接
                                    self.has_injection_risk = True
                                elif isinstance(arg, ast.JoinedStr):
                                    # f-string
                                    self.has_injection_risk = True

                self.generic_visit(node)

        checker = ShellInjectionChecker()
        checker.visit(tree)
        return checker.has_injection_risk

    def get_metric_name(self) -> str:
        return "security"
