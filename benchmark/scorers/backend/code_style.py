"""代码风格评分器。使用 pylint 和 flake8 进行代码风格检查。"""

from __future__ import annotations

import json
import subprocess
import tempfile

from benchmark.models.schemas import ScoreResult, ScoringContext
from benchmark.scorers.base import BaseScorer
from benchmark.scorers.backend._utils import extract_code


class CodeStyleScorer(BaseScorer):
    """代码风格评分器。

    使用 pylint 和 flake8 检查代码风格:
    - pylint: 评估整体代码质量（0-10 分），转换为 0-100 分
    - flake8: 检查严重错误（E9,F63,F7,F82），每个违规扣 2 分

    工具不可用时返回 100 分（不惩罚）。
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """对模型输出进行代码风格评分。"""
        code = extract_code(ctx.model_answer)

        # 尝试 pylint 评分
        pylint_score = self._run_pylint(code)
        if pylint_score is None:
            # pylint 不可用，返回 100 分
            return ScoreResult(
                score=100.0,
                passed=True,
                details={"pylint_available": False},
                reasoning="pylint 不可用，跳过代码风格检查"
            )

        # 尝试 flake8 检查
        flake8_penalty = self._run_flake8(code)
        if flake8_penalty is None:
            # flake8 不可用，仅使用 pylint 分数
            final_score = pylint_score
            details = {"pylint_score": pylint_score, "flake8_available": False}
        else:
            final_score = max(0, pylint_score - flake8_penalty)
            details = {
                "pylint_score": pylint_score,
                "flake8_penalty": flake8_penalty,
                "flake8_available": True
            }

        passed = final_score >= 60.0

        return ScoreResult(
            score=float(final_score),
            passed=passed,
            details=details,
            reasoning=f"代码风格得分: {final_score:.1f} (pylint: {pylint_score:.1f}, flake8扣分: {flake8_penalty if flake8_penalty else 0})"
        )

    def _run_pylint(self, code: str) -> float | None:
        """运行 pylint 获取评分。

        Returns:
            0-100 分，或 None 表示 pylint 不可用。
        """
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()

            result = subprocess.run(
                [
                    "pylint",
                    "--disable=all",
                    "--enable=C,W,R",
                    "-sn",  # 不显示报告
                    "--output-format=json",
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
            # pylint 输出 JSON 包含 score 字段（0-10）
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                pylint_raw = data.get("score", 0)
            elif isinstance(data, list):
                # 旧版本 pylint 可能返回空列表或不同格式
                pylint_raw = 10.0
            else:
                pylint_raw = 0.0

            # pylint 分数 0-10 转换为 0-100
            return float(pylint_raw) * 10
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _run_flake8(self, code: str) -> float | None:
        """运行 flake8 统计严重违规。

        Returns:
            扣分（每个违规 2 分，上限 20 分），或 None 表示 flake8 不可用。
        """
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()

            result = subprocess.run(
                [
                    "flake8",
                    "--select=E9,F63,F7,F82",  # 仅选严重错误
                    "--max-line-length=120",
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

        # 统计违规行数
        violations = result.stdout.strip().split("\n") if result.stdout.strip() else []
        if not violations or violations == [""]:
            return 0.0

        violation_count = len([v for v in violations if v.strip()])
        # 每个违规扣 2 分，上限 20 分
        penalty = min(20, violation_count * 2)
        return float(penalty)

    def get_metric_name(self) -> str:
        return "code_style"
