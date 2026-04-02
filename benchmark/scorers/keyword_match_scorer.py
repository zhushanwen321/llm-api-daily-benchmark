"""关键词匹配评分器。用于前端代码评测."""

from __future__ import annotations

import re
from typing import List

from benchmark.models.schemas import ScoreResult, TaskDefinition
from benchmark.scorers.base import BaseScorer


class KeywordMatchScorer(BaseScorer):
    """关键词匹配评分器.

    检查代码是否包含预期的关键词或正则表达式模式。
    根据匹配项数量计算得分（匹配数/总数 * 100）。
    """

    def __init__(
        self,
        keywords: List[str],
        use_regex: bool = False,
        case_sensitive: bool = False
    ):
        """初始化评分器.

        Args:
            keywords: 关键词列表或正则表达式列表.
            use_regex: 是否使用正则表达式匹配.
            case_sensitive: 是否区分大小写.
        """
        self.keywords = keywords
        self.use_regex = use_regex
        self.case_sensitive = case_sensitive

    def score(
        self,
        model_output: str,
        expected: str,  # noqa: ARG002 — 未使用
        task: TaskDefinition,  # noqa: ARG002 — 未使用
    ) -> ScoreResult:
        """对模型输出进行评分.

        Args:
            model_output: 模型生成的代码.
            expected: 未使用.
            task: 原始任务定义.

        Returns:
            ScoreResult 包含分数、是否通过、详情.
        """
        if not self.keywords:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"error": "No keywords configured"},
                reasoning="No keywords to match"
            )

        search_text = model_output if self.case_sensitive else model_output.lower()
        matched = []
        matched_indices = []

        for idx, keyword in enumerate(self.keywords):
            search_keyword = keyword if self.case_sensitive else keyword.lower()

            if self.use_regex:
                if re.search(search_keyword, search_text):
                    matched.append(keyword)
                    matched_indices.append(idx)
            else:
                if search_keyword in search_text:
                    matched.append(keyword)
                    matched_indices.append(idx)

        score = len(matched) / len(self.keywords) * 100
        passed = score >= 50.0  # 至少匹配50%才算通过

        return ScoreResult(
            score=score,
            passed=passed,
            details={
                "matched": matched,
                "matched_indices": matched_indices,
                "total_keywords": len(self.keywords),
                "match_rate": f"{len(matched)}/{len(self.keywords)}"
            },
            reasoning=f"Matched {len(matched)}/{len(self.keywords)} keywords: {matched}"
        )

    def get_metric_name(self) -> str:
        return "keyword_match"
