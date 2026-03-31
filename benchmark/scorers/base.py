"""评分器基类。

所有评分器必须继承 BaseScorer 并实现 score/get_metric_name 方法。
评分结果统一使用 benchmark.models.schemas.ScoreResult。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.models.schemas import ScoreResult, TaskDefinition


class BaseScorer(ABC):
    """评分器抽象基类。

    子类必须实现：
    - score(): 对模型输出进行评分
    - get_metric_name(): 返回指标名称
    """

    @abstractmethod
    def score(
        self,
        model_output: str,
        expected: str,
        task: TaskDefinition,
    ) -> ScoreResult:
        """对模型输出进行评分。

        Args:
            model_output: 模型生成的文本/代码。
            expected: 期望输出（答案或空字符串）。
            task: 原始任务定义。

        Returns:
            ScoreResult 包含分数、是否通过、详情、理由。
        """

    @abstractmethod
    def get_metric_name(self) -> str:
        """返回此评分器的指标名称（如 exact_match, execution）。"""
