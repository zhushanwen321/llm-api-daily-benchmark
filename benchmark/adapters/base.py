"""数据集适配器基类。

所有数据集适配器必须继承此基类并实现 load/validate/get_dimension 方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from benchmark.models.schemas import TaskDefinition


class DatasetAdapter(ABC):
    """数据集适配器抽象基类。

    子类必须实现：
    - load(): 从数据源加载任务列表
    - validate(): 验证单个任务格式是否合法
    - get_dimension(): 返回适配器对应的评测维度名称
    """

    @abstractmethod
    def load(self, path: str = "") -> List[TaskDefinition]:
        """加载数据集，返回任务定义列表。

        Args:
            path: 数据集路径。为空时使用默认缓存路径。

        Returns:
            TaskDefinition 列表。
        """

    @abstractmethod
    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式是否合法。

        Args:
            task: 待验证的任务定义。

        Returns:
            True 表示格式合法。
        """

    @abstractmethod
    def get_dimension(self) -> str:
        """返回此适配器对应的评测维度名称。"""
