"""维度注册表与常量定义。"""

from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
from benchmark.adapters.frontcode_adapter import FrontCodeAdapter
from benchmark.adapters.probe_adapter import ProbeAdapter
from benchmark.adapters.math_adapter import MATHAdapter
from benchmark.core.evaluator import SingleTurnEvaluator
from benchmark.scorers.backend import create_backend_composite
from benchmark.scorers.frontend import create_frontend_composite
from benchmark.scorers.probe_scorer import ProbeScorer
from benchmark.scorers.reasoning import create_reasoning_composite

DIMENSION_REGISTRY: dict[str, tuple] = {
    "reasoning": (MATHAdapter, create_reasoning_composite, SingleTurnEvaluator),
    "backend-dev": (BigCodeBenchAdapter, create_backend_composite, SingleTurnEvaluator),
    "frontend-dev": (FrontCodeAdapter, create_frontend_composite, SingleTurnEvaluator),
    "probe": (ProbeAdapter, lambda: [(1.0, ProbeScorer())], SingleTurnEvaluator),
}

DATASET_REGISTRY: dict[str, str] = {
    "reasoning": "math",
    "backend-dev": "bigcodebench",
    "frontend-dev": "frontcode",
    "probe": "probe",
}

THINKING_SYSTEM_MESSAGE = (
    "你是一个高效助手。根据任务难度自适应调节思考深度：\n"
    "- 简单任务（如选择题、事实查询）：直接回答，简短推理即可\n"
    "- 中等任务（如数学计算、代码编写）：适当推理，重点关注核心逻辑\n"
    "- 复杂任务（如系统设计、多步证明）：审慎推理，但避免重复验证已知结论\n"
    "如果已经找到答案，立即停止推理并给出最终结果。"
)
