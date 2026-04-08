"""System-Architecture 评分器模块."""
from benchmark.scorers.system_architecture.answer_correctness import AnswerCorrectnessScorer
from benchmark.scorers.system_architecture.option_analysis import OptionAnalysisScorer
from benchmark.scorers.system_architecture.reasoning_completeness import ReasoningCompletenessScorer
from benchmark.scorers.system_architecture.reasoning_confidence import ReasoningConfidenceScorer
from benchmark.scorers.system_architecture.subject_adaptation import SubjectAdaptationScorer


def create_sysarch_composite() -> list[tuple[float, object]]:
    return [
        (0.30, AnswerCorrectnessScorer()),
        (0.25, ReasoningCompletenessScorer()),
        (0.20, OptionAnalysisScorer()),
        (0.15, ReasoningConfidenceScorer()),
        (0.10, SubjectAdaptationScorer()),
    ]
