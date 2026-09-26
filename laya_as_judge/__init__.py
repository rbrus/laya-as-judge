"""Laya-as-a-Judge: Fast, calibrated, token-free LLM & AI agent evaluation.

Uses Laya non-autoregressive typed decision models instead of slow, expensive,
and fragile LLM-as-a-Judge.
"""

from .batch import BatchEvaluator
from .benchmark import BenchmarkComparison, BenchmarkRunner
from .cascade import CascadedJudge
from .engine import BaseDecisionEngine, EmulatorBackend, get_engine
from .judges import (
    AgentTrajectoryJudge,
    AnswerRelevanceJudge,
    BaseJudge,
    CustomJudge,
    CustomJudgeBuilder,
    FaithfulnessJudge,
    InstructionAdherenceJudge,
    PairwiseComparisonJudge,
    SafetyGuardJudge,
)
from .types import (
    BatchEvaluationReport,
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    JudgementResult,
    NoulResult,
    ScoreResult,
)

__version__ = "0.2.0"

__all__ = [
    # Judges
    "BaseJudge",
    "FaithfulnessJudge",
    "AnswerRelevanceJudge",
    "InstructionAdherenceJudge",
    "SafetyGuardJudge",
    "PairwiseComparisonJudge",
    "AgentTrajectoryJudge",
    "CustomJudge",
    "CustomJudgeBuilder",
    # Architecture & Workflows
    "CascadedJudge",
    "BatchEvaluator",
    "BenchmarkRunner",
    "BenchmarkComparison",
    # Engine & Factory
    "get_engine",
    "BaseDecisionEngine",
    "EmulatorBackend",
    # Types & Reports
    "DecisionType",
    "JudgementResult",
    "ScoreResult",
    "ChoiceResult",
    "NoulResult",
    "EvaluationReport",
    "BatchEvaluationReport",
]
