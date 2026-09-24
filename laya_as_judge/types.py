"""Type definitions and data models for Laya-as-a-Judge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class DecisionType(str, Enum):
    """The three fundamental typed decision primitives of Laya."""
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass
class JudgementResult:
    """Base evaluation result for a single question/criterion."""
    name: str
    confidence: float
    decision_type: DecisionType = DecisionType.CHOICE
    raw_answer: Dict[str, Any] = field(default_factory=dict)
    action_probability: Optional[float] = None

    @property
    def is_high_confidence(self) -> bool:
        """Returns True if confidence exceeds default high-confidence threshold (0.75)."""
        return self.confidence >= 0.75

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "decision_type": self.decision_type.value,
            "confidence": self.confidence,
            "action_probability": self.action_probability,
            **self._specific_dict(),
        }

    def _specific_dict(self) -> Dict[str, Any]:
        return {}


@dataclass
class ScoreResult(JudgementResult):
    """Result for an ordinal rubric evaluation ('score' primitive).

    Score is continuous expected value: E[S] = sum(i * P(level_i)).
    Provides calibrated probability distribution across all discrete levels.
    """
    decision_type: DecisionType = DecisionType.SCORE
    score: float = 0.0
    legend: Dict[str, str] = field(default_factory=dict)
    level_probabilities: Dict[str, float] = field(default_factory=dict)

    def _specific_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "legend": self.legend,
            "level_probabilities": self.level_probabilities,
        }


@dataclass
class ChoiceResult(JudgementResult):
    """Result for a categorical decision ('choice' primitive).

    Picks the most probable option among discrete candidates and provides
    the complete calibrated probability distribution.
    """
    decision_type: DecisionType = DecisionType.CHOICE
    choice: str = ""
    probabilities: Dict[str, float] = field(default_factory=dict)

    def _specific_dict(self) -> Dict[str, Any]:
        return {
            "choice": self.choice,
            "probabilities": self.probabilities,
        }


@dataclass
class NoulResult(JudgementResult):
    """Result for a boolean proposition evaluation ('noul' primitive).

    Returns calibrated P(true) for whether the proposition holds,
    calibrated against strictly proper scoring rules via RLCD.
    """
    decision_type: DecisionType = DecisionType.NOUL
    holds: bool = False
    prob_true: float = 0.0
    threshold: float = 0.5

    def _specific_dict(self) -> Dict[str, Any]:
        return {
            "holds": self.holds,
            "prob_true": self.prob_true,
            "threshold": self.threshold,
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for an input state across all questions."""
    state: Union[str, Dict[str, Any], List[Any]]
    judgements: Dict[str, JudgementResult]
    latency_ms: float
    input_tokens: int
    output_tokens: int = 0
    model: str = "laya-decision-engine"
    backend: str = "auto"
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_score(self, name: str) -> float:
        """Helper to get continuous score for a score judgement."""
        j = self.judgements.get(name)
        if isinstance(j, ScoreResult):
            return j.score
        raise KeyError(f"Judgement '{name}' is not a ScoreResult (is {type(j)})")

    def get_bool(self, name: str) -> bool:
        """Helper to get boolean verdict for a noul judgement."""
        j = self.judgements.get(name)
        if isinstance(j, NoulResult):
            return j.holds
        raise KeyError(f"Judgement '{name}' is not a NoulResult (is {type(j)})")

    def get_choice(self, name: str) -> str:
        """Helper to get chosen option for a choice judgement."""
        j = self.judgements.get(name)
        if isinstance(j, ChoiceResult):
            return j.choice
        raise KeyError(f"Judgement '{name}' is not a ChoiceResult (is {type(j)})")

    def get_confidence(self, name: str) -> float:
        """Helper to get confidence for a specific judgement."""
        return self.judgements[name].confidence

    @property
    def average_confidence(self) -> float:
        """Average confidence across all evaluated criteria."""
        if not self.judgements:
            return 0.0
        return sum(j.confidence for j in self.judgements.values()) / len(self.judgements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "backend": self.backend,
            "cost_usd": self.cost_usd,
            "judgements": {k: v.to_dict() for k, v in self.judgements.items()},
            "metadata": self.metadata,
        }


@dataclass
class BatchEvaluationReport:
    """Dataset-level evaluation summary across multiple items."""
    reports: List[EvaluationReport]
    total_latency_ms: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    estimated_llm_cost_saved_usd: float = 0.0
    estimated_time_saved_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": len(self.reports),
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": self.average_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "estimated_llm_cost_saved_usd": self.estimated_llm_cost_saved_usd,
            "estimated_time_saved_seconds": self.estimated_time_saved_seconds,
            "reports": [r.to_dict() for r in self.reports],
        }
