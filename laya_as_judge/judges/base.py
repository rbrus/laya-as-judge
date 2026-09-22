"""Base judge class and result transformation logic."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union

from ..engine import BaseDecisionEngine, get_engine
from ..types import (
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    JudgementResult,
    NoulResult,
    ScoreResult,
)


class BaseJudge(abc.ABC):
    """Abstract base class for all Laya-as-a-Judge evaluators."""

    def __init__(
        self,
        engine: Optional[BaseDecisionEngine] = None,
        backend: str = "auto",
        model_id: Optional[str] = None,
        **engine_kwargs: Any,
    ):
        if engine is not None:
            self.engine = engine
        else:
            self.engine = get_engine(backend=backend, model_id=model_id, **engine_kwargs)

    @property
    @abc.abstractmethod
    def questions(self) -> Dict[str, Dict[str, Any]]:
        """Dictionary of question specifications evaluated by this judge."""
        pass

    def evaluate(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Run all typed evaluation questions on the provided state.

        Args:
            state: The state / context / prompt to evaluate.
            metadata: Optional additional metadata to store in report.

        Returns:
            EvaluationReport containing typed results for every question.
        """
        raw_result = self.engine.predict(state, self.questions)
        return self._format_report(state, raw_result, metadata or {})

    def _format_report(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        raw_result: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> EvaluationReport:
        """Convert raw Laya answers into strictly typed JudgementResults."""
        answers = raw_result.get("answers", {})
        judgements: Dict[str, JudgementResult] = {}

        for qid, qdef in self.questions.items():
            ans = answers.get(qid)
            if ans is None:
                continue

            qtype = qdef["type"]
            conf = float(ans.get("confidence", 0.5))
            act_prob = ans.get("action", {}).get("act_probability")

            if qtype == "choice":
                judgements[qid] = ChoiceResult(
                    name=qid,
                    decision_type=DecisionType.CHOICE,
                    confidence=conf,
                    choice=ans.get("choice", ""),
                    probabilities=ans.get("probabilities", {}),
                    action_probability=act_prob,
                    raw_answer=ans,
                )
            elif qtype == "score":
                judgements[qid] = ScoreResult(
                    name=qid,
                    decision_type=DecisionType.SCORE,
                    confidence=conf,
                    score=float(ans.get("score", 0.0)),
                    legend=ans.get("legend", {}),
                    level_probabilities=ans.get("probabilities", {}),
                    action_probability=act_prob,
                    raw_answer=ans,
                )
            elif qtype == "noul":
                prob_true = float(ans.get("noul", 0.5))
                judgements[qid] = NoulResult(
                    name=qid,
                    decision_type=DecisionType.NOUL,
                    confidence=conf,
                    holds=(prob_true >= 0.5),
                    prob_true=prob_true,
                    threshold=0.5,
                    action_probability=act_prob,
                    raw_answer=ans,
                )

        usage = raw_result.get("usage", {})
        latency_ms = raw_result.get("latency_ms", 0.0)

        return EvaluationReport(
            state=state,
            judgements=judgements,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),  # Laya guarantee: 0
            model=raw_result.get("model", self.engine.model_id),
            backend=type(self.engine).__name__,
            cost_usd=0.0,  # Zero cloud token cost!
            metadata=metadata,
        )
