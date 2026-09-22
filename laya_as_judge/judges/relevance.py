"""Answer relevance and query addressing judge."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseJudge
from ..types import EvaluationReport


class AnswerRelevanceJudge(BaseJudge):
    """Evaluates whether an answer directly addresses the user query.

    Answers three questions in a single forward pass:
    - answers_query (noul): Calibrated probability that the response answers user intent.
    - relevance_score (score): 0 to 3 ordinal rating.
    - conciseness (noul): Whether the answer is direct without conversational padding.
    """

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "answers_query": {
                "type": "noul",
                "instructions": "Does `answer` directly and adequately address the question asked in `query`?",
                "criteria": {
                    "false": "evasive, off-topic, or fails to answer the question",
                    "true": "directly answers the core question or intent",
                },
            },
            "relevance_score": {
                "type": "score",
                "instructions": "How relevant and complete is `answer` with respect to `query`?",
                "criteria": [
                    "completely irrelevant or wrong topic",
                    "partially relevant but misses key components",
                    "mostly answers the question with minor gaps",
                    "exemplary, complete, and directly on-point",
                ],
            },
            "is_concise": {
                "type": "noul",
                "instructions": "Is `answer` concise and free of unnecessary fluff or repetitive disclaimers?",
            },
        }

    def evaluate_response(
        self,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Evaluate a query-answer pair."""
        state = {
            "query": query,
            "answer": answer,
        }
        meta = metadata or {}
        meta["task"] = "answer_relevance"
        return self.evaluate(state, metadata=meta)
