"""Faithfulness and hallucination judge for RAG and QA pipelines."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseJudge
from ..types import EvaluationReport


class FaithfulnessJudge(BaseJudge):
    """Evaluates whether an answer is grounded in the retrieved context without hallucinations.

    Answers three questions in a single forward pass:
    - is_faithful (noul): Calibrated probability that all claims are grounded in context.
    - hallucination_severity (score): Ordinal severity from 0 (none) to 3 (critical).
    - error_type (choice): Categorization of factual discrepancy.
    """

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "is_faithful": {
                "type": "noul",
                "instructions": (
                    "Does `answer` contain only factual claims that are directly supported "
                    "or clearly entailed by `context`?"
                ),
                "criteria": {
                    "false": "contains fabricated facts, hallucinations, or unverified claims",
                    "true": "entirely grounded in and supported by the context",
                },
            },
            "hallucination_severity": {
                "type": "score",
                "instructions": "Rate the severity of unsupported claims or hallucinations in `answer`:",
                "criteria": [
                    "none: completely faithful, zero hallucinations",
                    "minor: slight extrapolation or harmless unverified detail",
                    "moderate: noticeable factual error or unsupported key fact",
                    "severe: critical fabrication, completely contrary to context",
                ],
            },
            "error_type": {
                "type": "choice",
                "instructions": "What is the primary category of factual alignment in `answer`?",
                "criteria": {
                    "fully_grounded": "all facts directly supported by context",
                    "extrapolation": "answer adds plausible external details not in context",
                    "direct_contradiction": "answer directly contradicts statements in context",
                    "entity_confusion": "confuses people, dates, numbers, or specific entities",
                },
            },
        }

    def evaluate_rag(
        self,
        query: str,
        context: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Evaluate a RAG triplet: query, retrieved context, and generated answer."""
        state = {
            "query": query,
            "context": context,
            "answer": answer,
        }
        meta = metadata or {}
        meta["task"] = "rag_faithfulness"
        return self.evaluate(state, metadata=meta)
