"""Pairwise model comparison judge (Model A vs Model B A/B evaluation)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseJudge
from ..types import EvaluationReport


class PairwiseComparisonJudge(BaseJudge):
    """Evaluates two candidate responses against a prompt to determine the winner.

    Evaluates:
    - winner (choice): 'response_a', 'response_b', or 'tie'.
    - margin (score): 0 to 3 scale (from indistinguishable to decisive superiority).
    - a_is_helpful (noul): whether response A was helpful.
    - b_is_helpful (noul): whether response B was helpful.
    """

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "winner": {
                "type": "choice",
                "instructions": "Which response is better suited as an answer to `prompt`?",
                "criteria": {
                    "response_a": "Response A is superior in quality, accuracy, or helpfulness",
                    "response_b": "Response B is superior in quality, accuracy, or helpfulness",
                    "tie": "Both responses are roughly equal in quality or both fail equally",
                },
            },
            "preference_margin": {
                "type": "score",
                "instructions": "What is the margin of superiority between the two responses?",
                "criteria": [
                    "negligible: essentially identical quality",
                    "slight: small preference based on style or phrasing",
                    "clear: noticeable advantage in correctness or completeness",
                    "decisive: one response is excellent while the other is broken or wrong",
                ],
            },
            "response_a_helpful": {
                "type": "noul",
                "instructions": "Is `response_a` helpful, coherent, and factually plausible?",
            },
            "response_b_helpful": {
                "type": "noul",
                "instructions": "Is `response_b` helpful, coherent, and factually plausible?",
            },
        }

    def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Compare two competing responses."""
        state = {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
        }
        meta = metadata or {}
        meta["task"] = "pairwise_comparison"
        return self.evaluate(state, metadata=meta)
