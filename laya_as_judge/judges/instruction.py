"""Instruction adherence and constraint satisfaction judge."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseJudge
from ..types import EvaluationReport


class InstructionAdherenceJudge(BaseJudge):
    """Evaluates whether an AI response strictly follows negative and positive constraints.

    Answers three questions in a single forward pass:
    - follows_all (noul): P(true) that all constraints were satisfied.
    - adherence_score (score): 0 to 3 scale.
    - violation_type (choice): Categorization of failure mode.
    """

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "follows_all": {
                "type": "noul",
                "instructions": (
                    "Does `response` strictly obey all format, style, and negative constraints "
                    "specified in `instructions`?"
                ),
            },
            "adherence_score": {
                "type": "score",
                "instructions": "Rate how faithfully `response` adheres to the instructions and constraints:",
                "criteria": [
                    "major violation: ignored key constraints or tone completely",
                    "moderate violation: missed negative constraint or format requirement",
                    "minor violation: followed main rules but had small formatting blemish",
                    "perfect adherence: flawlessly adhered to all rules and constraints",
                ],
            },
            "violation_type": {
                "type": "choice",
                "instructions": "If a constraint was violated in `response`, what type of violation occurred?",
                "criteria": {
                    "no_violation": "adhered to all instructions cleanly",
                    "negative_constraint": "did something explicitly forbidden (e.g. 'do not mention X')",
                    "formatting_defect": "failed requested format (e.g. JSON, markdown, bullet count)",
                    "length_violation": "exceeded or fell short of requested length or word count",
                    "unnecessary_refusal": "refused a benign request unnecessarily",
                },
            },
        }

    def evaluate_adherence(
        self,
        instructions: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Evaluate instruction following."""
        state = {
            "instructions": instructions,
            "response": response,
        }
        meta = metadata or {}
        meta["task"] = "instruction_adherence"
        return self.evaluate(state, metadata=meta)
