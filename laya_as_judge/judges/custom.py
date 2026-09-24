"""Declarative builder for custom evaluation rubrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .base import BaseJudge
from ..engine import BaseDecisionEngine


class CustomJudge(BaseJudge):
    """A user-defined evaluation judge configured with custom questions and rubrics."""

    def __init__(
        self,
        name: str,
        questions_dict: Dict[str, Dict[str, Any]],
        engine: Optional[BaseDecisionEngine] = None,
        backend: str = "auto",
        model_id: Optional[str] = None,
        **engine_kwargs: Any,
    ):
        self.name = name
        self._questions = questions_dict
        super().__init__(engine=engine, backend=backend, model_id=model_id, **engine_kwargs)

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return self._questions

    @classmethod
    def builder(cls, name: str = "CustomJudge") -> CustomJudgeBuilder:
        """Create a builder for defining custom evaluation criteria."""
        return CustomJudgeBuilder(name)


class CustomJudgeBuilder:
    """Fluent builder for constructing custom evaluation rubrics."""

    def __init__(self, name: str):
        self.name = name
        self.questions: Dict[str, Dict[str, Any]] = {}

    def add_noul(
        self,
        name: str,
        instructions: str,
        false_desc: Optional[str] = None,
        true_desc: Optional[str] = None,
    ) -> CustomJudgeBuilder:
        """Add a boolean proposition question returning calibrated P(true)."""
        criteria = None
        if false_desc or true_desc:
            criteria = {
                "false": false_desc or "the proposition is false",
                "true": true_desc or "the proposition is true",
            }
        self.questions[name] = {
            "type": "noul",
            "instructions": instructions,
            "criteria": criteria,
        }
        return self

    def add_score(
        self,
        name: str,
        instructions: str,
        criteria: List[str],
    ) -> CustomJudgeBuilder:
        """Add an ordinal rubric scoring question returning expected score and distribution."""
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("Criteria for score question must be a list of at least 2 levels")
        self.questions[name] = {
            "type": "score",
            "instructions": instructions,
            "criteria": criteria,
        }
        return self

    def add_choice(
        self,
        name: str,
        instructions: str,
        criteria: Union[List[str], Dict[str, Optional[str]]],
    ) -> CustomJudgeBuilder:
        """Add a categorical choice question returning chosen label and probabilities."""
        if isinstance(criteria, list):
            crit_dict = {k: None for k in criteria}
        elif isinstance(criteria, dict):
            crit_dict = criteria
        else:
            raise ValueError("Criteria for choice question must be a list or dict")

        self.questions[name] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": crit_dict,
        }
        return self

    def build(
        self,
        engine: Optional[BaseDecisionEngine] = None,
        backend: str = "auto",
        model_id: Optional[str] = None,
        **engine_kwargs: Any,
    ) -> CustomJudge:
        """Build and instantiate the CustomJudge."""
        if not self.questions:
            raise ValueError("Cannot build CustomJudge without at least one question")
        return CustomJudge(
            name=self.name,
            questions_dict=self.questions,
            engine=engine,
            backend=backend,
            model_id=model_id,
            **engine_kwargs,
        )
