"""Base decision engine interface and math helpers."""

from __future__ import annotations

import abc
import json
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Union[str, dict, list]) -> str:
    """Serialize any state object (string, dict, list) to text for decision model."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_criterion(value: Any) -> str:
    """Render one criterion value as text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict[str, Any]) -> List[str]:
    """Render option strings in canonical order. Noul is always [false, true]."""
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {k: None for k in crit}
        return [
            k if v in (None, "") else f"{k}: {render_criterion(v)}"
            for k, v in crit.items()
        ]
    if t == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit = crit.get("false")
    true_crit = crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def confidence_from_probs(probs: np.ndarray, num_options: int) -> float:
    """Calculate normalized Shannon entropy confidence: 1 - H(p) / log(k).

    Returns a calibrated confidence in [0.0, 1.0]:
    - Uniform distribution (complete uncertainty) -> 0.0
    - One-hot distribution (complete certainty) -> 1.0
    """
    if num_options < 2:
        return 1.0
    p = probs[:num_options]
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    max_entropy = math.log(num_options)
    conf = 1.0 - (entropy / max_entropy)
    return float(np.clip(conf, 0.0, 1.0))


def validate_question_spec(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a typed decision question definition."""
    if not isinstance(spec, dict):
        raise ValueError(f"Question '{name}' must be a dictionary")
    kind = spec.get("type")
    if kind not in QTYPES:
        raise ValueError(f"Question '{name}' has unknown type '{kind}'; expected 'choice', 'score', or 'noul'")
    if "instructions" not in spec:
        raise ValueError(f"Question '{name}' is missing 'instructions'")
    criteria = spec.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            if not all(isinstance(c, str) for c in criteria):
                raise ValueError(f"Choice criteria for '{name}' must be strings")
            if len(set(criteria)) != len(criteria):
                raise ValueError(f"Choice labels for '{name}' must be unique")
            criteria = {c: None for c in criteria}
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"Choice question '{name}' requires non-empty dictionary or list criteria")
    elif kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError(f"Score question '{name}' requires list criteria with at least 2 levels")
    elif kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"Noul question '{name}' criteria must be a dictionary with 'false'/'true' descriptions")
    
    return {
        "type": kind,
        "instructions": spec["instructions"],
        "criteria": criteria,
    }


class BaseDecisionEngine(abc.ABC):
    """Abstract base class for Laya decision execution backends."""

    def __init__(self, model_id: str = "convaiinnovations/laya"):
        self.model_id = model_id

    @abc.abstractmethod
    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute typed decision questions over state in a single forward pass.

        Args:
            state: The context/state to evaluate (e.g. prompt, response, RAG context).
            questions: Dict of question ID to question specification.

        Returns:
            Dictionary matching Laya output schema:
            {
                "model": str,
                "answers": {qid: {...}},
                "usage": {"input_tokens": int, "output_tokens": 0},
                "latency_ms": float,
            }
        """
        pass
