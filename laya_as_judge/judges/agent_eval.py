"""Agent trajectory and multi-step tool execution evaluator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseJudge
from ..types import EvaluationReport


class AgentTrajectoryJudge(BaseJudge):
    """Evaluates agent multi-step actions, tool calls, and execution progress.

    Evaluates:
    - tool_call_valid (noul): whether the tool call and arguments are sensible.
    - progress_toward_goal (score): 0 to 3 scale.
    - is_looping_or_stuck (noul): whether the agent is in an unrecoverable loop.
    - next_step_assessment (choice): 'proceed', 'retry_tool', 'ask_user', 'abort'.
    """

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "tool_call_valid": {
                "type": "noul",
                "instructions": (
                    "Given `goal` and `history`, is the latest action/tool call in `action` "
                    "logical and appropriate?"
                ),
            },
            "progress_score": {
                "type": "score",
                "instructions": "How much progress has the agent made toward solving `goal`?",
                "criteria": [
                    "no progress: stuck, confused, or repeating previous errors",
                    "early progress: initial setup or information gathering completed",
                    "substantial progress: core steps executed, near completion",
                    "goal achieved: successfully fulfilled all requirements",
                ],
            },
            "is_stuck": {
                "type": "noul",
                "instructions": "Is the agent stuck in a repetitive loop or repeating a failed command?",
            },
            "recommended_action": {
                "type": "choice",
                "instructions": "What should the supervisory system do with this agent step?",
                "criteria": {
                    "proceed": "allow the agent to continue executing normally",
                    "correct_parameters": "intercept and fix invalid arguments",
                    "replan": "force the agent to reconsider its overall plan",
                    "escalate_to_human": "pause execution and request user guidance",
                },
            },
        }

    def evaluate_step(
        self,
        goal: str,
        history: List[Dict[str, Any]],
        action: Dict[str, Any],
        observation: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Evaluate a single agent trajectory step."""
        state = {
            "goal": goal,
            "history": history,
            "action": action,
            "observation": observation,
        }
        meta = metadata or {}
        meta["task"] = "agent_trajectory_eval"
        return self.evaluate(state, metadata=meta)
