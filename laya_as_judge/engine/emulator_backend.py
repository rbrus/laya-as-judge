"""Heuristic stand-in for the Laya model, for local development, CI, and tests.

IMPORTANT: this is NOT the Laya model and does not load any weights. It mimics
Laya's *output schema* (choice / score / noul answers with probabilities and a
normalized-entropy confidence, 0 output tokens) using simple text heuristics:

- choice / score: word overlap between the state text and each option's rubric
  text, pushed through a softmax;
- noul: a fixed list of regex patterns (e.g. "bypass", "system prompt",
  "password") combined with keyword matching on the question wording, mapped to
  hard-coded probabilities.

Its verdicts are therefore frequently wrong and its probabilities are not
calibrated. Use it to exercise pipelines, schemas and plumbing, never to draw
conclusions about the quality of the text being judged.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Union

import numpy as np

from .base import (
    BaseDecisionEngine,
    confidence_from_probs,
    render_options,
    serialize_state,
    validate_question_spec,
)


class EmulatorBackend(BaseDecisionEngine):
    """Keyword/regex heuristic emulator backend (no model weights).

    Requires no GPU or ML dependencies. Useful for testing, offline CI, and
    exercising the Laya-as-a-Judge API; its outputs are not real model verdicts.
    """

    def __init__(self, model_id: str = "convaiinnovations/laya-emulator", seed: int = 42):
        super().__init__(model_id=model_id)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _estimate_semantic_affinity(self, text: str, option_desc: str) -> float:
        """Estimate semantic affinity between state text and option rubric."""
        text_lower = text.lower()
        opt_lower = option_desc.lower()
        words = set(re.findall(r"\w+", opt_lower))
        if not words:
            return 0.0

        matches = sum(1 for w in words if len(w) > 3 and w in text_lower)
        affinity = matches / max(1, len(words))
        return float(affinity)

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        state_str = serialize_state(state)
        state_lower = state_str.lower()

        answers: Dict[str, Any] = {}
        total_input_tokens = 0

        for qid, q in normalized_q.items():
            opts = render_options(q)
            # Estimate input tokens: ~1.3 tokens per word
            prompt_words = len(re.findall(r"\w+", f"{q['instructions']} {' '.join(opts)} {state_str}"))
            total_input_tokens += int(prompt_words * 1.25)

            k = len(opts)
            qtype = q["type"]

            if qtype == "choice":
                labels = list(q["criteria"].keys())
                scores = []
                for label in labels:
                    crit_desc = str(q["criteria"].get(label) or label)
                    aff = self._estimate_semantic_affinity(state_str, f"{label} {crit_desc}")
                    # Special domain heuristics
                    if label in state_lower:
                        aff += 0.5
                    scores.append(aff)

                logits = np.array(scores, dtype=np.float32)
                # If all zero, give slight preference to the first or default
                if np.all(logits == 0):
                    logits[0] = 0.5
                # Temperature scaling & softmax
                exp_logits = np.exp(logits * 3.0 - np.max(logits * 3.0))
                probs = exp_logits / np.sum(exp_logits)
                best_idx = int(np.argmax(probs))

                conf = confidence_from_probs(probs, k)
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[best_idx],
                    "confidence": round(conf, 4),
                    "probabilities": {label: round(float(p), 4) for label, p in zip(labels, probs)},
                    "action": {"act_probability": round(float(probs[best_idx] * 0.1), 4)},
                }

            elif qtype == "score":
                criteria_list = q["criteria"]
                k_levels = len(criteria_list)
                # Compute progressive affinity
                level_affinities = []
                for i, crit in enumerate(criteria_list):
                    aff = self._estimate_semantic_affinity(state_str, str(crit))
                    # check for level indicators or positive/negative keywords
                    level_affinities.append(aff)

                # Turn overlap scores into a softmax distribution (heuristic, not calibrated)
                aff_arr = np.array(level_affinities, dtype=np.float32)
                if np.sum(aff_arr) > 0:
                    logits = aff_arr * 2.5
                else:
                    # Default healthy moderate-to-high distribution for general responses
                    logits = np.linspace(0.5, 2.0, k_levels)

                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                expected_score = float(np.sum(np.arange(k_levels) * probs))
                conf = confidence_from_probs(probs, k_levels)

                answers[qid] = {
                    "type": "score",
                    "score": round(expected_score, 4),
                    "confidence": round(conf, 4),
                    "legend": {str(i): str(c) for i, c in enumerate(criteria_list)},
                    "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probs)},
                    "action": {"act_probability": round(float((1.0 - (expected_score / (k_levels - 1))) * 0.2), 4)},
                }

            else:  # noul (boolean)
                # Check whether proposition holds
                ins = q["instructions"].lower()
                
                # Malicious or violation indicator patterns
                attack_patterns = [
                    r"ignore\s+(\w+\s+)?previous",
                    r"bypass",
                    r"jailbreak",
                    r"override",
                    r"\bdan\b",
                    r"system\s+prompt",
                    r"api[_\s\-]?keys?",
                    r"credentials?",
                    r"passwords?",
                    r"fabricated",
                    r"contrary\s+to",
                    r"contradict",
                    r"toxic",
                    r"harass",
                    r"threat",
                ]
                has_attack = any(re.search(pat, state_lower) for pat in attack_patterns)
                
                # Positive safety/faithfulness questions (expect true if benign)
                is_positive_prop = any(k in ins for k in ["is `text` safe", "is safe", "faithful", "directly and adequately address", "strictly obey", "helpful"])
                # Negative risk questions (expect false if benign)
                is_negative_prop = any(k in ins for k in ["jailbreak", "prompt injection", "sensitive data", "stuck", "violation"])

                if is_positive_prop:
                    prob_true = 0.15 if has_attack else 0.92
                elif is_negative_prop:
                    prob_true = 0.93 if has_attack else 0.04
                else:
                    aff_true = self._estimate_semantic_affinity(state_str, ins)
                    prob_true = min(0.95, max(0.05, 0.5 + (aff_true - 0.2) * 1.5))

                p_false = 1.0 - prob_true
                probs = np.array([p_false, prob_true], dtype=np.float32)
                conf = max(prob_true, p_false)

                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(prob_true), 4),
                    "confidence": round(float(conf), 4),
                    "action": {"act_probability": round(float(prob_true if is_negative_prop else p_false) * 0.1, 4)},
                }

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model": "laya-calibrated-emulator",
            "answers": answers,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": 0,  # Fundamental Laya guarantee!
            },
            "latency_ms": round(latency_ms, 2),
        }
