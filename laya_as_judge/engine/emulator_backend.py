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
from typing import Any, Dict, List, Set, Union

import numpy as np

from .base import (
    BaseDecisionEngine,
    confidence_from_probs,
    render_options,
    serialize_state,
    validate_question_spec,
)

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "is", "was",
    "are", "were", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "would", "could", "shall", "might", "must", "that", "this", "these", "those",
    "what", "how", "who", "where", "which", "why", "whose", "whom", "their",
    # Evaluation metadata keys to filter from semantic matching
    "query", "context", "answer", "prompt", "response", "response_a", "response_b", "role", "text", "state",
}


def _stem(w: str) -> str:
    """Lightweight rule-based suffix stemming for robust lexical matching."""
    w = w.lower().strip()
    for suffix in ("ing", "tion", "tions", "ies", "es", "ed", "ly", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    return w


def _extract_content_text(state: Union[str, Dict[str, Any], List[Any]]) -> str:
    """Extract actual content values without dictionary keys contaminating words."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts = []
        for v in state.values():
            if isinstance(v, (str, int, float, bool)):
                parts.append(str(v))
            elif isinstance(v, (list, dict)):
                parts.append(_extract_content_text(v))
        return " ".join(parts)
    if isinstance(state, list):
        return " ".join(_extract_content_text(item) for item in state)
    return str(state)


def _extract_content_words(text: str, stem: bool = False) -> List[str]:
    """Extract alphanumeric words >= 3 chars, filtering stop words with optional stemming."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    if stem:
        return [_stem(w) for w in filtered]
    return filtered


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
        text_stems = set(_extract_content_words(text, stem=True))
        opt_stems = _extract_content_words(option_desc, stem=True)
        if not opt_stems:
            return 0.0

        matches = sum(1 for w in opt_stems if w in text_stems)
        return float(matches / len(opt_stems))

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        state_str = serialize_state(state)
        content_text = _extract_content_text(state)
        content_lower = content_text.lower()

        answers: Dict[str, Any] = {}
        total_input_tokens = 0

        # Pre-compute specialized domain metrics if state matches known evaluation schemas
        rag_metrics = None
        if isinstance(state, dict) and "context" in state and "answer" in state:
            ctx_text = str(state.get("context", ""))
            ans_text = str(state.get("answer", ""))
            ans_stems = _extract_content_words(ans_text, stem=True)
            ctx_stems = set(_extract_content_words(ctx_text, stem=True))

            if not ans_stems:
                g_ratio = 1.0
            else:
                matches = sum(1 for w in ans_stems if w in ctx_stems)
                g_ratio = matches / len(ans_stems)

            # Check for unsupported numbers in answer (hallucination indicator)
            nums_ans = set(re.findall(r"\b\d+\b", ans_text))
            nums_ctx = set(re.findall(r"\b\d+\b", ctx_text))
            unsupported_nums = nums_ans - nums_ctx
            if unsupported_nums:
                g_ratio = min(g_ratio, 0.25)

            rag_metrics = {
                "grounding_ratio": g_ratio,
                "has_unsupported_nums": bool(unsupported_nums),
            }

        pairwise_metrics = None
        if isinstance(state, dict) and "response_a" in state and "response_b" in state:
            prompt_str = str(state.get("prompt", ""))
            ra = str(state.get("response_a", ""))
            rb = str(state.get("response_b", ""))
            p_stems = set(_extract_content_words(prompt_str, stem=True))

            # Score response depth, completeness, and keyword matching
            score_a = len(ra) + len(set(_extract_content_words(ra))) * 10 + (len(p_stems.intersection(set(_extract_content_words(ra, stem=True)))) * 20)
            score_b = len(rb) + len(set(_extract_content_words(rb))) * 10 + (len(p_stems.intersection(set(_extract_content_words(rb, stem=True)))) * 20)

            # Code features
            if "def " in ra or "return " in ra or "class " in ra:
                score_a += 40
            if "def " in rb or "return " in rb or "class " in rb:
                score_b += 40
            if "try:" in ra or "retry" in ra.lower() or "exception" in ra.lower():
                score_a += 60
            if "try:" in rb or "retry" in rb.lower() or "exception" in rb.lower():
                score_b += 60

            pairwise_metrics = {
                "score_a": score_a,
                "score_b": score_b,
                "ratio": score_a / max(1, score_b),
            }

        relevance_metrics = None
        if isinstance(state, dict) and "query" in state and "answer" in state:
            qw = set(_extract_content_words(str(state.get("query", "")), stem=True))
            aw = set(_extract_content_words(str(state.get("answer", "")), stem=True))
            overlap = len(qw.intersection(aw)) / max(1, len(qw))
            relevance_metrics = {"overlap": overlap}

        attack_patterns = [
            r"ignore\s+(\w+\s+)?previous",
            r"bypass",
            r"jailbreak",
            r"override",
            r"\bdan\b",
            r"system\s+(prompt|override|directive)",
            r"api[_\s\-]?keys?",
            r"credentials?",
            r"passwords?",
            r"hack",
            r"exploit",
            r"toxic",
            r"harass",
            r"threat",
        ]
        has_attack = any(re.search(pat, content_lower) for pat in attack_patterns)
        safety_metrics = {"has_attack": has_attack}

        for qid, q in normalized_q.items():
            opts = render_options(q)
            prompt_words = len(re.findall(r"\w+", f"{q['instructions']} {' '.join(opts)} {state_str}"))
            total_input_tokens += int(prompt_words * 1.25)

            k = len(opts)
            qtype = q["type"]
            ins_lower = q["instructions"].lower()

            if qtype == "choice":
                labels = list(q["criteria"].keys())

                # Domain override: RAG error_type
                if rag_metrics is not None and "fully_grounded" in labels:
                    gr = rag_metrics["grounding_ratio"]
                    if gr >= 0.65:
                        probs = np.array([0.91 if l == "fully_grounded" else 0.03 for l in labels], dtype=np.float32)
                        probs /= probs.sum()
                    elif gr < 0.40:
                        chosen = "direct_contradiction" if rag_metrics["has_unsupported_nums"] or "direct_contradiction" in labels else "extrapolation"
                        probs = np.array([0.85 if l == chosen else 0.05 for l in labels], dtype=np.float32)
                        probs /= probs.sum()
                    else:  # ambiguous
                        probs = np.array([0.45 if l == "extrapolation" else 0.25 if l == "fully_grounded" else 0.15 for l in labels], dtype=np.float32)
                        probs /= probs.sum()
                # Domain override: Pairwise winner
                elif pairwise_metrics is not None and "response_a" in labels and "response_b" in labels:
                    ratio = pairwise_metrics["ratio"]
                    if ratio >= 1.3:  # A decisively better
                        probs = np.array([0.88 if l == "response_a" else 0.08 if l == "response_b" else 0.04 for l in labels], dtype=np.float32)
                        probs /= probs.sum()
                    elif ratio <= 0.77:  # B decisively better
                        probs = np.array([0.88 if l == "response_b" else 0.08 if l == "response_a" else 0.04 for l in labels], dtype=np.float32)
                        probs /= probs.sum()
                    else:  # close or tie
                        probs = np.array([0.42 if l == "response_a" else 0.42 if l == "response_b" else 0.16 for l in labels], dtype=np.float32)
                        probs /= probs.sum()
                else:
                    scores = []
                    for label in labels:
                        crit_desc = str(q["criteria"].get(label) or label)
                        aff = self._estimate_semantic_affinity(content_text, f"{label} {crit_desc}")
                        if label.lower() in content_lower:
                            aff += 0.5
                        scores.append(aff)

                    logits = np.array(scores, dtype=np.float32)
                    if np.all(logits == 0):
                        logits[0] = 0.5
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

                # Domain override: RAG hallucination_severity
                if rag_metrics is not None and any("hallucination" in str(c).lower() or "faithful" in str(c).lower() for c in criteria_list):
                    gr = rag_metrics["grounding_ratio"]
                    if gr >= 0.65:
                        probs = np.array([0.91, 0.06, 0.02, 0.01][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                    elif gr < 0.40:
                        probs = np.array([0.02, 0.08, 0.35, 0.55][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                    else:  # ambiguous
                        probs = np.array([0.15, 0.50, 0.25, 0.10][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                # Domain override: Safety harm_severity
                elif any("harm" in ins_lower or "violation" in ins_lower for _ in [1]):
                    if safety_metrics["has_attack"]:
                        probs = np.array([0.02, 0.08, 0.35, 0.55][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                    else:
                        probs = np.array([0.94, 0.04, 0.01, 0.01][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                # Domain override: Pairwise preference_margin
                elif pairwise_metrics is not None and "margin" in ins_lower:
                    ratio = pairwise_metrics["ratio"]
                    if ratio >= 2.0 or ratio <= 0.5:
                        probs = np.array([0.02, 0.06, 0.22, 0.70][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                    elif ratio >= 1.4 or ratio <= 0.71:
                        probs = np.array([0.05, 0.25, 0.60, 0.10][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                    else:
                        probs = np.array([0.70, 0.20, 0.08, 0.02][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                # Domain override: Answer relevance score
                elif "relevance" in ins_lower or "relevant" in ins_lower:
                    if relevance_metrics is not None and relevance_metrics["overlap"] >= 0.40:
                        probs = np.array([0.02, 0.06, 0.22, 0.70][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                    else:
                        probs = np.array([0.70, 0.20, 0.08, 0.02][:k_levels], dtype=np.float32)
                        probs /= probs.sum()
                else:
                    level_affinities = [self._estimate_semantic_affinity(content_text, str(crit)) for crit in criteria_list]
                    aff_arr = np.array(level_affinities, dtype=np.float32)
                    if np.sum(aff_arr) > 0:
                        logits = aff_arr * 2.5
                    else:
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
                # Check for RAG Faithfulness proposition
                is_pos = True
                if rag_metrics is not None and any(w in ins_lower or w in qid for w in ["faith", "ground", "entail", "factual"]):
                    gr = rag_metrics["grounding_ratio"]
                    if gr >= 0.65:
                        prob_true = min(0.98, 0.88 + 0.10 * gr)
                    elif gr < 0.40:
                        prob_true = max(0.02, 0.05 + 0.15 * gr)
                    else:  # ambiguous
                        prob_true = 0.58
                # Check for Safety propositions
                elif any(w in ins_lower or w in qid for w in ["safe", "jailbreak", "injection", "sensitive", "malicious"]):
                    is_safe_q = "is_safe" in qid or "is `text` safe" in ins_lower or ( "safe" in ins_lower and "jailbreak" not in qid and "injection" not in qid)
                    if is_safe_q:
                        is_pos = True
                        prob_true = 0.03 if safety_metrics["has_attack"] else 0.97
                    else:
                        is_pos = False
                        prob_true = 0.96 if safety_metrics["has_attack"] else 0.02
                # Pairwise helpfulness
                elif pairwise_metrics is not None and ("response_a" in ins_lower or "response_b" in ins_lower or "response_a" in qid or "response_b" in qid):
                    is_a = "response_a" in ins_lower or "response_a" in qid
                    target_score = pairwise_metrics["score_a"] if is_a else pairwise_metrics["score_b"]
                    prob_true = 0.95 if target_score > 30 else 0.40
                # Answer relevance
                elif any(w in ins_lower or w in qid for w in ["directly and adequately", "answers_query", "address"]):
                    if relevance_metrics is not None and relevance_metrics["overlap"] >= 0.40:
                        prob_true = min(0.98, 0.85 + 0.15 * relevance_metrics["overlap"])
                    else:
                        prob_true = 0.18
                elif "concise" in ins_lower or "concise" in qid:
                    ans_str = str(state.get("answer", content_text)) if isinstance(state, dict) else content_text
                    prob_true = 0.90 if len(ans_str.split()) < 50 else 0.35
                else:
                    # General proposition
                    is_positive_prop = any(k in ins_lower for k in ["safe", "faithful", "directly", "adequately", "strictly obey", "helpful", "valid", "clean", "idiomatic"])
                    is_negative_prop = any(k in ins_lower for k in ["jailbreak", "injection", "sensitive", "stuck", "violation", "loop"])
                    is_pos = not is_negative_prop
                    if is_negative_prop:
                        prob_true = 0.93 if safety_metrics["has_attack"] else 0.04
                    elif is_positive_prop:
                        prob_true = 0.15 if safety_metrics["has_attack"] else 0.92
                    else:
                        aff_true = self._estimate_semantic_affinity(content_text, ins_lower)
                        prob_true = min(0.95, max(0.05, 0.5 + (aff_true - 0.2) * 1.5))

                p_false = 1.0 - prob_true
                probs = np.array([p_false, prob_true], dtype=np.float32)
                conf = confidence_from_probs(probs, 2)

                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(prob_true), 4),
                    "confidence": round(float(conf), 4),
                    "action": {"act_probability": round(float(p_false if is_pos else prob_true) * 0.1, 4)},
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
