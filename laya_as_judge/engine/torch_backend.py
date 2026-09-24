"""PyTorch / Transformers backend for Linux and NVIDIA/AMD/CPU hardware."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import (
    BaseDecisionEngine,
    confidence_from_probs,
    render_options,
    serialize_state,
    validate_question_spec,
)


class TorchBackend(BaseDecisionEngine):
    """EXPERIMENTAL, INCOMPLETE PyTorch backend for Laya typed decision models.

    Status: this backend downloads the checkpoint and loads the tokenizer, but it
    does not yet load the encoder or decision-head weights. Every question
    therefore receives a uniform probability distribution (zero confidence).
    It exists as a scaffold for a future port and is never auto-selected;
    use the MLX backend (Apple Silicon) for real Laya inference.
    """

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        torch_dtype: str = "float16",
    ):
        super().__init__(model_id=model_id)
        warnings.warn(
            "TorchBackend is an incomplete scaffold: it does not run the Laya network "
            "and returns uniform (uninformative) distributions for every question.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TorchBackend requires 'torch', 'transformers', and 'huggingface-hub'. "
                "Install with: pip install 'laya-as-judge[torch]'"
            ) from e

        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Download checkpoint directory if needed
        local_path = Path(model_id).expanduser()
        if not local_path.exists():
            local_path = Path(snapshot_download(model_id))
        self.model_dir = local_path

        # Load configs
        cfg_path = self.model_dir / "rl_agent_config.json"
        if cfg_path.is_file():
            self.cfg = json.loads(cfg_path.read_text())
        else:
            self.cfg = {"temperature": [1.0, 1.0, 1.0]}

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir / "tokenizer"))

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute PyTorch forward pass."""
        t0 = time.perf_counter()
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        state_str = serialize_state(state)

        answers = {}
        total_input_tokens = 0

        # Run each question forward
        for qid, q in normalized_q.items():
            opts = render_options(q)
            prompt = f"{q['type']} question: {q['instructions']} Options: {'; '.join(opts)} State: {state_str}"
            tokens = self.tokenizer(prompt, truncation=True, max_length=512)
            total_input_tokens += len(tokens["input_ids"])

            # Compute logits across options
            k = len(opts)
            # Default temperature
            t_val = self.cfg.get("temperature", [1.0, 1.0, 1.0])[0]
            # Placeholder: the Laya encoder/heads are not loaded yet, so logits are zero
            raw_logits = np.zeros(k, dtype=np.float32)
            # Softmax
            scaled = raw_logits / max(0.1, t_val)
            probs = np.exp(scaled - np.max(scaled))
            probs /= np.sum(probs)

            conf = confidence_from_probs(probs, k)
            ans: Dict[str, Any] = {
                "type": q["type"],
                "confidence": round(conf, 4),
                "action": {"act_probability": 0.05},
            }

            if q["type"] == "choice":
                labels = list(q["criteria"].keys())
                ans["choice"] = labels[int(np.argmax(probs))]
                ans["probabilities"] = {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}
            elif q["type"] == "score":
                expected_score = float(np.sum(np.arange(k) * probs))
                ans["score"] = round(expected_score, 4)
                ans["legend"] = {str(i): str(c) for i, c in enumerate(q["criteria"])}
                ans["probabilities"] = {str(i): round(float(p), 4) for i, p in enumerate(probs)}
            else:
                p_true = float(probs[1]) if k >= 2 else 0.5
                ans["noul"] = round(p_true, 4)
                ans["confidence"] = round(max(p_true, 1.0 - p_true), 4)

            answers[qid] = ans

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model": f"laya-torch ({self.model_id})",
            "answers": answers,
            "usage": {"input_tokens": total_input_tokens, "output_tokens": 0},
            "latency_ms": round(latency_ms, 2),
        }
