"""PyTorch / Transformers / Upstream Laya backend for Linux and NVIDIA/AMD/CPU hardware."""

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
    """PyTorch / Upstream Laya backend for Linux and NVIDIA/AMD/CPU hardware.

    Supports the official upstream 'laya' package (from PyPI) for full neural inference.
    If 'laya' is not installed, falls back to a PyTorch tokenizer scaffold that emits a
    warning and returns uniform distributions.
    """

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        torch_dtype: str = "float16",
        preload: bool = True,
    ):
        super().__init__(model_id=model_id)

        # 1. Prefer official upstream 'laya' package if installed
        try:
            import laya
            self._laya = laya
            if hasattr(laya, "load"):
                self._agent = laya.load(model_id, device=device)
            elif hasattr(laya, "Router"):
                self._agent = laya.Router(device=device, preload=preload)
            else:
                self._agent = laya.Agent(model_id, device=device)
            self._is_upstream_laya = True
            return
        except ImportError:
            self._is_upstream_laya = False

        warnings.warn(
            "TorchBackend fallback without 'laya' installed is an incomplete scaffold: "
            "it does not run the decision network and returns uniform distributions for every question. "
            "Install 'laya' (pip install laya) for full inference.",
            RuntimeWarning,
            stacklevel=2,
        )

        # 2. Fallback to PyTorch & Transformers
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TorchBackend requires the official 'laya' runtime (pip install laya) "
                "or 'torch' and 'transformers'. Install with: pip install 'laya-as-judge[torch]'"
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
        """Execute decision questions forward pass."""
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        t0 = time.perf_counter()

        if self._is_upstream_laya:
            raw_result = self._agent.predict(state, normalized_q)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raw_result["latency_ms"] = round(latency_ms, 2)
            return raw_result

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
            t_val = self.cfg.get("temperature", [1.0, 1.0, 1.0])[0]
            # Placeholder: without 'laya', custom encoder/heads are not loaded, so logits are uniform
            raw_logits = np.zeros(k, dtype=np.float32)
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
                ans["confidence"] = round(confidence_from_probs(np.array([1.0 - p_true, p_true], dtype=np.float32), 2), 4)

            answers[qid] = ans

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model": f"laya-torch ({self.model_id})",
            "answers": answers,
            "usage": {"input_tokens": total_input_tokens, "output_tokens": 0},
            "latency_ms": round(latency_ms, 2),
        }
