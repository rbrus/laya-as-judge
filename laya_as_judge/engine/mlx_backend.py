"""MLX backend for Apple Silicon hardware acceleration."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Union

from .base import BaseDecisionEngine, validate_question_spec


class MLXBackend(BaseDecisionEngine):
    """Executes Laya models natively on Apple Silicon GPU/ANE using MLX.

    Achieves sub-15ms end-to-end latency for typed decisions with zero output tokens.
    """

    def __init__(
        self,
        model_id: str = "aac6fef/laya-mlx",
        device: str = "gpu",
        dtype: str = "float16",
        batch_size: int = 16,
    ):
        super().__init__(model_id=model_id)
        try:
            import laya_mlx as laya
            self._agent = laya.load(model_id, device=device, dtype=dtype, batch_size=batch_size)
        except ImportError as e:
            raise ImportError(
                "MLXBackend requires Apple Silicon and 'laya-mlx'. "
                "Install with: pip install 'laya-as-judge[mlx]' on macOS."
            ) from e

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        t0 = time.perf_counter()
        raw_result = self._agent.predict(state, normalized_q)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_result["latency_ms"] = round(latency_ms, 2)
        return raw_result
