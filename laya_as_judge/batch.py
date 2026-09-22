"""Batch evaluation of datasets and CI/CD test suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .judges.base import BaseJudge
from .types import BatchEvaluationReport, EvaluationReport


class BatchEvaluator:
    """High-throughput batch evaluator for datasets, test suites, and CI/CD runs."""

    def __init__(
        self,
        judge: BaseJudge,
        estimated_llm_cost_per_eval: float = 0.03,
        estimated_llm_latency_seconds: float = 2.0,
    ):
        self.judge = judge
        self.estimated_llm_cost_per_eval = estimated_llm_cost_per_eval
        self.estimated_llm_latency_seconds = estimated_llm_latency_seconds

    def evaluate_items(
        self,
        items: List[Union[str, Dict[str, Any], List[Any]]],
        state_extractor: Optional[Callable[[Any], Union[str, Dict[str, Any]]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEvaluationReport:
        """Evaluate a list of items and generate an aggregated BatchEvaluationReport."""
        reports: List[EvaluationReport] = []
        latencies: List[float] = []
        total_input_tokens = 0

        total_items = len(items)
        for i, item in enumerate(items):
            state = state_extractor(item) if state_extractor else item
            report = self.judge.evaluate(state)
            reports.append(report)
            latencies.append(report.latency_ms)
            total_input_tokens += report.input_tokens

            if progress_callback:
                progress_callback(i + 1, total_items)

        total_lat_ms = sum(latencies)
        avg_lat_ms = total_lat_ms / max(1, total_items)
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        time_saved_s = (total_items * self.estimated_llm_latency_seconds) - (total_lat_ms / 1000.0)
        cost_saved_usd = total_items * self.estimated_llm_cost_per_eval

        return BatchEvaluationReport(
            reports=reports,
            total_latency_ms=round(total_lat_ms, 2),
            average_latency_ms=round(avg_lat_ms, 2),
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            total_input_tokens=total_input_tokens,
            total_output_tokens=0,
            total_cost_usd=0.0,
            estimated_llm_cost_saved_usd=round(cost_saved_usd, 4),
            estimated_time_saved_seconds=round(max(0.0, time_saved_s), 2),
        )

    def evaluate_jsonl(
        self,
        file_path: Union[str, Path],
        state_extractor: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> BatchEvaluationReport:
        """Evaluate a JSONL file containing evaluation samples."""
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        return self.evaluate_items(items, state_extractor=state_extractor)
