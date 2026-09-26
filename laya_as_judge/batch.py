"""Batch evaluation of datasets and CI/CD test suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .judges.base import BaseJudge
from .types import BatchEvaluationReport, EvaluationReport


def _default_state_adapter(item: Any) -> Any:
    """Normalize common evaluation schema synonyms (e.g. Ragas, TruLens, LangSmith)."""
    if not isinstance(item, dict):
        return item
    d = dict(item)
    # Question / query synonym
    if "question" in d and "query" not in d:
        d["query"] = d.pop("question")
    # Contexts list synonym
    if "contexts" in d and "context" not in d:
        ctxs = d.pop("contexts")
        d["context"] = "\n".join(ctxs) if isinstance(ctxs, list) else str(ctxs)
    # Output / generation synonym for answer
    if "output" in d and "answer" not in d and "response" not in d:
        d["answer"] = d.pop("output")
    elif "generation" in d and "answer" not in d and "response" not in d:
        d["answer"] = d.pop("generation")
    return d


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

        extractor = state_extractor or _default_state_adapter

        total_items = len(items)
        for i, item in enumerate(items):
            state = extractor(item)
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
        output_path: Optional[Union[str, Path]] = None,
    ) -> BatchEvaluationReport:
        """Evaluate a JSONL file containing evaluation samples.
        
        Args:
            file_path: Path to input JSONL file.
            state_extractor: Optional callable to extract/adapt state from JSON record.
            output_path: Optional path to save evaluated results as JSONL.
        """
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}") from e

        report = self.evaluate_items(items, state_extractor=state_extractor)
        if output_path:
            report.save_jsonl(output_path)
        return report
