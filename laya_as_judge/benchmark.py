"""Benchmark comparison engine between Laya-as-a-Judge and LLM-as-a-Judge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .judges.base import BaseJudge


@dataclass
class BenchmarkComparison:
    """Detailed side-by-side benchmark comparison."""
    sample_count: int
    laya_p50_latency_ms: float
    laya_p95_latency_ms: float
    laya_throughput_qps: float
    laya_output_tokens: int
    laya_cost_usd: float
    laya_schema_failures: int
    
    llm_p50_latency_ms: float
    llm_p95_latency_ms: float
    llm_throughput_qps: float
    llm_output_tokens: int
    llm_cost_usd: float
    llm_schema_failures: int

    speedup_factor: float
    cost_reduction_factor: str
    laya_backend: str = "unknown"
    llm_baseline_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "speedup_factor": f"{self.speedup_factor:.1f}x",
            "cost_reduction": self.cost_reduction_factor,
            "laya_backend": self.laya_backend,
            "llm_baseline_simulated": self.llm_baseline_simulated,
            "laya": {
                "p50_latency_ms": self.laya_p50_latency_ms,
                "p95_latency_ms": self.laya_p95_latency_ms,
                "throughput_qps": self.laya_throughput_qps,
                "output_tokens": self.laya_output_tokens,
                "cost_usd": self.laya_cost_usd,
                "schema_failure_rate": f"{(self.laya_schema_failures / self.sample_count) * 100:.1f}%",
            },
            "llm_judge": {
                "p50_latency_ms": self.llm_p50_latency_ms,
                "p95_latency_ms": self.llm_p95_latency_ms,
                "throughput_qps": self.llm_throughput_qps,
                "output_tokens": self.llm_output_tokens,
                "cost_usd": self.llm_cost_usd,
                "schema_failure_rate": f"{(self.llm_schema_failures / self.sample_count) * 100:.1f}%",
            },
        }


class BenchmarkRunner:
    """Runs head-to-head comparison between Laya and LLM-as-a-Judge."""

    def __init__(
        self,
        judge: BaseJudge,
        llm_judge_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ):
        self.judge = judge
        self.llm_judge_fn = llm_judge_fn

    def run(
        self,
        samples: List[Any],
        simulate_llm_if_missing: bool = True,
    ) -> BenchmarkComparison:
        """Run benchmark on test samples."""
        n = len(samples)
        if n == 0:
            raise ValueError("Benchmark samples list cannot be empty")

        # 1. Benchmark Laya
        laya_latencies = []
        laya_output_tokens = 0
        laya_schema_failures = 0

        t0_laya = time.perf_counter()
        for sample in samples:
            rep = self.judge.evaluate(sample)
            laya_latencies.append(rep.latency_ms)
            laya_output_tokens += rep.output_tokens
            if len(rep.judgements) != len(self.judge.questions):
                laya_schema_failures += 1
        total_laya_time = time.perf_counter() - t0_laya

        laya_p50 = float(np.percentile(laya_latencies, 50))
        laya_p95 = float(np.percentile(laya_latencies, 95))
        laya_qps = n / max(0.001, total_laya_time)
        laya_cost = 0.0

        # 2. Benchmark LLM Judge
        llm_latencies = []
        llm_output_tokens = 0
        llm_schema_failures = 0
        llm_cost = 0.0

        if self.llm_judge_fn:
            t0_llm = time.perf_counter()
            for sample in samples:
                t_item = time.perf_counter()
                try:
                    res = self.llm_judge_fn(sample)
                    llm_latencies.append((time.perf_counter() - t_item) * 1000.0)
                    llm_output_tokens += res.get("output_tokens", 250)
                    llm_cost += res.get("cost_usd", 0.03)
                except Exception:
                    llm_schema_failures += 1
                    llm_latencies.append((time.perf_counter() - t_item) * 1000.0)
            total_llm_time = time.perf_counter() - t0_llm
        else:
            # SIMULATED baseline (no LLM is called): synthetic numbers drawn from assumed
            # typical figures for a hosted LLM judge. These are assumptions, not measurements.
            # P50 ~1800ms, output ~220 tokens, cost ~$0.03 per judge call, ~6.5% JSON parse failures
            rng = np.random.default_rng(42)
            llm_latencies = (rng.normal(loc=1850.0, scale=350.0, size=n)).clip(min=900.0).tolist()
            llm_output_tokens = int(n * 240)
            llm_cost = round(n * 0.03, 4)
            llm_schema_failures = int(n * 0.065)
            total_llm_time = sum(llm_latencies) / 1000.0

        llm_p50 = float(np.percentile(llm_latencies, 50))
        llm_p95 = float(np.percentile(llm_latencies, 95))
        llm_qps = n / max(0.001, total_llm_time)

        speedup = llm_p50 / max(0.1, laya_p50)

        return BenchmarkComparison(
            sample_count=n,
            laya_p50_latency_ms=round(laya_p50, 2),
            laya_p95_latency_ms=round(laya_p95, 2),
            laya_throughput_qps=round(laya_qps, 1),
            laya_output_tokens=laya_output_tokens,
            laya_cost_usd=laya_cost,
            laya_schema_failures=laya_schema_failures,
            llm_p50_latency_ms=round(llm_p50, 2),
            llm_p95_latency_ms=round(llm_p95, 2),
            llm_throughput_qps=round(llm_qps, 1),
            llm_output_tokens=llm_output_tokens,
            llm_cost_usd=llm_cost,
            llm_schema_failures=llm_schema_failures,
            speedup_factor=round(speedup, 1),
            cost_reduction_factor="100% ($0.00 vs cloud tokens)",
            laya_backend=type(self.judge.engine).__name__,
            llm_baseline_simulated=self.llm_judge_fn is None,
        )
