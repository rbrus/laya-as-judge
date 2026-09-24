"""Example 05: Benchmark Runner comparing Laya vs LLM-as-a-Judge.

Measures Laya-side latency distribution (P50, P95) and throughput on synthetic
samples. No LLM is called: the LLM-as-a-Judge column is a simulated baseline
built from assumed typical figures (see BenchmarkRunner). Pass `llm_judge_fn`
to BenchmarkRunner to measure a real LLM judge instead.
"""

from laya_as_judge import FaithfulnessJudge, BenchmarkRunner

def main():
    print("=" * 70)
    print("Example 05: Laya-as-a-Judge Benchmark (LLM baseline simulated)")
    print("=" * 70)

    judge = FaithfulnessJudge()
    runner = BenchmarkRunner(judge=judge)

    # 50 simulated RAG evaluation test cases
    samples = [
        {
            "query": f"Benchmark evaluation prompt #{i}",
            "context": f"Reference context document with verifiable evidence for test sample #{i}.",
            "answer": f"Candidate answer addressing query #{i} directly with cited facts.",
        }
        for i in range(50)
    ]

    print(f"Running benchmark on {len(samples)} samples...")
    comp = runner.run(samples)
    print(f"Laya backend: {comp.laya_backend}"
          + ("  (heuristic emulator, not the Laya model)" if comp.laya_backend == "EmulatorBackend" else ""))

    print("\n" + "-" * 70)
    print(f"{'Metric':<28} | {'Laya-as-a-Judge':<18} | {'LLM (simulated)':<18}")
    print("-" * 70)
    print(f"{'P50 Latency (ms)':<28} | {comp.laya_p50_latency_ms:<18.2f} | {comp.llm_p50_latency_ms:<18.2f}")
    print(f"{'P95 Latency (ms)':<28} | {comp.laya_p95_latency_ms:<18.2f} | {comp.llm_p95_latency_ms:<18.2f}")
    print(f"{'Throughput (evals/sec)':<28} | {comp.laya_throughput_qps:<18.1f} | {comp.llm_throughput_qps:<18.2f}")
    print(f"{'Output Tokens Generated':<28} | {comp.laya_output_tokens:<18} | ~{comp.llm_output_tokens:<17}")
    print(f"{'Total Cost for 50 evals':<28} | ${comp.laya_cost_usd:<17.4f} | ${comp.llm_cost_usd:<17.2f}")
    print(f"{'Schema Parse Failures':<28} | {comp.laya_schema_failures:<18} | {comp.llm_schema_failures:<18}")
    print("-" * 70)
    print(f"\nSpeedup Factor:  {comp.speedup_factor}x faster")
    print(f"Cost Reduction:  {comp.cost_reduction_factor}")
    print("=" * 70)

if __name__ == "__main__":
    main()
