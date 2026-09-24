"""Unit tests for the benchmark comparison runner."""

from laya_as_judge import BenchmarkRunner, FaithfulnessJudge


def test_benchmark_runner():
    judge = FaithfulnessJudge()
    runner = BenchmarkRunner(judge=judge)

    samples = [
        {"query": f"Q{i}", "context": f"C{i}", "answer": f"A{i}"}
        for i in range(10)
    ]

    comp = runner.run(samples)
    assert comp.sample_count == 10
    assert comp.laya_output_tokens == 0
    assert comp.llm_output_tokens > 0
    assert comp.laya_cost_usd == 0.0
    assert comp.speedup_factor > 0
    d = comp.to_dict()
    assert "laya" in d
    assert "llm_judge" in d
    # Without llm_judge_fn the LLM side is a simulated baseline and must be flagged as such.
    assert comp.llm_baseline_simulated is True
    assert d["llm_baseline_simulated"] is True
    assert comp.laya_backend == type(judge.engine).__name__
