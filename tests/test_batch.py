"""Unit tests for batch dataset evaluation."""

from pathlib import Path
from laya_as_judge import BatchEvaluator, FaithfulnessJudge


def test_batch_evaluator_items():
    judge = FaithfulnessJudge()
    evaluator = BatchEvaluator(judge=judge)

    items = [
        {"query": f"Q{i}", "context": f"C{i}", "answer": f"A{i}"}
        for i in range(5)
    ]

    report = evaluator.evaluate_items(items)
    assert len(report.reports) == 5
    assert report.p50_latency_ms >= 0.0
    assert report.total_output_tokens == 0
    assert report.estimated_llm_cost_saved_usd > 0


def test_batch_evaluator_jsonl(tmp_path: Path):
    file_path = tmp_path / "test.jsonl"
    file_path.write_text(
        '{"query": "q1", "context": "c1", "answer": "a1"}\n'
        '{"query": "q2", "context": "c2", "answer": "a2"}\n'
    )

    judge = FaithfulnessJudge()
    evaluator = BatchEvaluator(judge=judge)
    report = evaluator.evaluate_jsonl(file_path)

    assert len(report.reports) == 2
    assert report.total_output_tokens == 0
