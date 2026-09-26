"""Unit tests for batch dataset evaluation."""

import json
from pathlib import Path
import pytest
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

    out_path = tmp_path / "evaluated.jsonl"
    judge = FaithfulnessJudge()
    evaluator = BatchEvaluator(judge=judge)
    report = evaluator.evaluate_jsonl(file_path, output_path=out_path)

    assert len(report.reports) == 2
    assert report.total_output_tokens == 0
    assert out_path.exists()
    saved_lines = out_path.read_text().strip().split("\n")
    assert len(saved_lines) == 2
    assert "judgements" in json.loads(saved_lines[0])


def test_batch_evaluator_schema_adaptation(tmp_path: Path):
    # Test Ragas format: 'question' instead of 'query', 'contexts' list instead of 'context'
    file_path = tmp_path / "ragas.jsonl"
    file_path.write_text(
        '{"question": "What is water?", "contexts": ["Water is H2O", "It is liquid."], "output": "Water is H2O."}\n'
    )

    judge = FaithfulnessJudge()
    evaluator = BatchEvaluator(judge=judge)
    report = evaluator.evaluate_jsonl(file_path)

    assert len(report.reports) == 1
    assert report.reports[0].get_bool("is_faithful") is True


def test_batch_evaluator_invalid_json(tmp_path: Path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("INVALID_JSON\n")

    judge = FaithfulnessJudge()
    evaluator = BatchEvaluator(judge=judge)
    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        evaluator.evaluate_jsonl(bad_file)
