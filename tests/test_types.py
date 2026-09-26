"""Unit tests for laya_as_judge.types."""

import pytest
from laya_as_judge.types import (
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    NoulResult,
    ScoreResult,
)


def test_score_result():
    res = ScoreResult(
        name="relevance",
        confidence=0.85,
        score=2.65,
        legend={"0": "bad", "1": "okay", "2": "good", "3": "excellent"},
        level_probabilities={"0": 0.05, "1": 0.10, "2": 0.35, "3": 0.50},
    )
    assert res.decision_type == DecisionType.SCORE
    assert res.score == 2.65
    assert res.is_high_confidence is True
    d = res.to_dict()
    assert d["score"] == 2.65
    assert d["decision_type"] == "score"


def test_choice_result():
    res = ChoiceResult(
        name="routing",
        confidence=0.92,
        choice="billing",
        probabilities={"billing": 0.92, "technical": 0.05, "sales": 0.03},
    )
    assert res.decision_type == DecisionType.CHOICE
    assert res.choice == "billing"
    d = res.to_dict()
    assert d["choice"] == "billing"
    assert d["probabilities"]["billing"] == 0.92


def test_noul_result():
    res = NoulResult(
        name="is_safe",
        confidence=0.95,
        holds=True,
        prob_true=0.95,
    )
    assert res.decision_type == DecisionType.NOUL
    assert res.holds is True
    assert res.prob_true == 0.95


def test_evaluation_report_helpers():
    score_res = ScoreResult(name="coherence", confidence=0.8, score=2.8)
    noul_res = NoulResult(name="is_factual", confidence=0.9, holds=True, prob_true=0.9)
    choice_res = ChoiceResult(name="winner", confidence=0.85, choice="model_a")

    report = EvaluationReport(
        state={"text": "test"},
        judgements={
            "coherence": score_res,
            "is_factual": noul_res,
            "winner": choice_res,
        },
        latency_ms=12.5,
        input_tokens=150,
        output_tokens=0,
    )

    assert report.get_score("coherence") == 2.8
    assert report.get_bool("is_factual") is True
    assert report.get_choice("winner") == "model_a"
    assert report.get_confidence("winner") == 0.85
    assert report.output_tokens == 0
    assert report.cost_usd == 0.0
    assert report.average_confidence == pytest.approx((0.8 + 0.9 + 0.85) / 3.0)

    # Key errors when types mismatch
    with pytest.raises(KeyError):
        report.get_score("is_factual")
    with pytest.raises(KeyError):
        report.get_bool("winner")
    with pytest.raises(KeyError):
        report.get_choice("coherence")


def test_evaluation_report_to_json_and_passed(tmp_path):
    from laya_as_judge.types import BatchEvaluationReport

    good_report = EvaluationReport(
        state={"text": "clean"},
        judgements={
            "is_safe": NoulResult(name="is_safe", confidence=0.9, holds=True, prob_true=0.95),
            "quality": ScoreResult(name="quality", confidence=0.8, score=2.5),
        },
        latency_ms=5.0,
        input_tokens=100,
    )
    assert good_report.passed is True
    json_str = good_report.to_json()
    assert '"is_safe"' in json_str

    bad_report = EvaluationReport(
        state={"text": "attack"},
        judgements={
            "is_safe": NoulResult(name="is_safe", confidence=0.9, holds=False, prob_true=0.05),
        },
        latency_ms=5.0,
        input_tokens=100,
    )
    assert bad_report.passed is False

    batch = BatchEvaluationReport(
        reports=[good_report, bad_report],
        total_latency_ms=10.0,
        average_latency_ms=5.0,
        p50_latency_ms=5.0,
        p95_latency_ms=5.0,
        total_input_tokens=200,
    )
    assert '"sample_count": 2' in batch.to_json()

    out_file = tmp_path / "out.jsonl"
    batch.save_jsonl(out_file)
    assert out_file.exists()
    lines = out_file.read_text().strip().split("\n")
    assert len(lines) == 2
