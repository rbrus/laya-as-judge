"""Unit tests for all judge presets and custom judge builder."""

from laya_as_judge.judges import (
    AgentTrajectoryJudge,
    AnswerRelevanceJudge,
    CustomJudge,
    FaithfulnessJudge,
    InstructionAdherenceJudge,
    PairwiseComparisonJudge,
    SafetyGuardJudge,
)
from laya_as_judge.types import DecisionType


def test_faithfulness_judge():
    judge = FaithfulnessJudge()
    report = judge.evaluate_rag(
        query="What is H2O?",
        context="H2O is water.",
        answer="H2O is water.",
    )
    assert "is_faithful" in report.judgements
    assert "hallucination_severity" in report.judgements
    assert "error_type" in report.judgements
    assert report.judgements["is_faithful"].decision_type == DecisionType.NOUL
    assert report.output_tokens == 0


def test_relevance_judge():
    judge = AnswerRelevanceJudge()
    report = judge.evaluate_response(
        query="What time is it?",
        answer="It is 3 PM.",
    )
    assert "answers_query" in report.judgements
    assert "relevance_score" in report.judgements


def test_instruction_judge():
    judge = InstructionAdherenceJudge()
    report = judge.evaluate_adherence(
        instructions="Respond in 1 sentence without using commas.",
        response="This is a sentence.",
    )
    assert "follows_all" in report.judgements
    assert "adherence_score" in report.judgements


def test_safety_guard_judge():
    judge = SafetyGuardJudge()
    report = judge.inspect("Hello, can you help me write an email?", role="prompt")
    assert "is_safe" in report.judgements
    assert "jailbreak_attempt" in report.judgements
    assert "harm_severity" in report.judgements
    assert report.get_bool("is_safe") is True


def test_pairwise_judge():
    judge = PairwiseComparisonJudge()
    report = judge.compare(
        prompt="Write a hello world program in python.",
        response_a="print('Hello, world!')",
        response_b="print('Hi')",
    )
    assert "winner" in report.judgements
    assert "preference_margin" in report.judgements
    assert report.get_choice("winner") in ["response_a", "response_b", "tie"]


def test_agent_eval_judge():
    judge = AgentTrajectoryJudge()
    report = judge.evaluate_step(
        goal="Find the price of BTC",
        history=[{"role": "user", "content": "Find BTC price"}],
        action={"tool": "crypto_price_api", "args": {"symbol": "BTC"}},
    )
    assert "tool_call_valid" in report.judgements
    assert "progress_score" in report.judgements
    assert "recommended_action" in report.judgements


def test_custom_judge_builder():
    judge = (
        CustomJudge.builder("CodeReviewJudge")
        .add_noul("is_clean", "Is the code readable?")
        .add_score("maintainability", "Rate maintainability", ["poor", "fair", "good", "great"])
        .add_choice("language", "Detect language", ["python", "rust", "go", "other"])
        .build()
    )

    report = judge.evaluate("def add(a, b): return a + b")
    assert "is_clean" in report.judgements
    assert "maintainability" in report.judgements
    assert "language" in report.judgements
    assert report.get_choice("language") in ["python", "rust", "go", "other"]
