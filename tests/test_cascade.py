"""Unit tests for speculative cascaded judging."""

from laya_as_judge import CascadedJudge, FaithfulnessJudge


def test_cascaded_judge_accepted_fast_path():
    judge = FaithfulnessJudge()

    llm_called = False
    def mock_llm_fn(state, questions):
        nonlocal llm_called
        llm_called = True
        return {"verdict": "ok"}

    # Clear-cut case resolves at Tier 1 with realistic threshold (0.70)
    cascade = CascadedJudge(
        tier1_judge=judge,
        tier2_llm_judge_fn=mock_llm_fn,
        confidence_threshold=0.70,
    )

    report = cascade.evaluate({
        "query": "Where is the Eiffel Tower located?",
        "context": "The Eiffel Tower is located in Paris, France.",
        "answer": "The Eiffel Tower is in Paris, France.",
    })
    assert llm_called is False
    assert report.metadata["escalated_to_llm"] is False
    assert cascade.stats["tier1_resolved"] == 1
    assert cascade.stats["tier2_escalated"] == 0


def test_cascaded_judge_escalation_path():
    judge = FaithfulnessJudge()

    llm_called = False
    def mock_llm_fn(state, questions):
        nonlocal llm_called
        llm_called = True
        return {"verdict": "escalated_ok", "cost_usd": 0.03}

    # Set threshold high (0.99) so it always escalates
    cascade = CascadedJudge(
        tier1_judge=judge,
        tier2_llm_judge_fn=mock_llm_fn,
        confidence_threshold=0.99,
        estimated_llm_cost_per_eval=0.03,
    )

    report = cascade.evaluate({"query": "q", "context": "c", "answer": "a"})
    assert llm_called is True
    assert report.metadata["escalated_to_llm"] is True
    assert report.metadata["speculative_tier"] == "tier2_llm"
    assert cascade.stats["tier2_escalated"] == 1
    assert report.cost_usd == 0.03

    summary = cascade.summary()
    assert summary["total_evaluations"] == 1
    assert summary["tier2_escalated_count"] == 1
