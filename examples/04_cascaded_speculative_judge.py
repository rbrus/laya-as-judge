"""Example 04: Cascaded / Speculative Judging Architecture.

Uses fast, local Laya as Tier 1 for 95% of evaluations.
Only escalates ambiguous, low-confidence edge cases to a heavy Tier 2 cloud LLM.
Saves >90% on latency and API costs while maintaining frontier evaluation accuracy.
"""

import time
from laya_as_judge import CascadedJudge, FaithfulnessJudge

def dummy_cloud_llm_judge(state, questions):
    """Simulates a heavy cloud LLM judge (e.g. GPT-4o, Claude 3.5 Sonnet).

    In real production, you would call OpenAI, Anthropic, or Google Gemini API here.
    """
    time.sleep(0.05)  # Simulate network hop and token generation
    return {
        "model": "cloud-frontier-llm-judge",
        "reasoning": "Detailed multi-token chain-of-thought analysis by frontier model...",
        "verdict": "faithful",
        "confidence": 0.99,
        "cost_usd": 0.03,
        "output_tokens": 280,
    }

def main():
    print("=" * 70)
    print("Example 04: Speculative Cascaded Judge Pipeline")
    print("=" * 70)

    # 1. Initialize Fast Laya Judge as Tier 1
    tier1 = FaithfulnessJudge()

    # 2. Build Cascaded Judge with 0.70 confidence threshold
    cascade = CascadedJudge(
        tier1_judge=tier1,
        tier2_llm_judge_fn=dummy_cloud_llm_judge,
        confidence_threshold=0.70,
        estimated_llm_cost_per_eval=0.03,
    )

    test_queries = [
        # Clear-cut cases (Laya resolves locally with high confidence)
        {
            "query": "Where is the Eiffel Tower located?",
            "context": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
            "answer": "The Eiffel Tower is located in Paris, France.",
        },
        {
            "query": "What is water's formula?",
            "context": "Water is a chemical compound consisting of two hydrogen atoms bonded to one oxygen atom (H2O).",
            "answer": "The chemical formula for water is H2O.",
        },
        # Nuanced ambiguous edge case
        {
            "query": "Is artificial general intelligence safe?",
            "context": "Safety research around AGI presents varying expert perspectives without scientific consensus.",
            "answer": "AGI is partially dangerous but some argue safety protocols will prevail.",
        },
    ]

    print("\nRunning evaluations through Speculative Cascaded Judge...")
    for idx, item in enumerate(test_queries, 1):
        report = cascade.evaluate(item)
        escalated = report.metadata.get("escalated_to_llm", False)
        print(f"\nItem #{idx}: Query = '{item['query']}'")
        print(f"  → Handled by:  {'Tier 2 (Cloud LLM Judge)' if escalated else 'Tier 1 (Laya Local Judge)'}")
        print(f"  → Confidence:  {report.metadata.get('average_confidence', 0.0) * 100:.1f}%")
        print(f"  → Latency:     {report.latency_ms:.2f} ms")
        print(f"  → Cost:        ${report.cost_usd:.4f}")

    print("\n" + "=" * 70)
    print("Cascade Summary & Economic Savings:")
    print("=" * 70)
    summary = cascade.summary()
    for k, v in summary.items():
        print(f"{k:26}: {v}")

if __name__ == "__main__":
    main()
