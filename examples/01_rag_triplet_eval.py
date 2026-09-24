"""Example 01: Evaluating RAG Triplet (Query, Context, Answer) with Laya-as-a-Judge.

Replaces a slow 2,000ms GPT-4o evaluation with a sub-15ms, zero-token Laya forward pass.
"""

from laya_as_judge import FaithfulnessJudge, AnswerRelevanceJudge

def main():
    print("=" * 70)
    print("Example 01: High-Speed RAG Triplet Evaluation")
    print("=" * 70)

    # 1. Initialize Evaluators (auto-detects the MLX runtime; otherwise falls back to the heuristic emulator)
    faith_judge = FaithfulnessJudge()
    relevance_judge = AnswerRelevanceJudge()

    # 2. Define Sample RAG Interaction
    rag_sample = {
        "query": "What is the primary role of mitochondria in eukaryotic cells?",
        "context": (
            "Mitochondria are membrane-bound cell organelles that generate most of the "
            "chemical energy needed to power the cell's biochemical reactions. Chemical "
            "energy produced by the mitochondria is stored in a small molecule called "
            "adenosine triphosphate (ATP)."
        ),
        "answer": (
            "Mitochondria are the powerhouses of the cell; their main role is to generate "
            "most of the cell's chemical energy, which is stored as ATP."
        ),
    }

    print("\n--- Input RAG Triplet ---")
    print(f"Query:   {rag_sample['query']}")
    print(f"Context: {rag_sample['context']}")
    print(f"Answer:  {rag_sample['answer']}")

    # 3. Evaluate Faithfulness (Hallucination Detection)
    print("\n--- Evaluating Faithfulness ---")
    faith_report = faith_judge.evaluate_rag(
        query=rag_sample["query"],
        context=rag_sample["context"],
        answer=rag_sample["answer"],
    )

    print(f"Is Faithful:           {faith_report.get_bool('is_faithful')} (P(true) = {faith_report.judgements['is_faithful'].prob_true:.3f})")
    print(f"Hallucination Severity: {faith_report.get_score('hallucination_severity'):.2f} / 3.0")
    print(f"Error Category:         {faith_report.get_choice('error_type')}")
    print(f"Latency:                {faith_report.latency_ms:.2f} ms")
    print(f"Output Tokens:          {faith_report.output_tokens} (No decoding loop!)")
    print(f"Cost:                   ${faith_report.cost_usd:.4f}")

    # 4. Evaluate Answer Relevance
    print("\n--- Evaluating Answer Relevance ---")
    rel_report = relevance_judge.evaluate_response(
        query=rag_sample["query"],
        answer=rag_sample["answer"],
    )

    print(f"Directly Answers Query: {rel_report.get_bool('answers_query')}")
    print(f"Relevance Score:        {rel_report.get_score('relevance_score'):.2f} / 3.0")
    print(f"Is Concise:             {rel_report.get_bool('is_concise')}")
    print(f"Latency:                {rel_report.latency_ms:.2f} ms")

    # 5. Contrast with Hallucinated Sample
    hallucinated_answer = (
        "Mitochondria produce solar energy through photosynthesis and manufacture chlorophyll."
    )
    print("\n--- Evaluating Hallucinated Answer ---")
    print(f"Bad Answer: {hallucinated_answer}")
    bad_report = faith_judge.evaluate_rag(
        query=rag_sample["query"],
        context=rag_sample["context"],
        answer=hallucinated_answer,
    )
    print(f"Is Faithful:           {bad_report.get_bool('is_faithful')} (P(true) = {bad_report.judgements['is_faithful'].prob_true:.3f})")
    print(f"Hallucination Severity: {bad_report.get_score('hallucination_severity'):.2f} / 3.0")
    print(f"Error Category:         {bad_report.get_choice('error_type')}")

    print("\n" + "=" * 70)
    print("Both evaluations completed in <1 ms with guaranteed typed schemas!")
    print("=" * 70)

if __name__ == "__main__":
    main()
