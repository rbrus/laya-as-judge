"""Example 02: Pairwise A/B Model Comparison with Laya-as-a-Judge.

Compares two candidate responses to determine the preferred answer,
preference margin, and factual coherence without paying for frontier LLM judge tokens.
"""

from laya_as_judge import PairwiseComparisonJudge

def main():
    print("=" * 70)
    print("Example 02: Pairwise Model Evaluation (A/B Testing)")
    print("=" * 70)

    judge = PairwiseComparisonJudge()

    prompt = (
        "Write a Python function to safely fetch JSON from a URL with timeout and retry."
    )

    response_a = '''
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_json_with_retry(url, timeout=5, retries=3):
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()
'''

    response_b = '''
import requests

def fetch_json(url):
    # Just requests.get without retries or error handling
    return requests.get(url).json()
'''

    print(f"\nUser Prompt:\n{prompt}\n")
    print(f"Candidate Model A:\n{response_a.strip()}\n")
    print(f"Candidate Model B:\n{response_b.strip()}\n")

    report = judge.compare(prompt, response_a, response_b)

    winner_result = report.judgements["winner"]
    margin_result = report.judgements["preference_margin"]

    print("--- Evaluation Verdict ---")
    print(f"Winner:            {report.get_choice('winner').upper()}")
    print(f"Winner Confidence: {winner_result.confidence * 100:.1f}%")
    print(f"All Probabilities: {winner_result.probabilities}")
    print(f"Margin of Win:     {margin_result.score:.2f} / 3.0 (0=negligible, 3=decisive)")
    print(f"Latency:           {report.latency_ms:.2f} ms")
    print(f"Output Tokens:     {report.output_tokens} tokens")
    print(f"Judge API Cost:    ${report.cost_usd:.4f}")

    print("\n" + "=" * 70)
    print("Zero token generation; completed in sub-millisecond execution!")
    print("=" * 70)

if __name__ == "__main__":
    main()
