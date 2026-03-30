"""
example_acc.py
==============
Demonstrates the Adaptive Compute Controller (ACC) in a simulated episode.

Run with:
    python example_acc.py

No API key is required – this example uses mock complexity scores to show
how the ACC maps scores to depths and enforces a budget.
"""

import json
from RLM.acc import AdaptiveComputeController, ComplexityScorer

# ---------------------------------------------------------------------------
# Simulated environment: a list of (query, context_length_chars) pairs
# ---------------------------------------------------------------------------

EPISODE_STEPS = [
    ("What is the capital of France?",                   500),
    ("Explain Newton's three laws of motion.",           2_000),
    ("Analyse the implications of quantum entanglement " 
     "on classical communication theory.",              50_000),
    ("List the planets in the solar system.",              300),
    ("Compare and contrast the ethical frameworks of "
     "utilitarianism and Kantian deontology.",          10_000),
    ("Why does water expand when it freezes?",           1_500),
    ("Evaluate the long-term strategic consequences of "
     "the industrial revolution on modern supply chains.",
                                                        80_000),
    ("Who wrote Hamlet?",                                  200),
    ("Design a distributed caching strategy for a "
     "high-traffic e-commerce platform.",               30_000),
    ("Deduce the missing value in the sequence: "
     "2, 6, 12, 20, 30, ?",                             1_000),
]


def main():
    # -----------------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------------
    scorer = ComplexityScorer()
    acc    = AdaptiveComputeController(max_api_calls=25)

    print("=" * 60)
    print("  Adaptive Compute Controller – Episode Demo")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 2. Run episode
    # -----------------------------------------------------------------------
    acc.new_episode()

    for i, (query, ctx_len) in enumerate(EPISODE_STEPS):
        # Simulate a context string of ctx_len characters
        mock_context = "x" * ctx_len

        # Score the step
        score = scorer.score(query, context=mock_context)

        # Ask the ACC for the appropriate depth
        depth = acc.select_depth(complexity_score=score)

        if depth == 0:
            print(f"\n  Step {i:02d} | BUDGET EXHAUSTED – episode terminated early.")
            break

        print(
            f"  Step {i:02d} | "
            f"score={score:.4f} | "
            f"depth={depth} | "
            f"budget_used={acc.api_calls_used:3d} / {acc.max_api_calls}"
        )

    # -----------------------------------------------------------------------
    # 3. End episode & report
    # -----------------------------------------------------------------------
    report = acc.end_episode()

    print("\n" + "=" * 60)
    print("  Episode Report")
    print("=" * 60)
    print(json.dumps(report.summary(), indent=2))

    print("\nDepth-by-step breakdown:")
    print(f"  {'Step':>4}  {'Complexity':>10}  {'Depth':>5}  {'Calls Used':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*5}  {'-'*10}")
    for r in report.records:
        print(
            f"  {r.step:>4}  {r.complexity_score:>10.4f}  "
            f"{r.depth_selected:>5}  {r.api_calls_used:>10}"
        )

    print()
    corr = report.depth_complexity_correlation
    if abs(corr) >= 0.5:
        print(f"  ✓ Strong correlation between complexity and depth (r={corr:.3f}).")
    else:
        print(f"  ~ Weak correlation (r={corr:.3f}) – may need more diverse steps.")


if __name__ == "__main__":
    main()
