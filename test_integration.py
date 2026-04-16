"""
Full Integration Benchmark — 100 HotpotQA examples with all RLM modules enabled.

Runs:
  - ACC (Adaptive Compute Controller)
  - Semantic Episodic Memory (DenseRetriever)
  - Step-over-Step Engine

Results and per-question traces are saved to benchmark/results/full_integration/.
"""

import os
import json
from datetime import datetime

from benchmark.hotpotqa_runner import load_hotpotqa, run_benchmark, save_results
from RLM.integrated_repl import IntegratedRLM
from RLM.utils.llm import DEFAULT_MODEL

NUM_EXAMPLES = 100

# ── Output paths ──────────────────────────────────────────────────────
RUN_DIR   = os.path.join("benchmark", "results", "full_integration")
TRACE_DIR = os.path.join(RUN_DIR, "traces")
SUMMARY   = os.path.join(RUN_DIR, "summary.json")


def rlm_factory():
    """Return a fully-integrated RLM instance (all modules ON)."""
    return IntegratedRLM(
        enable_acc=True,
        enable_memory=True,
        enable_engine=True,
        enable_logging=True,
    )


def on_result(result: dict):
    """Callback invoked after each question — prints a one-liner progress update."""
    status = "✓" if result["em"] == 1 else "✗"
    print(
        f"  [{status}] Q{result['index']+1:>3d}/{NUM_EXAMPLES}  "
        f"EM={result['em']}  F1={result['f1']:.2f}  "
        f"Time={result['time_s']:.1f}s  "
        f"| {result['question'][:55]}..."
    )


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)

    print("=" * 70)
    print("  RLM Full Integration Benchmark")
    print("=" * 70)
    print(f"  Model      : {DEFAULT_MODEL}")
    print(f"  Examples   : {NUM_EXAMPLES}")
    print(f"  Modules    : ACC + Memory + Engine")
    print(f"  Output dir : {RUN_DIR}")
    print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load dataset
    examples = load_hotpotqa(num_examples=NUM_EXAMPLES)

    # Run benchmark
    results = run_benchmark(
        examples=examples,
        rlm_factory=rlm_factory,
        mode="full_integration",
        on_result=on_result,
        trace_dir=TRACE_DIR,
    )

    # Save results
    save_results(results, SUMMARY)

    # Print final report
    agg = results["aggregate"]
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Exact Match (EM) : {agg['em']:.4f}  ({agg['correct_em']}/{agg['total']})")
    print(f"  F1 Score         : {agg['f1']:.4f}")
    print(f"  Avg Time/Q       : {agg['avg_time']:.1f}s")
    print(f"  Total Questions  : {agg['total']}")
    print("=" * 70)
    print(f"  Results saved to : {SUMMARY}")
    print(f"  Traces saved to  : {TRACE_DIR}")
    print(f"  Finished at      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
