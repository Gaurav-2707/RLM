"""
Generates the calibration dataset for Conformal Prediction.

Runs N questions from GSM8K TRAIN split through the full forced-iteration
pipeline. The output is a single JSON file containing all traces with
ground-truth labels, which is consumed by ConformalCalibrator.calibrate().

Usage:
    uv run python -m RLM.experiments.run_calibration_set --num_samples 200
"""

import os
import json
import argparse
from datasets import load_dataset
from RLM.integrated_repl import IntegratedRLM


def main():
    parser = argparse.ArgumentParser(description="Generate conformal calibration traces.")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of calibration questions (minimum 200 for tight bounds)")
    parser.add_argument("--max_iterations", type=int, default=20,
                        help="Max forced iterations per question")
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for calibration traces JSON")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "results", "conformal_calibration_traces.json")

    print(f"[Calibration] Loading GSM8K TRAIN split ({args.num_samples} samples)...")
    dataset = load_dataset("gsm8k", "main", split="train")

    all_traces = []

    for idx in range(min(args.num_samples, len(dataset))):
        example = dataset[idx]
        question = example["question"]
        gold = example["answer"]
        gold_answer = gold.split("#### ")[-1].strip()

        print(f"[Calibration] Q{idx+1}/{args.num_samples}: {question[:80]}...")

        rlm = IntegratedRLM(
            model=args.model,
            recursive_model=args.model,
            max_iterations=args.max_iterations,
            force_iterations=True,
            enable_logging=False,
            enable_acc=False,
            enable_engine=False,
            enable_memory=False,
        )

        predicted = rlm.completion(context="", query=question)

        # Build the trace with ground truth labels
        trace = rlm.tracer.to_dict()
        trace["metadata"]["gold_answer"] = gold
        trace["metadata"]["gold_answer_extracted"] = gold_answer

        # Label each snapshot with correctness
        for step in trace.get("repl_history", []):
            snapshot = step.get("snapshot_answer")
            if snapshot is not None:
                step["is_correct_at_step"] = gold_answer in str(snapshot)
            else:
                step["is_correct_at_step"] = False

        all_traces.append(trace)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_traces, f, indent=2)

    print(f"\n[Calibration] Saved {len(all_traces)} calibration traces to {args.output}")
    print(f"[Calibration] Total (iteration, confidence, correctness) tuples: "
          f"{sum(len(t.get('repl_history', [])) for t in all_traces)}")


if __name__ == "__main__":
    main()
