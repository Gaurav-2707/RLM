"""
Scaling Law Sweep — Run the fixed-budget protocol across multiple model sizes.

Generates the data for "Figure 4: Inference-Time Chinchilla" in the paper.
Each model produces its own non-monotonic curve with a different N_opt.

Usage:
    # Local 8B
    uv run python -m RLM.experiments.run_scaling_law_sweep --model ollama/llama3.1:8b --num_samples 300

    # API 70B (requires TOGETHER_API_KEY or similar)
    uv run python -m RLM.experiments.run_scaling_law_sweep --model together/meta-llama/Llama-3.1-70B-Instruct --num_samples 300

    # GPT-4o-mini
    uv run python -m RLM.experiments.run_scaling_law_sweep --model gpt-4o-mini --num_samples 300
"""

import os
import json
import argparse
from datasets import load_dataset
from RLM.integrated_repl import IntegratedRLM

# Loaders
from RLM.experiments.datasets.hotpotqa_loader import load_hotpotqa_sample, format_hotpotqa_prompt, evaluate_hotpotqa_answer
from RLM.experiments.datasets.math_loader import load_math_sample, format_math_prompt, evaluate_math_answer

def evaluate_gsm8k_answer(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)


def main():
    parser = argparse.ArgumentParser(description="Run scaling law sweep across model sizes.")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--model", type=str, required=True,
                        help="Model string (e.g., ollama/llama3.1:8b, gpt-4o-mini)")
    parser.add_argument("--model_label", type=str, default=None,
                        help="Short label for file naming (e.g., 'llama8b', 'llama70b', 'gpt4omini')")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "hotpotqa", "math"],
                        help="Dataset to run the sweep on")
    args = parser.parse_args()

    model_label = args.model_label or args.model.replace("/", "_").replace(":", "_")
    
    print(f"Loading {args.dataset.upper()} dataset...")
    if args.dataset == "gsm8k":
        dataset_raw = load_dataset("gsm8k", "main", split="test")
        dataset = dataset_raw.select(range(min(args.num_samples, len(dataset_raw))))
    elif args.dataset == "hotpotqa":
        dataset = load_hotpotqa_sample(args.num_samples)
    elif args.dataset == "math":
        dataset = load_math_sample(args.num_samples)

    budgets = [1, 3, 5, 10, 15, 20]
    results_dir = os.path.join(os.path.dirname(__file__), "results", f"scaling_law_{model_label}_{args.dataset}")
    os.makedirs(results_dir, exist_ok=True)

    summary = {}

    for budget in budgets:
        correct_count = 0
        total = min(args.num_samples, len(dataset))

        print(f"\n=== {model_label} | Budget = {budget} ===")

        for idx in range(total):
            example = dataset[idx]
            
            # Format according to dataset
            if args.dataset == "gsm8k":
                question = example["question"]
                prompt = question
                gold = example["answer"]
            elif args.dataset == "hotpotqa":
                prompt = format_hotpotqa_prompt(example)
                question = example["question"]
                gold = example["answer"]
            elif args.dataset == "math":
                prompt = format_math_prompt(example)
                question = example["problem"]
                gold = example["solution"]

            rlm = IntegratedRLM(
                model=args.model,
                recursive_model=args.model,
                max_iterations=budget,
                force_iterations=True,
                enable_logging=False,
                enable_acc=False,
                enable_engine=False,
                enable_memory=False,
            )

            predicted = rlm.completion(context="", query=prompt)
            
            if args.dataset == "gsm8k":
                is_correct = evaluate_gsm8k_answer(predicted, gold)
            elif args.dataset == "hotpotqa":
                is_correct = evaluate_hotpotqa_answer(predicted, gold)
            elif args.dataset == "math":
                is_correct = evaluate_math_answer(predicted, gold)
                
            if is_correct:
                correct_count += 1
                
            # Calculate Oracle Bound (did it get it right at any point?)
            oracle_correct = False
            snapshot_answers = rlm._snapshot_answers
            for iter_num, ans in snapshot_answers.items():
                if args.dataset == "gsm8k" and evaluate_gsm8k_answer(ans, gold):
                    oracle_correct = True
                    break
                elif args.dataset == "hotpotqa" and evaluate_hotpotqa_answer(ans, gold):
                    oracle_correct = True
                    break
                elif args.dataset == "math" and evaluate_math_answer(ans, gold):
                    oracle_correct = True
                    break

            # Save individual trace
            trace_path = os.path.join(results_dir, f"trace_b{budget}_q{idx}.json")
            rlm.tracer.set_metadata({
                "model": args.model,
                "model_label": model_label,
                "dataset": args.dataset,
                "question": question,
                "gold_answer": gold,
                "is_correct": is_correct,
                "oracle_correct": oracle_correct,
                "budget": budget,
            })
            rlm.tracer.save(trace_path)

            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx+1}/{total}, Running Accuracy: {correct_count/(idx+1)*100:.1f}%")

        accuracy = correct_count / total
        summary[budget] = {"accuracy": accuracy, "correct": correct_count, "total": total}
        print(f"  Budget {budget} Final Accuracy: {accuracy*100:.1f}%")

    # Save summary
    summary_path = os.path.join(results_dir, "scaling_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "model_label": model_label, "results": summary}, f, indent=2)

    print(f"\nScaling law summary saved to {summary_path}")

    # Find N_opt
    best_budget = max(summary, key=lambda b: summary[b]["accuracy"])
    print(f"N_opt for {model_label}: {best_budget} (accuracy: {summary[best_budget]['accuracy']*100:.1f}%)")


if __name__ == "__main__":
    main()
