"""
ReAct Baseline Sweep — Same fixed-budget sweep protocol on the ReAct agent.

Proves that Reasoning Overshoot is a universal phenomenon of iterative agents,
not specific to the RLM REPL architecture.
"""

import os
import json
import argparse
from datasets import load_dataset
from RLM.baselines.react_agent import ReActAgent


def evaluate_gsm8k_answer(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)


def main():
    parser = argparse.ArgumentParser(description="Run fixed budget sweep on ReAct baseline.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b")
    args = parser.parse_args()

    dataset = load_dataset("gsm8k", "main", split="test")
    budgets = [1, 3, 5, 10, 15, 20]
    results_dir = os.path.join(os.path.dirname(__file__), "results", "react_sweep")
    os.makedirs(results_dir, exist_ok=True)

    for budget in budgets:
        print(f"\n=== ReAct Sweep: Budget = {budget} ===")
        for idx in range(min(args.num_samples, len(dataset))):
            example = dataset[idx]
            question = example["question"]
            gold = example["answer"]

            agent = ReActAgent(
                model=args.model,
                max_iterations=budget,
                force_iterations=True,
            )

            predicted = agent.completion(question)
            is_correct = evaluate_gsm8k_answer(predicted, gold)
            peak_answer = agent.get_peak_confidence_answer()
            peak_correct = evaluate_gsm8k_answer(peak_answer, gold)

            trace = {
                "metadata": {
                    "agent": "react_baseline",
                    "question": question,
                    "gold_answer": gold,
                    "is_correct": is_correct,
                    "peak_answer_correct": peak_correct,
                    "budget": budget,
                },
                "repl_history": agent.trace_history,
                "snapshot_answers": agent._snapshot_answers,
                "snapshot_confidences": agent._snapshot_confidences,
            }

            trace_path = os.path.join(results_dir, f"react_b{budget}_q{idx}.json")
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2)

            print(f"  Q{idx+1}: Correct={is_correct}, Peak_Correct={peak_correct}")


if __name__ == "__main__":
    main()
