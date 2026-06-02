import os
import json
import argparse
from datasets import load_dataset
from RLM.integrated_repl import IntegratedRLM

def evaluate_gsm8k_answer(predicted: str, gold: str) -> bool:
    """Very naive GSM8K evaluation. Just checks if the gold answer string is in the predicted output."""
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)

def main():
    parser = argparse.ArgumentParser(description="Run fixed budget sweep for Reasoning Overshoot paper.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of questions to run (10 for pilot/sanity check)")
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b", help="Model to use")
    args = parser.parse_args()

    print(f"Loading GSM8K dataset... (Running {args.num_samples} samples)")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    budgets = [1, 3, 5, 10, 15, 20]
    results_dir = os.path.join(os.path.dirname(__file__), "results", "fixed_budget_sweep")
    os.makedirs(results_dir, exist_ok=True)

    for budget in budgets:
        print(f"\n=============================================")
        print(f"Running Sweep with Budget (Max Iterations) = {budget}")
        print(f"=============================================\n")
        
        for idx in range(min(args.num_samples, len(dataset))):
            example = dataset[idx]
            question = example["question"]
            gold = example["answer"]
            
            print(f"--- Q{idx+1}/{args.num_samples} [Budget: {budget}] ---")
            
            # Initialize RLM with force_iterations=True to force exactly N loops
            rlm = IntegratedRLM(
                model=args.model,
                recursive_model=args.model,
                max_iterations=budget,
                force_iterations=True,
                enable_logging=False,
                enable_acc=False,
                enable_engine=False,
                enable_memory=False
            )
            
            # The IntegratedRLM will loop `budget` times, capturing snapshots, 
            # and return the final salvaged snapshot.
            predicted = rlm.completion(context="", query=question)
            
            is_correct = evaluate_gsm8k_answer(predicted, gold)
            oracle_correct = any(evaluate_gsm8k_answer(ans, gold) for ans in rlm._snapshot_answers.values())
            
            # Save the trace for downstream overshoot delta calculation
            trace_path = os.path.join(results_dir, f"trace_b{budget}_q{idx}.json")
            rlm.tracer.set_metadata({
                "question": question,
                "gold_answer": gold,
                "is_correct": is_correct,
                "oracle_correct": oracle_correct,
                "budget": budget
            })
            rlm.tracer.save(trace_path)
            
            print(f"Done. Correct: {is_correct}. Trace saved to {trace_path}\n")

if __name__ == "__main__":
    main()
