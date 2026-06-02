"""
Best-of-N Majority Voting Baseline.

The standard test-time compute scaling baseline (Snell et al., 2024).
Instead of iterating reasoning steps sequentially, it runs N independent
single-shot completions and takes the majority vote.

This baseline answers: "Is iterative reasoning better or worse than
parallel independent sampling at the same compute budget?"
"""

import os
import json
import argparse
from collections import Counter
from datasets import load_dataset
from RLM.utils.llm import LLMClient, DEFAULT_MODEL


def single_shot_answer(llm: LLMClient, question: str) -> str:
    """Get a single-shot answer from the model."""
    prompt = (
        f"Solve the following math problem step by step. "
        f"At the end, write your final answer as a single number after 'ANSWER:'.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:"
    )
    response = llm.completion(prompt)
    
    # Extract the answer after "ANSWER:"
    if "ANSWER:" in response.upper():
        parts = response.upper().split("ANSWER:")
        return parts[-1].strip().split()[0] if parts[-1].strip() else ""
    
    # Fallback: try to find the last number in the response
    import re
    numbers = re.findall(r'-?\d+\.?\d*', response)
    return numbers[-1] if numbers else ""


def evaluate_gsm8k_answer(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)


def main():
    parser = argparse.ArgumentParser(description="Run Best-of-N majority voting baseline.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b")
    args = parser.parse_args()

    dataset = load_dataset("gsm8k", "main", split="test")
    
    # N values matched to the iteration budgets for fair compute comparison
    n_values = [1, 3, 5, 10, 15, 20]
    
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results", "best_of_n"))
    os.makedirs(results_dir, exist_ok=True)
    
    llm = LLMClient(model=args.model)

    for N in n_values:
        print(f"\n=== Best-of-{N} Sweep ===")
        for idx in range(min(args.num_samples, len(dataset))):
            example = dataset[idx]
            question = example["question"]
            gold = example["answer"]
            
            # Generate N independent answers
            answers = []
            for _ in range(N):
                ans = single_shot_answer(llm, question)
                answers.append(ans)
            
            # Majority vote
            counter = Counter(answers)
            majority_answer = counter.most_common(1)[0][0] if counter else ""
            is_correct = evaluate_gsm8k_answer(majority_answer, gold)
            
            trace = {
                "metadata": {
                    "baseline": "best_of_n",
                    "question": question,
                    "gold_answer": gold,
                    "N": N,
                    "is_correct": is_correct,
                    "majority_answer": majority_answer,
                    "all_answers": answers,
                    "vote_distribution": dict(counter),
                },
            }
            
            trace_path = os.path.join(results_dir, f"bon_n{N}_q{idx}.json")
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2)
            
            print(f"  Q{idx+1}: N={N}, Majority={majority_answer}, Correct={is_correct}")


if __name__ == "__main__":
    main()
