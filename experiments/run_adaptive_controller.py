import os
import argparse
from datasets import load_dataset
from RLM.integrated_repl import IntegratedRLM

def evaluate_gsm8k_answer(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)

def main():
    parser = argparse.ArgumentParser(description="Run adaptive controller sweep for Reasoning Overshoot paper.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of questions to run (10 for pilot/sanity check)")
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b", help="Model to use")
    args = parser.parse_args()

    print(f"Loading GSM8K dataset... (Running {args.num_samples} samples)")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results", "adaptive_controller")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n=============================================")
    print(f"Running Adaptive Controller Sweep")
    print(f"=============================================\n")
    
    for idx in range(min(args.num_samples, len(dataset))):
        example = dataset[idx]
        question = example["question"]
        gold = example["answer"]
        
        print(f"--- Q{idx+1}/{args.num_samples} ---")
        
        # Initialize RLM with force_iterations=False and enable_acc=True
        rlm = IntegratedRLM(
            model=args.model,
            recursive_model=args.model,
            max_iterations=20, # Set a high theoretical max, let ACC exit early
            force_iterations=False,
            enable_logging=False,
            enable_acc=True,
            enable_engine=False,
            enable_memory=False
        )
        
        predicted = rlm.completion(context="", query=question)
        
        is_correct = evaluate_gsm8k_answer(predicted, gold)
        
        trace_path = os.path.join(results_dir, f"trace_acc_q{idx}.json")
        rlm.tracer.set_metadata({
            "question": question,
            "gold_answer": gold,
            "is_correct": is_correct,
            "controller": "adaptive_early_exit"
        })
        rlm.tracer.save(trace_path)
        
        actual_iters = rlm._current_iteration
        print(f"Done. Correct: {is_correct}. Exited at Iteration: {actual_iters}. Trace saved to {trace_path}\n")

if __name__ == "__main__":
    main()
