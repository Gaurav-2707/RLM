import os
import argparse
from RLM.integrated_repl import IntegratedRLM
from RLM.experiments.datasets.hotpotqa_loader import load_hotpotqa_sample, format_hotpotqa_prompt, evaluate_hotpotqa_answer

def main():
    parser = argparse.ArgumentParser(description="Run fixed budget sweep on HotpotQA.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of questions to run")
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b", help="Model to use")
    args = parser.parse_args()

    print(f"Loading HotpotQA dataset... (Running {args.num_samples} samples)")
    dataset = load_hotpotqa_sample(args.num_samples)
    
    budgets = [1, 3, 5, 10, 15, 20]
    results_dir = os.path.join(os.path.dirname(__file__), "results", "hotpotqa_sweep")
    os.makedirs(results_dir, exist_ok=True)

    for budget in budgets:
        print(f"\n=============================================")
        print(f"Running Sweep with Budget = {budget}")
        print(f"=============================================\n")
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            prompt = format_hotpotqa_prompt(example)
            gold = example["answer"]
            
            print(f"--- Q{idx+1}/{args.num_samples} [Budget: {budget}] ---")
            
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
            
            predicted = rlm.completion(context="", query=prompt)
            
            is_correct = evaluate_hotpotqa_answer(predicted, gold)
            oracle_correct = any(evaluate_hotpotqa_answer(ans, gold) for ans in rlm._snapshot_answers.values())
            
            # Save the trace for downstream overshoot delta calculation
            trace_path = os.path.join(results_dir, f"trace_b{budget}_q{idx}.json")
            rlm.tracer.set_metadata({
                "dataset": "hotpot_qa",
                "question": example["question"],
                "gold_answer": gold,
                "is_correct": is_correct,
                "oracle_correct": oracle_correct,
                "budget": budget
            })
            rlm.tracer.save(trace_path)
            
            print(f"Done. Correct: {is_correct}. Trace saved to {trace_path}\n")

if __name__ == "__main__":
    main()
