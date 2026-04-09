import os
import argparse
import sys
from datetime import datetime

# Add root to path so we can import RLM and benchmark modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.hotpotqa_runner import load_hotpotqa, run_benchmark, save_results
from RLM.integrated_repl import IntegratedRLM

def main():
    parser = argparse.ArgumentParser(description="Phase 0 Baseline Runner for RLM NLP Integration Research")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to run per configuration")
    parser.add_argument("--model", type=str, default="ollama/llama3", help="Root model to use")
    parser.add_argument("--recursive_model", type=str, default="ollama/llama3", help="Recursive model to use")
    args = parser.parse_args()

    # Base path for results
    base_dir = os.path.join("benchmark", "results", "phase0_baseline")
    os.makedirs(base_dir, exist_ok=True)

    print(f"[{datetime.now().isoformat()}] Starting Phase 0 Baseline Runs")
    print(f"Target: {args.num_examples} examples per configuration")
    print(f"Models: Root={args.model}, Recursive={args.recursive_model}")

    # 1. Load Data
    examples = load_hotpotqa(num_examples=args.num_examples)

    # Configurations to run
    # We define factory functions that return a fresh IntegratedRLM instance
    configs = [
        {
            "name": "standard",
            "factory": lambda: IntegratedRLM(
                model=args.model, 
                recursive_model=args.recursive_model,
                enable_acc=False, 
                enable_memory=False, 
                enable_engine=False
            )
        },
        {
            "name": "memory",
            "factory": lambda: IntegratedRLM(
                model=args.model, 
                recursive_model=args.recursive_model,
                enable_acc=False, 
                enable_memory=True, 
                enable_engine=False
            )
        },
        {
            "name": "full",
            "factory": lambda: IntegratedRLM(
                model=args.model, 
                recursive_model=args.recursive_model,
                enable_acc=True, 
                enable_memory=True, 
                enable_engine=True
            )
        }
    ]

    for config in configs:
        print(f"\n{'='*80}")
        print(f" CONFIGURATION: {config['name'].upper()}")
        print(f"{'='*80}")
        
        run_name = config["name"]
        run_dir = os.path.join(base_dir, run_name)
        trace_dir = os.path.join(run_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)

        try:
            results = run_benchmark(
                examples=examples,
                rlm_factory=config["factory"],
                mode=run_name,
                trace_dir=trace_dir
            )
            
            summary_path = os.path.join(run_dir, "summary.json")
            save_results(results, summary_path)
            
            agg = results.get("aggregate", {})
            print(f"\n[DONE] {run_name}: EM={agg.get('em')}, F1={agg.get('f1')}, Time={agg.get('avg_time')}s")
        
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed configuration {run_name}: {str(e)}")

    print(f"\n[{datetime.now().isoformat()}] Phase 0 Baseline Successfully Completed.")
    print(f"Results available in: {base_dir}")

if __name__ == "__main__":
    main()
