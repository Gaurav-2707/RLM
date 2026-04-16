import sys
import os
import json

# Add current directory to path so we can import RLM
sys.path.append(os.getcwd())

from RLM.integrated_repl import IntegratedRLM

def run_test():
    # Target question from HotpotQA
    query = "Were Scott Derrickson and Ed Wood of the same nationality?"
    
    print(f"\n[PHASE 1] Running query with Semantic Search & Routing:")
    print(f"Query: {query}")
    print("-" * 60)
    
    # Initialize RLM with Phase 1 Semantic Features
    rlm = IntegratedRLM(
        enable_acc=True,
        enable_memory=True,
        enable_engine=True,
        use_semantic=True
    )
    
    # Run completion
    # Pass a dummy context as we are testing zero-shot logic + router
    answer = rlm.completion(context="HotpotQA Benchmark Case", query=query)
    
    print("-" * 60)
    print(f"FINAL ANSWER: {answer}")
    print(f"ACC DEPTH SELECTED: {rlm.last_depth}")
    
    # Store results for inspection
    results_dir = "benchmark/results/phase1_semantic/traces"
    os.makedirs(results_dir, exist_ok=True)
    
    trace_path = os.path.join(results_dir, "5a8b57f25542995d1e6f1371_semantic_run.json")
    with open(trace_path, "w") as f:
        json.dump(rlm.tracer.to_dict(), f, indent=2)
        
    print(f"Trace saved to: {trace_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_test()
