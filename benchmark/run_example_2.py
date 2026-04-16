import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

from RLM.integrated_repl import IntegratedRLM

def run_test():
    # Multi-hop question requiring director lookup and date comparison
    query = "Which 2014 film was directed by the person who also directed 'Inception'?"
    
    print(f"\n[PHASE 1 - EXAMPLE 2] Running multi-hop query:")
    print(f"Query: {query}")
    print("-" * 60)
    
    rlm = IntegratedRLM(
        enable_acc=True,
        enable_memory=True,
        enable_engine=True,
        use_semantic=True
    )
    
    # Run completion
    answer = rlm.completion(context="HotpotQA Multi-hop Case", query=query)
    
    print("-" * 60)
    print(f"FINAL ANSWER: {answer}")
    print(f"ACC DEPTH SELECTED: {rlm.last_depth}")
    
    # Store results
    results_dir = "benchmark/results/phase1_semantic/traces"
    os.makedirs(results_dir, exist_ok=True)
    
    trace_path = os.path.join(results_dir, "example_2_interstellar.json")
    with open(trace_path, "w") as f:
        json.dump(rlm.tracer.to_dict(), f, indent=2)
        
    print(f"Individual trace saved to: {trace_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_test()
