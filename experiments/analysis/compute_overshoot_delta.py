import os
import json
import glob
from collections import defaultdict
import numpy as np

def evaluate_answer(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)

def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fixed_budget_sweep"))
    files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not files:
        print(f"No result files found in {results_dir}.")
        return

    deltas = []
    overshoot_count = 0
    total_analyzed = 0

    print("Analyzing Overshoot Deltas...")
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
            
            meta = data.get("metadata", {})
            gold = meta.get("gold_answer")
            if not gold:
                continue
                
            repl_history = data.get("repl_history", [])
            if not repl_history:
                continue

            total_analyzed += 1
            
            first_correct_iter = -1
            lost_correct_iter = -1
            
            # Find when correct answer first appeared in snapshots
            for step in repl_history:
                snapshot = step.get("snapshot_answer")
                iteration = step.get("iteration")
                if snapshot and evaluate_answer(snapshot, gold):
                    first_correct_iter = iteration
                    break
                    
            if first_correct_iter != -1:
                # Check if it was lost later
                for step in repl_history:
                    snapshot = step.get("snapshot_answer")
                    iteration = step.get("iteration")
                    if iteration > first_correct_iter and snapshot:
                        if not evaluate_answer(snapshot, gold):
                            lost_correct_iter = iteration
                            break
                            
                if lost_correct_iter != -1:
                    delta = lost_correct_iter - first_correct_iter
                    deltas.append(delta)
                    overshoot_count += 1
                    
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print("\n=== Overshoot Delta Analysis ===")
    print(f"Total traces analyzed: {total_analyzed}")
    print(f"Traces exhibiting overshoot: {overshoot_count} ({(overshoot_count/total_analyzed)*100 if total_analyzed > 0 else 0:.1f}%)")
    
    if deltas:
        print(f"Average overshoot delta: {np.mean(deltas):.2f} iterations")
        print(f"Median overshoot delta: {np.median(deltas):.2f} iterations")
        print(f"Max overshoot delta: {max(deltas)} iterations")
    else:
        print("No overshoot deltas could be computed. This may happen if accuracy is perfectly monotonic or traces don't contain enough snapshots.")

if __name__ == "__main__":
    main()
