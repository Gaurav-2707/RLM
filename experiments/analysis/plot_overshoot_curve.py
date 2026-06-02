import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fixed_budget_sweep"))
    files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not files:
        print(f"No result files found in {results_dir}.")
        return

    # budget -> [is_correct, is_correct, ...]
    budget_results = defaultdict(list)

    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                budget = meta.get("budget")
                is_correct = meta.get("is_correct")
                oracle_correct = meta.get("oracle_correct", is_correct) # fallback to is_correct if old trace
                
                if budget is not None and is_correct is not None:
                    budget_results[budget].append({
                        "correct": is_correct,
                        "oracle": oracle_correct
                    })
        except Exception as e:
            print(f"Error reading {file}: {e}")

    budgets = sorted(budget_results.keys())
    accuracies = []
    oracle_accuracies = []

    for b in budgets:
        results = budget_results[b]
        if results:
            acc = sum(r["correct"] for r in results) / len(results)
            oracle_acc = sum(r["oracle"] for r in results) / len(results)
        else:
            acc, oracle_acc = 0, 0
            
        accuracies.append(acc)
        oracle_accuracies.append(oracle_acc)
        print(f"Budget: {b} | Accuracy: {acc*100:.2f}% | Oracle: {oracle_acc*100:.2f}% | Samples: {len(results)}")

    plt.figure(figsize=(10, 6))
    plt.plot(budgets, accuracies, marker='o', linestyle='-', linewidth=2, color='blue', label='Standard Agent')
    
    # Plot Oracle Bound
    if any(oracle_accuracies):
        plt.plot(budgets, oracle_accuracies, marker='^', linestyle='--', linewidth=2, color='green', alpha=0.7, label='Oracle Upper Bound')
        
    plt.title("Reasoning Overshoot: Accuracy vs Iteration Budget", fontsize=14)
    plt.xlabel("Max Iterations (Budget)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = os.path.join(results_dir, "overshoot_curve.png")
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
