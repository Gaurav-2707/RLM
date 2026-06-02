import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    fixed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fixed_budget_sweep"))
    acc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "adaptive_controller"))
    bon_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "best_of_n"))
    react_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "react_sweep"))
    
    fixed_files = glob.glob(os.path.join(fixed_dir, "*.json"))
    acc_files = glob.glob(os.path.join(acc_dir, "*.json"))
    bon_files = glob.glob(os.path.join(bon_dir, "*.json"))
    react_files = glob.glob(os.path.join(react_dir, "*.json"))
    
    budget_results = defaultdict(list)
    acc_results = []
    acc_iters = []
    bon_results = defaultdict(list)
    react_results = defaultdict(list)

    # Process Fixed Budget
    for file in fixed_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                budget = meta.get("budget")
                is_correct = meta.get("is_correct")
                if budget is not None and is_correct is not None:
                    budget_results[budget].append(is_correct)
        except Exception:
            pass

    # Process Adaptive Controller
    for file in acc_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                is_correct = meta.get("is_correct")
                repl_history = data.get("repl_history", [])
                
                if is_correct is not None and repl_history:
                    acc_results.append(is_correct)
                    # Number of iterations executed before stopping/rolling back
                    acc_iters.append(len(repl_history))
        except Exception:
            pass

    # Process Best-of-N
    for file in bon_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                N = meta.get("N")
                is_correct = meta.get("is_correct")
                if N is not None and is_correct is not None:
                    bon_results[N].append(is_correct)
        except Exception:
            pass

    # Process ReAct Sweep
    for file in react_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                budget = meta.get("budget")
                is_correct = meta.get("is_correct")
                if budget is not None and is_correct is not None:
                    react_results[budget].append(is_correct)
        except Exception:
            pass

    if not budget_results:
        print("No fixed budget data found.")
        return

    budgets = sorted(budget_results.keys())
    accuracies = [sum(budget_results[b])/len(budget_results[b]) for b in budgets]

    acc_accuracy = sum(acc_results) / len(acc_results) if acc_results else 0
    acc_avg_iters = sum(acc_iters) / len(acc_iters) if acc_iters else 0

    plt.figure(figsize=(10, 6))
    
    # Fixed Budget Curve
    plt.plot(budgets, accuracies, marker='o', linestyle='-', linewidth=2, color='blue', label='RLM Fixed Budget Baseline')
    
    # ReAct Sweep Curve
    if react_results:
        react_budgets = sorted(react_results.keys())
        react_accuracies = [sum(react_results[b])/len(react_results[b]) for b in react_budgets]
        plt.plot(react_budgets, react_accuracies, marker='s', linestyle='-.', linewidth=2, color='purple', label='ReAct Baseline')

    # Best-of-N Curve
    if bon_results:
        bon_n_vals = sorted(bon_results.keys())
        bon_accuracies = [sum(bon_results[n])/len(bon_results[n]) for n in bon_n_vals]
        plt.plot(bon_n_vals, bon_accuracies, marker='^', linestyle=':', linewidth=2, color='orange', label='Best-of-N Majority Vote')

    # Adaptive Controller Point
    if acc_results:
        plt.plot(acc_avg_iters, acc_accuracy, marker='*', markersize=15, color='red', label='Adaptive Early-Exit Controller (Ours)')
        
    plt.title("Reasoning Overshoot: Controller vs Baselines", fontsize=14)
    plt.xlabel("Average Iterations (Compute Cost)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(acc_dir, "controller_comparison.png")
    os.makedirs(acc_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    print(f"Fixed Budget Peak: {max(accuracies)*100:.1f}%")
    if acc_results:
        print(f"Adaptive Controller: {acc_accuracy*100:.1f}% at avg {acc_avg_iters:.1f} iterations")
    if react_results:
        react_accs = [sum(react_results[b])/len(react_results[b]) for b in react_budgets]
        print(f"ReAct Baseline Peak: {max(react_accs)*100:.1f}%")
    if bon_results:
        bon_accs = [sum(bon_results[n])/len(bon_results[n]) for n in bon_n_vals]
        print(f"Best-of-N Peak: {max(bon_accs)*100:.1f}%")

if __name__ == "__main__":
    main()
