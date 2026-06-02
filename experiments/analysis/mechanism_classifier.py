import os
import json
import glob

def evaluate_answer(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    gold_answer = gold.split("#### ")[-1].strip()
    return gold_answer in str(predicted)

def classify_mechanism(trace: dict) -> str:
    repl_history = trace.get("repl_history", [])
    if len(repl_history) < 2:
        return "Unknown"
        
    gold = trace.get("metadata", {}).get("gold_answer")
    if not gold:
        return "Unknown"
        
    # Find overshoot point
    first_correct_iter = -1
    lost_correct_iter = -1
    
    for step in repl_history:
        snapshot = step.get("snapshot_answer")
        if snapshot and evaluate_answer(snapshot, gold):
            first_correct_iter = step.get("iteration")
            break
            
    if first_correct_iter == -1:
        return "No Overshoot"
        
    for step in repl_history:
        snapshot = step.get("snapshot_answer")
        iteration = step.get("iteration")
        if iteration > first_correct_iter and snapshot:
            if not evaluate_answer(snapshot, gold):
                lost_correct_iter = iteration
                break
                
    if lost_correct_iter == -1:
        return "No Overshoot"
        
    # Classify the mechanism leading to the drop
    drop_step = next((s for s in repl_history if s.get("iteration") == lost_correct_iter), None)
    prev_step = next((s for s in repl_history if s.get("iteration") == lost_correct_iter - 1), None)
    
    if not drop_step or not prev_step:
        return "Unknown"
        
    # 1. Error Rate Spike (Tool Rejection)
    if drop_step.get("stderr") or "SyntaxError" in str(drop_step.get("response", "")):
        return "Tool Error Spike"
        
    # 2. Confidence Drop (Self-Doubt Overwrite)
    prev_conf = prev_step.get("confidence", 1.0)
    curr_conf = drop_step.get("confidence", 1.0)
    if curr_conf < prev_conf - 0.1:
        return "Confidence Drop / Self-Doubt"
        
    # 3. Context Noise Spike
    # If the context length grew massively between previous step and current drop step
    prev_ctx = prev_step.get("context_length", 0)
    curr_ctx = drop_step.get("context_length", 0)
    if curr_ctx > prev_ctx + 1500:
        return "Context Noise Accumulation"
        
    return "Other / Complex Interaction"

def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fixed_budget_sweep"))
    files = glob.glob(os.path.join(results_dir, "*.json"))
    
    mechanisms = {
        "Tool Error Spike": 0,
        "Confidence Drop / Self-Doubt": 0,
        "Context Noise Accumulation": 0,
        "Other / Complex Interaction": 0
    }
    
    total_overshoots = 0
    
    for file in files:
        try:
            with open(file, "r") as f:
                trace = json.load(f)
            mech = classify_mechanism(trace)
            if mech != "No Overshoot" and mech != "Unknown":
                mechanisms[mech] += 1
                total_overshoots += 1
        except Exception:
            pass

    print("=== Mechanistic Taxonomy of Overshoot Events ===")
    print(f"Total Overshoot Events Analyzed: {total_overshoots}")
    if total_overshoots > 0:
        for k, v in mechanisms.items():
            print(f"- {k}: {v} ({(v/total_overshoots)*100:.1f}%)")

if __name__ == "__main__":
    main()
