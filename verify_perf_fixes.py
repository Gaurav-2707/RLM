import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from benchmark.hotpotqa_runner import llm_judge
from RLM.integrated_repl import IntegratedRLM

def test_judge_hardening():
    print("--- Testing Judge Hardening ---")
    q = "What is the capital of France?"
    gt = "Paris"
    
    # Test 1: Code leakage (Should be 0)
    pred_code = "```python\nprint('Paris')\n```"
    s1 = llm_judge(q, pred_code, gt)
    print(f"Code Leakage Test: {s1} (Expected 0)")
    
    # Test 2: Variable name leakage (Should be 0)
    pred_var = "final_answer"
    s2 = llm_judge(q, pred_var, gt)
    print(f"Var Name Leakage Test: {s2} (Expected 0)")
    
    # Test 3: Normal match (Should be 1)
    pred_ok = "The capital is Paris."
    s3 = llm_judge(q, pred_ok, gt)
    print(f"Normal Match Test: {s3} (Expected 1)")

def test_final_extractor():
    print("\n--- Testing Final Answer Extractor ---")
    # Using ollama provider explicitly
    rlm = IntegratedRLM(model="ollama/llama3.1:8b")
    rlm.messages = [
        {"role": "user", "content": "Analyze the context."},
        {"role": "assistant", "content": "```repl\nresult = 'Paris'\n```"},
        {"role": "user", "content": "REPL output: 'Paris'"},
        {"role": "assistant", "content": "The result of my calculation is Paris."}
    ]
    
    print("Running _force_final_extraction...")
    try:
        ans = rlm._force_final_extraction("What is the capital of France?")
        print(f"Extracted Answer: '{ans}'")
    except Exception as e:
        print(f"Extraction Test Failed: {e}")

if __name__ == "__main__":
    test_judge_hardening()
    test_final_extractor()
