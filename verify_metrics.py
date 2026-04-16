
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from benchmark.hotpotqa_runner import llm_judge, run_benchmark, load_hotpotqa
from RLM.integrated_repl import IntegratedRLM

def test_judge():
    print("Testing 1-5 Judge logic...")
    q = "What is the capital of France?"
    gold = "Paris"
    
    # Test 5: Perfect
    score_5 = llm_judge(q, "Paris", gold)
    print(f"Perfect match score: {score_5}")
    
    # Test 4: Fluff
    score_4 = llm_judge(q, "The answer is Paris.", gold)
    print(f"Fluff match score: {score_4}")
    
    # Test 3: Reasoning leakage
    score_3 = llm_judge(q, "llm_query('capital of France') -> Paris", gold)
    print(f"Leakage score: {score_3}")
    
    # Test 1: Wrong
    score_1 = llm_judge(q, "London", gold)
    print(f"Wrong score: {score_1}")

def test_repl_hints():
    print("\nTesting REPL Hints...")
    from RLM.repl import REPLEnv
    env = REPLEnv(context_str="Paris is the capital of France.")
    
    # Hallucination 1: .answer attribute
    result_1 = env.code_execution("res = llm_query('capital'); print(res.answer)")
    print(f"Hallucination (AttributeError) stderr: {result_1.stderr}")
    
    # Hallucination 2: search_context(text=...) - Should NOT error now (Robustness)
    result_2 = env.code_execution("res = search_context('capital', text='something'); print('SUCCESS')")
    print(f"Robustness (TypeError avoidance) stdout: {result_2.stdout.strip()}")

if __name__ == "__main__":
    test_judge()
    test_repl_hints()
