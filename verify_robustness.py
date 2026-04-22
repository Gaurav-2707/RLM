
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from RLM.repl import REPLEnv

def test_repl_robustness():
    print("Testing REPL Robustness (Hallucination Compatibility)...")
    env = REPLEnv(context_str="The capital of Japan is Tokyo. The population is large.")
    
    # Test 1: matches[0]['text'] hallucination
    # search_context returns a list of Snippets. Snippet['text'] should return Snippet.
    print("\n--- Test 1: matches[0]['text'] ---")
    code_1 = """
matches = search_context("Japan")
print(f"Match type: {type(matches[0])}")
print(f"Direct: {matches[0]}")
print(f"Dict access: {matches[0]['text']}")
"""
    res_1 = env.code_execution(code_1)
    print(f"stdout: {res_1.stdout.strip()}")
    print(f"stderr: {res_1.stderr.strip()}")

    # Test 2: IndentationError Hint
    print("\n--- Test 2: IndentationError Hint ---")
    code_2 = """
if True:
print("Missing indent")
"""
    res_2 = env.code_execution(code_2)
    print(f"stderr: {res_2.stderr.strip()}")

    # Test 3: List indexing on string (TypeError) Hint
    # This happens if a model thinks a string is a list
    print("\n--- Test 3: TypeError Hint ---")
    code_3 = "s = 'hello'; print(s['key'])"
    res_3 = env.code_execution(code_3)
    print(f"stderr: {res_3.stderr.strip()}")

if __name__ == "__main__":
    test_repl_robustness()
