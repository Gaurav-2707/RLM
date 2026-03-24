import os
import logging
from RLM.repl import Sub_RLM
from RLM.engine.rlm_engine import RLMEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_engine():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY or GENAI_API_KEY environment variable.")
        return

    # Initialize RLM and Engine
    # Using a fast model for testing
    rlm = Sub_RLM(model="gemini-1.5-flash")
    engine = RLMEngine(rlm)

    # Sample Problem: A logical puzzle or a complex question
    problem = (
        "Three people (Alice, Bob, and Charlie) are in a room. "
        "Alice says Bob is lying. Bob says Charlie is lying. "
        "Charlie says Alice and Bob are both lying. "
        "Who is telling the truth?"
    )

    print(f"\n--- ORIGINAL PROBLEM ---\n{problem}\n")

    # Run the engine
    result = engine.run(problem)

    print("\n--- REASONING TRACE ---")
    for step in result["steps"]:
        print(f"\n>> STEP {step['step']} OUTPUT:\n{step['full_output']}")
        if "summary" in step:
            print(f"\n>> STEP {step['step']} SUMMARY:\n{step['summary']}")
        print("-" * 40)

    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])

if __name__ == "__main__":
    test_engine()
