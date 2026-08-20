import os
import logging
from RLM.utils.llm import LLMClient

logger = logging.getLogger(__name__)

class TestGenerator:
    """
    RLM-TestGen (Auto-Verifier): Generates failing test cases to establish
    the mathematical boundary condition for Test-Driven Reinforcement Learning (TDRL).
    """
    def __init__(self):
        # We initialize the LLM Client. This will auto-detect OpenAI/Gemini/Ollama based on env keys.
        self.llm = LLMClient()
        
    def _read_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Source file not found: {filepath}")
        with open(filepath, "r") as f:
            return f.read()
            
    def generate_test(self, source_filepath: str, developer_intent: str) -> str:
        """
        Reads the target file and developer intent, and uses the LLM to write a failing pytest case.
        Saves the test to a local file and returns the raw python code.
        """
        logger.info(f"Generating test case for intent: '{developer_intent}' in {source_filepath}")
        source_code = self._read_file(source_filepath)
        
        system_prompt = (
            "You are an expert Python Auto-Verifier Agent for Recursive Labs.\n"
            "The developer wants to fix a bug in their code, but there is no test case yet.\n"
            "Your job is to write a single, isolated, fully-functional `pytest` test case that will currently FAIL "
            "because the bug exists, but will PASS once the developer fixes the bug.\n"
            "Rules:\n"
            "1. Output ONLY valid, runnable Python code.\n"
            "2. Do NOT wrap the code in markdown backticks (no ```python). Just the raw python code.\n"
            "3. Make sure to import the target file correctly assuming the test is run from the project root.\n"
        )
        
        user_prompt = (
            f"Developer Intent: {developer_intent}\n\n"
            f"Target File: {source_filepath}\n"
            f"Source Code:\n{source_code}\n\n"
            "Write the failing pytest file now:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Generate the test code
            test_code = self.llm.completion(messages)
            
            # Clean up markdown if the LLM hallucinated it despite instructions
            test_code = test_code.replace("```python\n", "").replace("```python", "").replace("```\n", "").replace("```", "").strip()
            
            # Write the test to a file
            output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "test_auto_generated.py")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w") as f:
                f.write(test_code)
                
            logger.info(f"Test case successfully generated at {output_path}")
            return test_code
            
        except Exception as e:
            logger.error(f"Failed to generate test case: {e}")
            raise

# Singleton instance for easy API usage
test_generator = TestGenerator()
