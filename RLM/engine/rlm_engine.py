import re
import logging
from typing import Optional, List, Dict, Any
from RLM.rlm import RLM
try:
    from RLM.engine.templates import STEP_0_DECOMPOSITION, STEP_1_REFINEMENT, STEP_2_SYNTHESIS
except ImportError:
    # Fallback if the module structure is different during testing
    from templates import STEP_0_DECOMPOSITION, STEP_1_REFINEMENT, STEP_2_SYNTHESIS

logger = logging.getLogger(__name__)

class RLMEngine:
    """
    Reasoning Engine that implements a fixed-depth (D=3) recursive reasoning loop.
    Phases: Decomposition, Refinement, Synthesis.
    Features: Anchoring, Summary Extraction, and Self-Correction.
    """

    def __init__(self, rlm: RLM):
        self.rlm = rlm
        self.history = []

    def _extract_summary(self, text: str) -> str:
        """
        Extract the content within <summary> tags.
        If tags are missing, returns a truncated version of the full text.
        """
        match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: take the last 150 tokens/words if tags are missing
        words = text.split()
        if len(words) > 150:
            return " ".join(words[-150:])
        return text

    def run(self, problem: str) -> Dict[str, Any]:
        """
        Execute the 3-step reasoning loop.
        Returns a dictionary containing the full logs and the final result.
        """
        self.history = []
        
        # --- STEP 0: Decomposition ---
        logger.info("Executing Step 0: Decomposition...")
        prompt_0 = STEP_0_DECOMPOSITION.format(problem=problem)
        resp_0 = self.rlm.completion(prompt_0)
        summary_0 = self._extract_summary(resp_0)
        self.history.append({"step": 0, "full_output": resp_0, "summary": summary_0})

        # --- STEP 1: Refinement ---
        logger.info("Executing Step 1: Refinement...")
        prompt_1 = STEP_1_REFINEMENT.format(problem=problem, prev_summary=summary_0)
        resp_1 = self.rlm.completion(prompt_1)
        summary_1 = self._extract_summary(resp_1)
        self.history.append({"step": 1, "full_output": resp_1, "summary": summary_1})

        # --- STEP 2: Synthesis ---
        logger.info("Executing Step 2: Synthesis...")
        prompt_2 = STEP_2_SYNTHESIS.format(problem=problem, prev_summary=summary_1)
        resp_2 = self.rlm.completion(prompt_2)
        # Note: Step 2 doesn't need a summary as it is the final step
        self.history.append({"step": 2, "full_output": resp_2})

        return {
            "problem": problem,
            "final_output": resp_2,
            "steps": self.history
        }

    def reset(self):
        """Reset the engine state."""
        self.history = []
        self.rlm.reset()
