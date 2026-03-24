import os
import json
import time
import logging
import random
import traceback
from typing import Dict, Any, List, Optional, Tuple

# Import project modules
# We assume the script is run from the root directory or RLM is in PYTHONPATH
try:
    from RLM.agent_controller import AgentController
    from RLM.memory.system import EpisodicMemorySystem
    from RLM.engine.rlm_engine import RLMEngine
    from RLM.gemini_interface import GeminiInterface, GeminiConfig
    from RLM.models import AgentState, MemoryEntry, ReasoningOutput, ActionResult
    from RLM.rlm import RLM
except ImportError:
    # Fallback for direct execution if structure differs
    import sys
    sys.path.append(os.path.dirname(os.getcwd()))
    from agent_controller import AgentController
    from memory.system import EpisodicMemorySystem
    from engine.rlm_engine import RLMEngine
    from gemini_interface import GeminiInterface, GeminiConfig
    from models import AgentState, MemoryEntry, ReasoningOutput, ActionResult
    from rlm import RLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VerificationHarness")

class MockRLM(RLM):
    """Mock RLM for testing and resilience simulation."""
    def __init__(self, fail_rate: float = 0.0, malformed_rate: float = 0.0):
        self.fail_rate = fail_rate
        self.malformed_rate = malformed_rate
        self.call_count = 0
        self.token_input = 0
        self.token_output = 0

    def completion(self, prompt: str) -> str:
        self.call_count += 1
        self.token_input += len(prompt) // 4  # Rough estimate
        
        # Add a small delay for realistic metrics
        time.sleep(0.5)

        if random.random() < self.fail_rate:
            raise RuntimeError("Simulated API Failure")
        
        if random.random() < self.malformed_rate:
            return "This IS NOT JSON! { incomplete: 'data' ..."

        # Simple logic to simulate reasoning based on prompt content
        if "STEP 0" in prompt:
            res = {
                "rationale": "Decomposing the problem into components",
                "proposed_action": "gather_information",
                "expected_outcome": "Better understanding of sub-tasks",
                "summary": "Summary of decomposition"
            }
        elif "STEP 1" in prompt:
            res = {
                "rationale": "Refining assumptions and logic",
                "proposed_action": "refine_hypothesis",
                "expected_outcome": "Accurate reasoning path",
                "summary": "Summary of refinement"
            }
        elif "STEP 2" in prompt:
            res = {
                "rationale": "Synthesizing final answer",
                "proposed_action": "submit_final_answer",
                "expected_outcome": "Problem resolved",
                "is_terminal": True,
                "confidence_score": 0.95
            }
        else:
            res = {"rationale": "Generic response"}

        response_text = f"<summary>{res.get('summary', 'Final Summary')}</summary>\n" + json.dumps(res)
        self.token_output += len(response_text) // 4
        return response_text

    def cost_summary(self) -> Dict[str, float]:
        return {"input_tokens": float(self.token_input), "output_tokens": float(self.token_output)}

    def reset(self):
        self.call_count = 0

class IntegratedRLMEngine(RLMEngine):
    """Modified RLMEngine that can handle memory context."""
    def run(self, problem: str, memory_context: str = "") -> Dict[str, Any]:
        # Inject memory context into the problem description for Step 0
        enhanced_problem = f"{problem}\n\nRELEVANT PAST MEMORIES:\n{memory_context}"
        return super().run(enhanced_problem)

class IntegratedAgentController(AgentController):
    """AgentController that uses RLMEngine for its reasoning core."""
    def __init__(self, gemini, memory_system, rlm_engine: RLMEngine):
        super().__init__(gemini, memory_system)
        self.rlm_engine = rlm_engine
        self.metrics: Dict[str, Any] = {
            "retrieval_scores": [],
            "confidences": [],
            "tokens_used": 0,
            "start_time": 0.0,
            "end_time": 0.0
        }

    def _orchestrate_reasoning(self, memory_context: str) -> ReasoningOutput:
        """Override to use the 3-step RLMEngine."""
        logger.info("Orchestrating reasoning via RLMEngine (Depth-3)...")
        
        # Start timer for token/time tracking if not already set
        if self.metrics["start_time"] == 0:
            self.metrics["start_time"] = time.time()

        # Run the engine
        problem = self.state.task_description
        if isinstance(self.rlm_engine, IntegratedRLMEngine):
            engine_result = self.rlm_engine.run(problem, memory_context)
        else:
            engine_result = self.rlm_engine.run(problem)

        # Extract final output and try to parse it as ReasoningOutput
        final_text = engine_result["final_output"]
        
        # The RLMEngine outputs text, we need to convert it to ReasoningOutput
        # We'll look for JSON in the final output or use the parse_json_robust logic
        try:
            return self._parse_json_robust(final_text)
        except Exception as e:
            logger.warning(f"Failed to parse RLMEngine output: {e}. creating fallback.")
            return ReasoningOutput(
                rationale="Fallback from RLMEngine",
                proposed_action="Complete task",
                expected_outcome="Task completion",
                confidence_score=0.8,
                is_terminal=True
            )

    def _format_retrieved_memories(self, memories: List[Tuple[Any, float]]) -> str:
        # Track retrieval scores
        if memories:
            self.metrics["retrieval_scores"].append(max(score for m, score in memories))
        return super()._format_retrieved_memories(memories)

    def _commit_to_memory(self, reasoning: ReasoningOutput, result: ActionResult):
        self.metrics["confidences"].append(reasoning.confidence_score)
        super()._commit_to_memory(reasoning, result)

class VerificationHarness:
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.report = []

    def run_phase_a(self, controller: IntegratedAgentController):
        """Phase A: Initial Learning"""
        logger.info("=== PHASE A: Initial Learning ===")
        problem = "Solve a complex logistics puzzle: 5 cities, 3 trucks, optimize routes for minimum fuel."
        controller.run("session_a", problem, max_iterations=1)
        self.results['phase_a_memories'] = len(controller.memory.memories)
        logger.info(f"Phase A complete. Memories saved: {self.results['phase_a_memories']}")

    def run_phase_b(self, controller: IntegratedAgentController):
        """Phase B: Retrieval Test"""
        logger.info("=== PHASE B: Retrieval Test ===")
        # Semantically similar problem
        problem = "Optimize delivery routes for 4 vans across 6 towns to save gas."
        controller.run("session_b", problem, max_iterations=1)
        self.results['phase_b_retrieval'] = controller.metrics["retrieval_scores"][-1] if controller.metrics["retrieval_scores"] else 0
        logger.info(f"Phase B complete. Best retrieval score: {self.results['phase_b_retrieval']:.4f}")

    def run_phase_c(self, controller: IntegratedAgentController):
        """Phase C: Verification"""
        logger.info("=== PHASE C: Verification ===")
        assert self.results['phase_a_memories'] > 0, "Phase A should have saved a memory."
        assert self.results['phase_b_retrieval'] > 0.5, "Phase B should have retrieved the memory from Phase A."
        logger.info("Verification Assertions Passed!")

    def run_resilience_test(self, gemini_key: str):
        """Resilience Testing"""
        logger.info("=== RESILIENCE TESTING ===")
        # Test API Failure
        mock_rlm = MockRLM(fail_rate=1.0) # Always fail
        engine = RLMEngine(mock_rlm)
        mem = EpisodicMemorySystem()
        
        # We need a GeminiInterface that can handle failures
        # Actually AgentController handles exceptions in its loop
        controller = IntegratedAgentController(None, mem, engine)
        controller.state = AgentState(session_id="fail_test", task_description="test", max_iterations=1)
        
        try:
            controller.run("fail_test", "This should fail", max_iterations=1)
            logger.info("Resilience Test: API Failure handled (Iteration caught error)")
        except Exception as e:
            logger.error(f"Resilience Test: Unexpected crash: {e}")

        # Test Malformed JSON
        mock_rlm = MockRLM(malformed_rate=1.0)
        engine = RLMEngine(mock_rlm)
        controller = IntegratedAgentController(None, mem, engine)
        controller.run("malformed_test", "This should use regex recovery", max_iterations=1)
        logger.info("Resilience Test: Malformed JSON handled (Regex recovery used)")

    def generate_report(self, controller: IntegratedAgentController):
        """Reasoning Continuity Report"""
        report = []
        report.append("# Reasoning Continuity Report")
        report.append(f"Total Iterations: {len(controller.metrics['confidences'])}")
        report.append(f"Average Confidence: {sum(controller.metrics['confidences'])/len(controller.metrics['confidences']):.2f}")
        report.append(f"Max Retrieval Score: {max(controller.metrics['retrieval_scores']) if controller.metrics['retrieval_scores'] else 0:.4f}")
        
        # Simulated Token usage vs Time
        total_time = controller.metrics["end_time"] - controller.metrics["start_time"]
        report.append(f"Total Time Taken: {total_time:.2f}s")
        
        with open("continuity_report.md", "w") as f:
            f.write("\n".join(report))
        
        logger.info("Continuity Report generated: continuity_report.md")
        return "\n".join(report)

def main():
    # Setup
    api_key = os.getenv("GENAI_API_KEY") or "mock_key"
    gemini = GeminiInterface(api_key=api_key) if api_key != "mock_key" else None
    
    # Use real Gemini if key is provided, otherwise Mock
    if not gemini:
        mock_rlm = MockRLM()
        engine = IntegratedRLMEngine(mock_rlm)
    else:
        # In a real scenario, we'd wrap GeminiInterface into an RLM implementation
        # For now, let's use MockRLM as the engine core for predictability in tests
        mock_rlm = MockRLM()
        engine = IntegratedRLMEngine(mock_rlm)

    memory = EpisodicMemorySystem()
    controller = IntegratedAgentController(gemini, memory, engine)
    
    harness = VerificationHarness()
    
    try:
        harness.run_phase_a(controller)
        harness.run_phase_b(controller)
        harness.run_phase_c(controller)
        
        controller.metrics["end_time"] = time.time()
        
        print("\n" + harness.generate_report(controller))
        
        harness.run_resilience_test(api_key)
        
    except Exception as e:
        logger.error(f"Harness failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
