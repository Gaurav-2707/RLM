import json
import re
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from .models import AgentState, MemoryEntry, ReasoningOutput, ActionResult
from .gemini_interface import GeminiInterface
from .memory.system import EpisodicMemorySystem

# Setup logger
logger = logging.getLogger("AgentController")

class AgentController:
    """
    The central coordinator for the Recursive Language Model (RLM).
    Implements the loop: State -> Memory Retrieval -> Recursive Reasoning -> Action -> Outcome -> Memory Update.
    """
    def __init__(
        self,
        gemini: GeminiInterface,
        memory_system: EpisodicMemorySystem,
        adaptive_compute: Optional[Any] = None  # Future-proof ACC hook
    ):
        self.gemini = gemini
        self.memory = memory_system
        self.acc = adaptive_compute
        self.state: Optional[AgentState] = None

    def run(self, session_id: str, task: str, max_iterations: int = 5) -> str:
        """
        Executes the recursive reasoning loop for a given task.
        """
        self.state = AgentState(
            session_id=session_id,
            task_description=task,
            max_iterations=max_iterations,
            status="initializing"
        )

        logger.info(f"Starting session {session_id} for task: {task}")

        while self.state.is_active and self.state.current_iteration < self.state.max_iterations:
            self.state.current_iteration += 1
            it_num = self.state.current_iteration
            logger.info(f"--- Iteration {it_num} ---")

            try:
                # 1. Memory Retrieval
                self.state.status = "searching_memory"
                query_state = self.state.current_context or self.state.task_description
                retrieved = self.memory.retrieve(query_state, top_k=3)
                memory_str = self._format_retrieved_memories(retrieved)

                # 2. Recursive Reasoning
                self.state.status = "reasoning"
                reasoning = self._orchestrate_reasoning(memory_str)
                logger.debug(f"Reasoning rationale: {reasoning.rationale}")

                # 3. Action execution
                self.state.status = "acting"
                # Note: In a real system, the Action module would handle proposed_action.
                # Here we simulate or dispatch based on proposed_action.
                result = self._dispatch_action(reasoning)
                
                # 4. Memory Update
                self.state.status = "updating_memory"
                self._commit_to_memory(reasoning, result)

                # Update reasoning history
                self.state.accumulated_reasoning.append(reasoning.rationale)

                # Check for completion
                if reasoning.is_terminal:
                    self.state.is_active = False
                    self.state.status = "completed"
                    logger.info(f"Task completed successfully in {it_num} iterations.")
                    return result.observation

                # Update context for next iteration
                self.state.current_context = result.observation

                # Optional ACC adjustment
                if self.acc:
                    # Future implementation for dynamical depth adjustment
                    # self.acc.adjust_recursion(self.state, reasoning)
                    pass

            except Exception as e:
                logger.error(f"Error in iteration {it_num}: {e}")
                self.state.status = "failed"
                self.state.is_active = False
                return f"Agent failed: {str(e)}"

        if self.state.current_iteration >= self.state.max_iterations:
            logger.warning("Max iterations reached without terminal state.")
            self.state.status = "max_depth_reached"
        
        return self.state.current_context or "Task finished without explicit output."

    def _orchestrate_reasoning(self, memory_context: str) -> ReasoningOutput:
        """
        Calls Gemini to perform reasoning based on current state and retrieved memory.
        """
        sys_prompt = (
            "You are a recursive reasoning engine. Analyze the current state and provide the next step. "
            "Return your response in strict JSON format with the following keys: "
            "rationale (str), proposed_action (str), action_parameters (dict), "
            "expected_outcome (str), confidence_score (float), is_terminal (bool)."
        )
        
        user_prompt = (
            f"Original Task: {self.state.task_description}\n"
            f"Current Context/Observation: {self.state.current_context or 'None'}\n"
            f"Historical Context: {' | '.join(self.state.accumulated_reasoning[-3:])}\n"
            f"Relevant Memories:\n{memory_context}\n"
            "What is the next rational step?"
        )

        raw_output = self.gemini.generate_content(user_prompt, system_instruction=sys_prompt)
        return self._parse_json_robust(raw_output)

    def _parse_json_robust(self, text: str) -> ReasoningOutput:
        """
        Attempts to parse JSON with a regex fallback if the LLM output is malformed.
        """
        try:
            # Try direct parse
            clean_text = text.strip()
            # Remove markdown blocks if present
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3].strip()
            
            data = json.loads(clean_text)
            return ReasoningOutput(**data)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed ({e}). Attempting regex recovery.")
            
            # Regex patterns for key fields
            patterns = {
                "rationale": r'"rationale":\s*"(.*?)"',
                "proposed_action": r'"proposed_action":\s*"(.*?)"',
                "expected_outcome": r'"expected_outcome":\s*"(.*?)"',
                "is_terminal": r'"is_terminal":\s*(true|false)',
                "confidence_score": r'"confidence_score":\s*([\d\.]+)'
            }
            
            extracted = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    val = match.group(1).strip()
                    if key == "is_terminal":
                        extracted[key] = val.lower() == "true"
                    elif key == "confidence_score":
                        extracted[key] = float(val)
                    else:
                        extracted[key] = val
                else:
                    # Defaults
                    if key == "is_terminal": extracted[key] = False
                    elif key == "confidence_score": extracted[key] = 0.5
                    else: extracted[key] = "Unknown (recovery)"
            
            return ReasoningOutput(
                rationale=extracted.get("rationale"),
                proposed_action=extracted.get("proposed_action"),
                expected_outcome=extracted.get("expected_outcome"),
                is_terminal=extracted.get("is_terminal"),
                confidence_score=extracted.get("confidence_score")
            )

    def _dispatch_action(self, reasoning: ReasoningOutput) -> ActionResult:
        """
        Dispatches the proposed action to the environment/tools.
        Placeholder for systems integration with actual tools.
        """
        # Example simulation
        logger.info(f"Action: {reasoning.proposed_action}")
        
        # This would normally be a call to a ToolManager or REPL environment
        observation = f"Simulated outcome of {reasoning.proposed_action} for {reasoning.expected_outcome}"
        
        return ActionResult(
            observation=observation,
            success=True,
            outcome_score=0.9 if reasoning.confidence_score > 0.8 else 0.5
        )

    def _commit_to_memory(self, reasoning: ReasoningOutput, result: ActionResult):
        """
        Adds the current interaction to the episodic memory system.
        """
        entry = MemoryEntry(
            state=self.state.current_context or self.state.task_description,
            reasoning=reasoning.rationale,
            action=reasoning.proposed_action,
            outcome=result.observation,
            outcome_score=result.outcome_score
        )
        
        # Note: The EpisodicMemorySystem.add_memory accepts the memory entry dataclass.
        # Our pydantic model is compatible if we pass it correctly or convert.
        # Given memory/base.py exists, we should be careful. 
        # But we designed models.py to be compatible.
        self.memory.add_memory(entry)

    def _format_retrieved_memories(self, memories: List[Tuple[Any, float]]) -> str:
        """Helper to format memory objects for the prompt."""
        if not memories:
            return "No prior relevant experiences found."
        
        lines = []
        for i, (m, score) in enumerate(memories):
            # Assume m has state, action, outcome attributes (as per MemoryEntry)
            lines.append(f"{i+1}. [State: {m.state}] -> [Action: {m.action}] -> [Outcome: {m.outcome}] (Score: {score:.2f})")
        return "\n".join(lines)
