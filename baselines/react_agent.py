"""
ReAct Baseline Agent — Minimal chain-of-thought + tool-call agent.

This is NOT the full RLM/REPL pipeline. It is a deliberately simple
ReAct-style agent (Yao et al., 2023) used to demonstrate that Reasoning
Overshoot is a universal phenomenon of iterative agents, not an artifact
of the RLM architecture.

The agent follows the Thought → Action → Observation loop without a
sandboxed REPL. Tool calls are simulated as LLM self-queries.
"""

from typing import Optional, List, Dict, Any
from RLM.utils.llm import LLMClient, DEFAULT_MODEL


class ReActAgent:
    """
    Minimal ReAct agent for baseline comparison.
    
    Parameters
    ----------
    model : str
        LLM model string.
    max_iterations : int
        Maximum number of Thought-Action-Observation loops.
    force_iterations : bool
        If True, force the agent to use all iterations (for overshoot study).
    """
    
    def __init__(self, model: str = None, max_iterations: int = 10,
                 force_iterations: bool = False):
        self.model = model or DEFAULT_MODEL
        self.llm = LLMClient(model=self.model)
        self.max_iterations = max_iterations
        self.force_iterations = force_iterations
        self._snapshot_answers: Dict[int, str] = {}
        self._snapshot_confidences: Dict[int, float] = {}
    
    def completion(self, question: str) -> str:
        """
        Run the ReAct loop on a question.
        Returns the final answer (or peak-confidence snapshot if force_iterations).
        """
        system_prompt = (
            "You are a reasoning agent. For each step, output exactly:\n"
            "Thought: <your reasoning>\n"
            "Action: <what you would do next, or 'FINISH' if done>\n"
            "Answer: <your current best answer to the question>\n"
            "Confidence: <0.0 to 1.0>\n\n"
            "You must always provide Answer and Confidence even if uncertain."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        self._snapshot_answers = {}
        self._snapshot_confidences = {}
        self.trace_history: List[Dict[str, Any]] = []
        
        final_answer = "unknown"
        
        for iteration in range(1, self.max_iterations + 1):
            prompt_addition = f"\n\nThis is reasoning step {iteration} of {self.max_iterations}. Think carefully."
            
            response = self.llm.completion(
                messages + [{"role": "user", "content": prompt_addition}]
            )
            
            # Parse the structured response
            answer = self._extract_field(response, "Answer")
            confidence = self._extract_confidence(response)
            action = self._extract_field(response, "Action")
            
            if answer:
                self._snapshot_answers[iteration] = answer
                self._snapshot_confidences[iteration] = confidence
            
            # Compute context length
            context_length = sum(len(str(m.get("content", ""))) for m in messages)
            
            self.trace_history.append({
                "iteration": iteration,
                "response": response,
                "snapshot_answer": answer,
                "confidence": confidence,
                "context_length": context_length,
                "action": action,
            })
            
            # Append to conversation history
            messages.append({"role": "assistant", "content": response})
            
            if action and "FINISH" in action.upper() and not self.force_iterations:
                final_answer = answer or "unknown"
                break
            
            # Add observation (self-reflection prompt)
            messages.append({
                "role": "user",
                "content": "Observation: Review your previous answer. Is it correct? "
                           "If not, revise it. Continue reasoning."
            })
        
        # Final answer selection
        if self.force_iterations and self._snapshot_answers:
            # Return the last snapshot
            last_iter = max(self._snapshot_answers.keys())
            final_answer = self._snapshot_answers[last_iter]
        elif self._snapshot_answers:
            last_iter = max(self._snapshot_answers.keys())
            final_answer = self._snapshot_answers[last_iter]
        
        return final_answer
    
    def get_peak_confidence_answer(self) -> str:
        """Return the answer from the iteration with the highest confidence."""
        if not self._snapshot_confidences:
            return "unknown"
        peak_iter = max(self._snapshot_confidences, key=self._snapshot_confidences.get)
        return self._snapshot_answers.get(peak_iter, "unknown")
    
    def _extract_field(self, text: str, field: str) -> Optional[str]:
        """Extract a field value from structured ReAct output."""
        for line in text.split("\n"):
            if line.strip().startswith(f"{field}:"):
                return line.split(":", 1)[1].strip()
        return None
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence value from structured ReAct output."""
        raw = self._extract_field(text, "Confidence")
        if raw:
            try:
                return float(raw.strip())
            except ValueError:
                return 0.5
        return 0.5
