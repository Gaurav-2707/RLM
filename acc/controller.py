"""
Adaptive Compute Controller (ACC) - rewritten for True Adaptive Compute.

Now relies on tool-driven budget extension rather than static score mappings.
Includes a rigorous NLI-based grounding gate to verify answers.
Optionally wraps confidence signals in conformal prediction guarantees.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AdaptiveComputeController:
    """
    Manages the rigorous NLI grounding gate for true adaptive compute.
    The budget extension is handled via tools in the REPL.
    """
    def __init__(self, max_grounding_retries: int = 3, conformal_calibrator=None):
        self.max_grounding_retries = max_grounding_retries
        self._grounding_attempts = 0
        self._conformal = conformal_calibrator

    def new_episode(self, complexity_score: float = 0.5) -> None:
        self._grounding_attempts = 0
        self._confidence_history = []
        self._records = []
        self._cumulative_cost = 0.0
        self._rollback_count = 0
        self.max_rollback_retries = 2
        
        # Task-Complexity Prior: Simpler tasks tolerate less confidence drop before rollback
        self._overshoot_threshold = 0.15 if complexity_score < 0.4 else 0.25
        self._max_budget = 0.3 if complexity_score < 0.4 else 0.6

    def should_exit(
        self,
        current_answer: str,
        retrieved_context: str,
        llm_client: any,
        confidence: float = 1.0,
        iteration: int = 1,
        executed_code: Optional[str] = None
    ) -> dict:
        """
        Evaluates whether to exit based on Marginal Compute Utility and NLI grounding.
        """
        if not hasattr(self, '_confidence_history'):
            self._confidence_history = []
        if not hasattr(self, '_records'):
            self._records = []
        if not hasattr(self, '_cumulative_cost'):
            self._cumulative_cost = 0.0
            
        self._confidence_history.append(confidence)
        
        # 0. Dynamic Cost Estimation
        base_cost = 0.02
        action_cost = 0.0
        if executed_code:
            if "deep_reason" in executed_code:
                action_cost += 0.15
            elif "llm_query" in executed_code:
                action_cost += 0.05
            elif "search_context" in executed_code:
                action_cost += 0.03
            else:
                action_cost += 0.01
        step_cost = base_cost + action_cost
        self._cumulative_cost += step_cost

        # Conformal gate: if calibrator is loaded, check whether confidence
        # at this iteration is within the calibrated region before trusting it.
        if self._conformal is not None:
            if not self._conformal.is_calibrated(confidence, iteration):
                reason = f"Conformal gate: confidence {confidence} at iter {iteration} is outside calibrated region. Blocking exit."
                self._records.append(reason)
                logger.info(f"ACC: {reason}")
                return {"exit": False, "reason": reason, "warning": False}

        # Check budget limit
        if self._cumulative_cost > getattr(self, '_max_budget', 0.5):
            reason = f"Budget exhausted: cumulative cost {self._cumulative_cost:.2f} exceeded max budget {self._max_budget:.2f}."
            self._records.append(reason)
            logger.info(f"ACC Exit: {reason}")
            # If we overshoot budget, trigger exit but rollback to peak if confidence is low
            if len(self._confidence_history) > 1:
                peak_conf = max(self._confidence_history[:-1])
                peak_idx = self._confidence_history.index(peak_conf)
                if confidence < peak_conf - 0.1:
                    return {
                        "exit": True,
                        "reason": reason + " Rolling back to peak answer.",
                        "warning": True,
                        "rollback": True,
                        "peak_iteration": peak_idx + 1
                    }
            return {"exit": True, "reason": reason, "warning": True}
        
        # 2. Marginal Utility Check (Overshoot detection)
        if len(self._confidence_history) > 2:
            peak_conf = max(self._confidence_history[:-1])
            peak_idx = self._confidence_history.index(peak_conf)
            threshold = getattr(self, '_overshoot_threshold', 0.2)
            
            if confidence < peak_conf - threshold:
                reason = f"Marginal utility dropped: confidence fell from peak {peak_conf} to {confidence}."
                self._records.append(reason)
                logger.info(f"ACC Exit/Rollback: {reason}")
                self._rollback_count += 1
                # If we still have rollback retries remaining, we can suggest a non-terminal rollback
                terminal_exit = self._rollback_count > self.max_rollback_retries
                return {
                    "exit": terminal_exit, 
                    "reason": reason, 
                    "warning": True,
                    "rollback": True,
                    "peak_iteration": peak_idx + 1,
                    "terminal_rollback": terminal_exit
                }
                
        # 3. Strict Natural Language Inference (NLI) grounding gate.
        # The model must output ONLY 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'.
        prompt = (
            "You are a strict Natural Language Inference (NLI) Grounding Verifier.\n"
            "Evaluate whether the provided CONTEXT entails the ANSWER.\n\n"
            f"CONTEXT:\n{retrieved_context[:5000]}\n\n"
            f"ANSWER:\n{current_answer}\n\n"
            "INSTRUCTION:\n"
            "Output EXACTLY and ONLY one of the following words:\n"
            "- ENTAILMENT: The answer is fully supported and logically follows from the context.\n"
            "- CONTRADICTION: The context explicitly contradicts the answer.\n"
            "- NEUTRAL: The context does not provide enough information to support or contradict the answer.\n\n"
            "Do not output anything else."
        )

        try:
            grounding_result = llm_client.completion(prompt, max_tokens=10).strip().upper()
        except Exception as e:
            logger.error(f"ACC Grounding Check Failed: {e}")
            return {"exit": True, "reason": f"Grounding check error: {e}", "warning": True}

        if "ENTAILMENT" in grounding_result:
            self._grounding_attempts = 0
            reason = "Grounding verified via NLI entailment."
            self._records.append(reason)
            return {
                "exit": True, 
                "reason": reason, 
                "warning": False, 
                "evidence": grounding_result
            }
        else:
            reason = f"Grounding check returned {grounding_result}. Blocking exit."
            self._records.append(reason)
            logger.info(f"ACC: {reason}")
            return {"exit": False, "reason": grounding_result, "warning": False}
            
    def update_conformal(self, is_correct: bool, confidence: float, iteration: int) -> None:
        """Adds an online conformal observation to adjust safety regions dynamically."""
        if self._conformal is not None:
            self._conformal.add_online_observation(confidence, is_correct, iteration)

    def filter_context_for_rl(self, memory_system: any, state_query: str, max_tokens: int = 4000) -> str:
        """
        Uses the Episodic Graph Memory (RLM Context Manager) to filter the full codebase
        down to a strict token limit before feeding it to the Test-Driven RL agent.
        """
        scored_memories, conflicts = memory_system.retrieve(state_query, top_k=5)
        
        filtered_context = ""
        if conflicts:
            filtered_context += "WARNING CONFLICTS DETECTED:\n" + "\n".join(conflicts) + "\n\n"
            
        for mem, score in scored_memories:
            filtered_context += f"--- Action: {mem.action} (Score: {score:.2f}) ---\n"
            filtered_context += str(mem.outcome) + "\n"
            
        if len(filtered_context) > max_tokens:
            filtered_context = filtered_context[:max_tokens] + "\n...[TRUNCATED BY RLM]"
            
        return filtered_context

    def end_episode(self) -> dict:
        """Close the current episode and return a summary report."""
        return {
            "total_steps": len(getattr(self, "_confidence_history", [])),
            "total_rollbacks": getattr(self, "_rollback_count", 0),
            "cumulative_cost": round(getattr(self, "_cumulative_cost", 0.0), 4),
            "peak_confidence": max(self._confidence_history, default=0.0)
                if getattr(self, "_confidence_history", None) else 0.0,
            "records": list(getattr(self, "_records", [])),
        }

    @property
    def records(self):
        return getattr(self, '_records', [])

class RLController:
    """
    RL-trained Adaptive Compute Controller.
    
    Uses a DQN policy trained offline on sweep traces to decide when to stop.
    Evaluates the 5D state vector: [iteration, confidence, context_length, complexity_prior, delta_confidence]
    """
    def __init__(self, weights_path: str = "weights/dqn_controller.zip"):
        self.weights_path = weights_path
        self._model = None
        self._records = []
        self._confidence_history = []
        
        try:
            from stable_baselines3 import DQN
            if __import__("os").path.exists(weights_path):
                self._model = DQN.load(weights_path)
                logger.info(f"RLController loaded weights from {weights_path}")
            else:
                logger.warning(f"RL weights not found at {weights_path}. Will fallback to heuristic.")
        except ImportError:
            logger.error("stable-baselines3 not installed. RLController cannot function.")
            
    def new_episode(self, complexity_score: float = 0.5) -> None:
        self._confidence_history = []
        self._records = []
        self._complexity_prior = complexity_score
        
    def should_exit(
        self,
        current_answer: str,
        retrieved_context: str,
        llm_client: any,
        confidence: float = 1.0,
        iteration: int = 1
    ) -> dict:
        self._confidence_history.append(confidence)
        
        # Fallback if no model
        if self._model is None:
            if confidence > 0.9:
                return {"exit": True, "reason": "Fallback exit", "warning": False}
            return {"exit": False, "reason": "No RL model, continue", "warning": False}
            
        import numpy as np
        
        # Construct 5D State
        norm_iter = min(1.0, iteration / 20.0)
        norm_ctx = min(1.0, len(retrieved_context) / 8000.0)
        
        delta_conf = 0.0
        if len(self._confidence_history) > 1:
            delta_conf = confidence - self._confidence_history[-2]
            
        state = np.array([
            norm_iter,
            confidence,
            norm_ctx,
            getattr(self, "_complexity_prior", 0.5),
            delta_conf
        ], dtype=np.float32)
        
        # Predict action: 0 = STOP, 1 = CONTINUE
        action, _ = self._model.predict(state, deterministic=True)
        
        if action == 0:
            # RL agent decided to stop. Check if we should rollback.
            # If the current confidence is significantly lower than a past peak, rollback.
            peak_conf = max(self._confidence_history)
            peak_idx = self._confidence_history.index(peak_conf)
            
            reason = f"RL Policy Action=STOP. Conf={confidence:.2f}, State={state}"
            self._records.append(reason)
            logger.info(f"RLController Exit: {reason}")
            
            if confidence < peak_conf - 0.1:
                return {
                    "exit": True, 
                    "reason": reason, 
                    "warning": True,
                    "rollback": True,
                    "peak_iteration": peak_idx + 1
                }
            return {"exit": True, "reason": reason, "warning": False}
            
        # action == 1
        reason = f"RL Policy Action=CONTINUE. State={state}"
        self._records.append(reason)
        return {"exit": False, "reason": reason, "warning": False}
        
    @property
    def records(self):
        return getattr(self, '_records', [])
