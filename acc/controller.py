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
        
        # Task-Complexity Prior: Simpler tasks tolerate less confidence drop before rollback
        self._overshoot_threshold = 0.15 if complexity_score < 0.4 else 0.25

    def should_exit(
        self,
        current_answer: str,
        retrieved_context: str,
        llm_client: any,
        confidence: float = 1.0,
        iteration: int = 1
    ) -> dict:
        """
        Evaluates whether to exit based on Marginal Compute Utility and NLI grounding.
        """
        if not hasattr(self, '_confidence_history'):
            self._confidence_history = []
        if not hasattr(self, '_records'):
            self._records = []
            
        self._confidence_history.append(confidence)
        
        # 1. Conformal gate: if calibrator is loaded, check whether confidence
        #    at this iteration is within the calibrated region before trusting it.
        if self._conformal is not None:
            if not self._conformal.is_calibrated(confidence, iteration):
                reason = f"Conformal gate: confidence {confidence} at iter {iteration} is outside calibrated region. Blocking exit."
                self._records.append(reason)
                logger.info(f"ACC: {reason}")
                return {"exit": False, "reason": reason, "warning": False}
        
        # 2. Marginal Utility Check (Overshoot detection)
        if len(self._confidence_history) > 2:
            peak_conf = max(self._confidence_history[:-1])
            peak_idx = self._confidence_history.index(peak_conf)
            threshold = getattr(self, '_overshoot_threshold', 0.2)
            
            if confidence < peak_conf - threshold:
                reason = f"Marginal utility dropped: confidence fell from peak {peak_conf} to {confidence}."
                self._records.append(reason)
                logger.info(f"ACC Exit: {reason}")
                return {
                    "exit": True, 
                    "reason": reason, 
                    "warning": True,
                    "rollback": True,
                    "peak_iteration": peak_idx + 1
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
            
    @property
    def records(self):
        return getattr(self, '_records', [])

class RLController:
    """
    RL-trained Adaptive Compute Controller.
    
    Uses a DQN policy trained offline on sweep traces to decide when to stop.
    Evaluates the 5D state vector: [iteration, confidence, context_length, complexity_prior, delta_confidence]
    """
    def __init__(self, weights_path: str = "RLM/weights/dqn_controller.zip"):
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
