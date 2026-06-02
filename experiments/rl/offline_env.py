"""
Offline Reinforcement Learning Environment for the Adaptive Compute Controller.

Replays JSON traces from the fixed-budget sweep, formulating the reasoning
overshoot problem as a Markov Decision Process (MDP) for offline training.

State space: [iteration, confidence, context_length, complexity_prior, delta_confidence]
Action space: 0 (STOP/ROLLBACK), 1 (CONTINUE)
Reward: +1 (correct stop), -1 (incorrect stop), -0.03 (continue penalty)
"""

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Any, Tuple


class ReasoningOvershootEnv(gym.Env):
    """
    Gym environment that replays offline reasoning traces.
    
    The agent steps through a sequence of iterations for a single question.
    At each step, it observes the state and decides whether to stop or continue.
    """
    
    def __init__(self, traces: List[Dict[str, Any]], continuation_penalty: float = -0.03):
        super(ReasoningOvershootEnv, self).__init__()
        
        self.traces = traces
        self.continuation_penalty = continuation_penalty
        self.current_trace_idx = 0
        self.current_step = 0
        
        # Action space: 0 = STOP, 1 = CONTINUE
        self.action_space = spaces.Discrete(2)
        
        # State space: [iteration, confidence, context_length, complexity_prior, delta_confidence]
        # Using a normalized Box space
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Internal state for current episode
        self._current_trace = None
        self._max_steps = 0
        self._history = []
        
    def _get_observation(self) -> np.ndarray:
        """Construct the 5D state vector from the current trace step."""
        if self._current_trace is None or self.current_step >= len(self._history):
            return np.zeros(5, dtype=np.float32)
            
        step_data = self._history[self.current_step]
        
        # Normalize features
        norm_iter = min(1.0, step_data["iteration"] / 20.0)
        conf = step_data["confidence"]
        
        # Normalize context length (assume max 8000 for simplicity)
        norm_ctx = min(1.0, step_data.get("context_length", 0) / 8000.0)
        
        # Complexity prior (if missing, assume 0.5)
        complex_prior = self._current_trace.get("metadata", {}).get("complexity_score", 0.5)
        
        # Delta confidence
        delta_conf = 0.0
        if self.current_step > 0:
            prev_conf = self._history[self.current_step - 1]["confidence"]
            delta_conf = conf - prev_conf
            
        return np.array([
            norm_iter,
            conf,
            norm_ctx,
            complex_prior,
            delta_conf
        ], dtype=np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to the start of a new trace."""
        super().reset(seed=seed)
        
        if len(self.traces) == 0:
            return np.zeros(5, dtype=np.float32), {}
            
        # Select trace sequentially (or random if needed, but sequential is fine for replay)
        self._current_trace = self.traces[self.current_trace_idx]
        self.current_trace_idx = (self.current_trace_idx + 1) % len(self.traces)
        
        # Extract the sequence of steps
        self._history = self._current_trace.get("repl_history", [])
        if not self._history:
            # Fallback if trace format is different (e.g. from early sweeps)
            # Try to build a mock history from snapshot dicts if available
            snapshots = self._current_trace.get("snapshot_confidences", {})
            snapshot_answers = self._current_trace.get("snapshot_answers", {})
            self._history = [
                {
                    "iteration": int(k), 
                    "confidence": float(v),
                    "snapshot_answer": snapshot_answers.get(k, "")
                }
                for k, v in sorted(snapshots.items(), key=lambda x: int(x[0]))
            ]
            
        self._max_steps = len(self._history)
        self.current_step = 0
        
        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        action: 0 = STOP, 1 = CONTINUE
        """
        if self._current_trace is None or self.current_step >= self._max_steps:
            return np.zeros(5, dtype=np.float32), 0.0, True, False, {}
            
        done = False
        reward = 0.0
        info = {}
        
        gold = self._current_trace.get("metadata", {}).get("gold_answer", "")
        
        # Helper to check correctness
        def is_correct(predicted: str) -> bool:
            if not predicted: return False
            # Simple fallback check since we don't have the specific dataset loaded here
            return gold.split("#### ")[-1].strip() in str(predicted)
            
        if action == 0:  # STOP
            done = True
            
            # The agent decided to stop here.
            # In our system, stopping triggers Rollback to the peak confidence snapshot.
            # So the reward is based on whether the rollback answer is correct.
            
            # Find peak confidence up to current step
            peak_conf = -1.0
            best_ans = ""
            for i in range(self.current_step + 1):
                step_data = self._history[i]
                c = step_data.get("confidence", 0.0)
                if c > peak_conf:
                    peak_conf = c
                    best_ans = step_data.get("snapshot_answer", step_data.get("answer", ""))
                    
            if is_correct(best_ans):
                reward = 1.0
            else:
                reward = -1.0
                
        else:  # CONTINUE
            reward = self.continuation_penalty
            self.current_step += 1
            
            if self.current_step >= self._max_steps:
                done = True
                # Forced to stop at the end. Evaluate peak.
                peak_conf = -1.0
                best_ans = ""
                for i in range(self._max_steps):
                    step_data = self._history[i]
                    c = step_data.get("confidence", 0.0)
                    if c > peak_conf:
                        peak_conf = c
                        best_ans = step_data.get("snapshot_answer", step_data.get("answer", ""))
                
                if is_correct(best_ans):
                    reward += 1.0
                else:
                    reward -= 1.0
                    
        obs = self._get_observation()
        return obs, reward, done, False, info


def load_traces_from_dir(directory: str) -> List[Dict]:
    """Helper to load JSON traces from a directory."""
    traces = []
    if not os.path.exists(directory):
        return traces
        
    for filename in os.listdir(directory):
        if filename.endswith(".json") and filename.startswith("trace_"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    traces.append(json.load(f))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return traces
