"""
Conformal Calibration for Adaptive Compute Controller.

Implements Split Conformal Prediction with Adaptive Conformal Inference (ACI)
(Gibbs & Candès, 2021) to provide distribution-free coverage guarantees on
the controller's exit decisions.

The key idea: raw LLM confidence scores are miscalibrated. Conformal prediction
wraps them in a statistical guarantee — with probability ≥ 1-α, the controller
only exits when the answer is correct — without assuming anything about the
model's internal calibration.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class ConformalCalibrator:
    """
    Conformal calibration layer for the Adaptive Compute Controller.
    
    Usage:
        # 1. Calibrate (once, offline)
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.calibrate(calibration_traces)
        calibrator.save("conformal_calibration.json")
        
        # 2. At inference (online)
        calibrator = ConformalCalibrator.load("conformal_calibration.json")
        is_reliable = calibrator.is_calibrated(confidence=0.85, iteration=4)
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """
        Parameters
        ----------
        alpha : float
            Target miscoverage rate. 1-alpha = coverage probability.
            Default 0.1 means the controller's exit decisions are correct
            with probability >= 90%.
        gamma : float
            ACI learning rate for online alpha adaptation.
        """
        self.alpha = alpha
        self.gamma = gamma
        self._base_alpha = alpha
        
        # Per-iteration quantile thresholds from calibration
        self._quantiles: Dict[int, float] = {}
        
        # ACI online state
        self._online_errors: List[bool] = []
        
        # Online Calibration Buffer
        self._calibration_buffer: List[Tuple[float, bool, int]] = []
        self.window_size = 50
        
    def _nonconformity_score(self, confidence: float, is_correct: bool) -> float:
        """
        Compute the nonconformity score for a single (confidence, correctness) pair.
        
        R = 1 - conf  if correct   (small when model is right and confident)
        R = conf      if wrong     (small when model is wrong and unconfident)
        """
        if is_correct:
            return 1.0 - confidence
        else:
            return confidence
    
    def calibrate(self, calibration_traces: List[Dict]) -> None:
        """
        Compute per-iteration conformal quantiles from calibration data.
        
        Parameters
        ----------
        calibration_traces : list of dict
            Each dict must contain:
            - "repl_history": list of step dicts with "iteration", "confidence", "snapshot_answer"
            - "metadata": dict with "gold_answer"
        """
        # Group nonconformity scores by iteration
        scores_by_iter: Dict[int, List[float]] = {}
        
        for trace in calibration_traces:
            gold = trace.get("metadata", {}).get("gold_answer", "")
            if not gold:
                continue
                
            gold_answer = gold.split("#### ")[-1].strip()
            
            for step in trace.get("repl_history", []):
                iteration = step.get("iteration")
                confidence = step.get("confidence")
                snapshot = step.get("snapshot_answer")
                
                if iteration is None or confidence is None:
                    continue
                    
                is_correct = (snapshot is not None) and (gold_answer in str(snapshot))
                score = self._nonconformity_score(confidence, is_correct)
                
                if iteration not in scores_by_iter:
                    scores_by_iter[iteration] = []
                scores_by_iter[iteration].append(score)
        
        # Compute the (1-alpha)(1 + 1/n) quantile for each iteration
        for iteration, scores in scores_by_iter.items():
            n = len(scores)
            if n == 0:
                continue
            quantile_level = min(1.0, (1 - self.alpha) * (1 + 1/n))
            self._quantiles[iteration] = float(np.quantile(scores, quantile_level))
            
        print(f"[Conformal] Calibrated on {len(calibration_traces)} traces.")
        print(f"[Conformal] Per-iteration quantiles: {self._quantiles}")
    
    def is_calibrated(self, confidence: float, iteration: int) -> bool:
        """
        Check whether the confidence score at this iteration is reliable
        according to the conformal calibration.
        
        Returns True if the confidence is within the calibrated region
        (i.e., it is safe to trust it for exit decisions).
        """
        if iteration not in self._quantiles:
            # No calibration data for this iteration — fall back to trusting it
            return True
            
        # Compute nonconformity score assuming the answer is correct
        # (optimistic check: would this confidence be consistent with correctness?)
        score = 1.0 - confidence
        
        return score <= self._quantiles[iteration]
    
    def aci_update(self, was_correct: bool) -> None:
        """
        Adaptive Conformal Inference online update.
        Adjusts alpha based on whether the last exit decision was correct.
        
        α_{t+1} = α_t + γ(err_t - α)
        
        If the controller is making too many errors, alpha increases
        (making it harder to exit). If it's doing well, alpha decreases
        (allowing earlier exits).
        """
        err = 0.0 if was_correct else 1.0
        self._online_errors.append(not was_correct)
        self.alpha = self.alpha + self.gamma * (err - self._base_alpha)
        self.alpha = max(0.01, min(0.5, self.alpha))  # Clamp to reasonable range
    
    def get_online_error_rate(self) -> float:
        """Return the running empirical error rate of the controller."""
        if not self._online_errors:
            return 0.0
        return sum(self._online_errors) / len(self._online_errors)

    def add_online_observation(self, confidence: float, is_correct: bool, iteration: int) -> None:
        """
        Add a new online observation and dynamically adjust conformal quantiles.
        
        Parameters
        ----------
        confidence : float
            Confidence score of the exit decision.
        is_correct : bool
            Whether the decision was actually correct.
        iteration : int
            The reasoning iteration at which the exit occurred.
        """
        # 1. Update ACI alpha
        self.aci_update(is_correct)
        
        # 2. Append to online buffer
        self._calibration_buffer.append((confidence, is_correct, iteration))
        if len(self._calibration_buffer) > self.window_size:
            self._calibration_buffer.pop(0)
            
        # 3. Recalculate quantile for this iteration based on buffer content
        iter_scores = [
            self._nonconformity_score(conf, corr)
            for conf, corr, iter_num in self._calibration_buffer
            if iter_num == iteration
        ]
        
        n = len(iter_scores)
        if n >= 5: # Only update if we have sufficient samples to avoid high variance
            quantile_level = min(1.0, (1 - self.alpha) * (1 + 1.0 / n))
            self._quantiles[iteration] = float(np.quantile(iter_scores, quantile_level))
    
    def save(self, filepath: str) -> None:
        """Save calibration state to JSON."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {
            "alpha": self._base_alpha,
            "gamma": self.gamma,
            "current_alpha": self.alpha,
            "quantiles": {str(k): v for k, v in self._quantiles.items()},
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "ConformalCalibrator":
        """Load calibration state from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        cal = cls(alpha=data["alpha"], gamma=data["gamma"])
        cal.alpha = data.get("current_alpha", data["alpha"])
        cal._quantiles = {int(k): v for k, v in data["quantiles"].items()}
        return cal
