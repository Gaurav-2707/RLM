"""
Data models for the Adaptive Compute Controller (ACC).
"""

from dataclasses import dataclass, field
from typing import List
import time


@dataclass
class DepthRecord:
    """
    Records a single depth-selection decision made by the ACC at one step.

    Attributes:
        step:            The decision step index (0-based) within the episode.
        complexity_score: The raw complexity score received from the environment.
        depth_selected:  The reasoning depth (1, 2, or 3) chosen by the ACC.
        api_calls_used:  Cumulative API calls consumed BEFORE this step.
        timestamp:       Unix timestamp of when this decision was made.
    """
    step: int
    complexity_score: float
    depth_selected: int
    api_calls_used: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "complexity_score": self.complexity_score,
            "depth_selected": self.depth_selected,
            "api_calls_used": self.api_calls_used,
            "timestamp": self.timestamp,
        }


@dataclass
class EpisodeReport:
    """
    Post-hoc summary of all depth decisions across a full episode.

    Attributes:
        records:          Ordered list of every DepthRecord in the episode.
        total_api_calls:  Total API calls consumed during the episode.
        budget_exhausted: True if the budget cap was hit before the episode ended.
    """
    records: List[DepthRecord] = field(default_factory=list)
    total_api_calls: int = 0
    budget_exhausted: bool = False

    # ------------------------------------------------------------------ #
    # Convenience properties for analysis                                  #
    # ------------------------------------------------------------------ #

    @property
    def depth_distribution(self) -> dict:
        """Returns the count of each depth tier selected across the episode."""
        dist = {1: 0, 2: 0, 3: 0}
        for r in self.records:
            dist[r.depth_selected] = dist.get(r.depth_selected, 0) + 1
        return dist

    @property
    def is_non_uniform(self) -> bool:
        """
        Returns True if the depth distribution is genuinely non-uniform,
        i.e. more than one distinct depth tier was selected.
        """
        used = {r.depth_selected for r in self.records}
        return len(used) > 1

    @property
    def average_complexity(self) -> float:
        """Returns the mean complexity score over all steps."""
        if not self.records:
            return 0.0
        return sum(r.complexity_score for r in self.records) / len(self.records)

    @property
    def depth_complexity_correlation(self) -> float:
        """
        Computes the Pearson correlation coefficient between complexity scores
        and depth selected. A positive value confirms that the ACC is correctly
        assigning deeper reasoning to more complex steps.
        """
        n = len(self.records)
        if n < 2:
            return 0.0

        scores = [r.complexity_score for r in self.records]
        depths = [float(r.depth_selected) for r in self.records]

        mean_s = sum(scores) / n
        mean_d = sum(depths) / n

        cov = sum((s - mean_s) * (d - mean_d) for s, d in zip(scores, depths)) / n
        std_s = (sum((s - mean_s) ** 2 for s in scores) / n) ** 0.5
        std_d = (sum((d - mean_d) ** 2 for d in depths) / n) ** 0.5

        if std_s == 0.0 or std_d == 0.0:
            return 0.0

        return cov / (std_s * std_d)

    def summary(self) -> dict:
        """Returns a concise summary dictionary for logging or reporting."""
        return {
            "total_steps": len(self.records),
            "total_api_calls": self.total_api_calls,
            "budget_exhausted": self.budget_exhausted,
            "depth_distribution": self.depth_distribution,
            "is_non_uniform_distribution": self.is_non_uniform,
            "average_complexity": round(self.average_complexity, 4),
            "depth_complexity_correlation": round(self.depth_complexity_correlation, 4),
        }
