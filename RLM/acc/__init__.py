"""
Adaptive Compute Controller (ACC) for the RLM framework.

The ACC dynamically selects reasoning depth based on measured
complexity scores, while enforcing a configurable API call budget
across an episode.
"""

from .controller import AdaptiveComputeController
from .complexity import ComplexityScorer
from .models import DepthRecord, EpisodeReport

__all__ = [
    "AdaptiveComputeController",
    "ComplexityScorer",
    "DepthRecord",
    "EpisodeReport",
]
