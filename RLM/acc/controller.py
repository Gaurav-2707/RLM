"""
Adaptive Compute Controller (ACC).

Receives a complexity score at each decision step and selects a
reasoning depth d via a three-tier mapping rule:

    Score < 0.35          →  d = 1  (shallow reasoning)
    Score 0.35 – 0.70     →  d = 2  (medium reasoning)
    Score > 0.70          →  d = 3  (deep reasoning)

A budget enforcement mechanism caps the total API calls per episode
at a configurable maximum.  The controller records every depth choice
throughout the episode for post-hoc analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import DepthRecord, EpisodeReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Depth-tier thresholds (from the paper spec)
# ---------------------------------------------------------------------------

_THRESHOLD_SHALLOW = 0.35  # score < this  → depth 1
_THRESHOLD_DEEP    = 0.70  # score > this  → depth 3
# otherwise depth 2


class AdaptiveComputeController:
    """
    Selects reasoning depth based on complexity score with budget enforcement.

    Parameters
    ----------
    max_api_calls : int
        Hard cap on total API calls allowed within one episode.
        When the remaining budget would be exhausted by the requested
        depth, the depth is clamped to whatever the budget allows.
        Set to ``None`` to disable budget enforcement entirely.
    depth_costs : dict[int, int]
        How many API calls each depth tier consumes per step.
        Defaults to {1: 1, 2: 2, 3: 3}.

    Example
    -------
    >>> acc = AdaptiveComputeController(max_api_calls=20)
    >>> acc.new_episode()
    >>> depth = acc.select_depth(complexity_score=0.82)  # returns 3
    >>> report = acc.end_episode()
    >>> print(report.summary())
    """

    # default cost per depth tier (number of API calls consumed)
    _DEFAULT_DEPTH_COSTS: dict[int, int] = {1: 1, 2: 2, 3: 3}

    def __init__(
        self,
        max_api_calls: Optional[int] = 10,
        depth_costs: Optional[dict[int, int]] = None,
    ) -> None:
        self.max_api_calls = max_api_calls
        self.depth_costs: dict[int, int] = depth_costs or dict(
            self._DEFAULT_DEPTH_COSTS
        )

        # Episode state (reset via new_episode())
        self._step: int = 0
        self._api_calls_used: int = 0
        self._records: list[DepthRecord] = []
        self._budget_exhausted: bool = False

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def new_episode(self) -> None:
        """Reset all episode state.  Must be called before each new episode."""
        self._step = 0
        self._api_calls_used = 0
        self._records = []
        self._budget_exhausted = False
        logger.debug(
            "ACC: new episode started (budget=%s).", self.max_api_calls
        )

    def end_episode(self) -> EpisodeReport:
        """
        Finalise the current episode and return a full EpisodeReport.

        Returns
        -------
        EpisodeReport
            Immutable report containing all DepthRecords and summary stats.
        """
        report = EpisodeReport(
            records=list(self._records),
            total_api_calls=self._api_calls_used,
            budget_exhausted=self._budget_exhausted,
        )
        logger.info(
            "ACC: episode ended. %s",
            report.summary(),
        )
        return report

    # ------------------------------------------------------------------ #
    # Core decision logic                                                  #
    # ------------------------------------------------------------------ #

    def select_depth(self, complexity_score: float) -> int:
        """
        Select reasoning depth for the current step.

        Parameters
        ----------
        complexity_score : float
            A value in [0, 1] representing problem complexity at this step.

        Returns
        -------
        int
            The chosen depth (1, 2, or 3).  Returns 0 if the budget is
            already exhausted (caller should terminate the episode).

        Raises
        ------
        ValueError
            If select_depth is called before new_episode().
        """
        if self._budget_exhausted:
            logger.warning(
                "ACC: budget already exhausted; returning depth 0 (stop signal)."
            )
            return 0

        # --- Three-tier mapping rule ---
        raw_depth = self._map_score_to_depth(complexity_score)

        # --- Budget enforcement ---
        depth = self._apply_budget(raw_depth)

        # --- Record the decision ---
        record = DepthRecord(
            step=self._step,
            complexity_score=complexity_score,
            depth_selected=depth,
            api_calls_used=self._api_calls_used,
        )
        self._records.append(record)

        # Consume API calls for the chosen depth
        cost = self.depth_costs.get(depth, depth)
        self._api_calls_used += cost
        self._step += 1

        logger.debug(
            "ACC step %d | score=%.4f | raw_depth=%d | selected_depth=%d "
            "| cost=%d | total_calls=%d/%s",
            self._step - 1,
            complexity_score,
            raw_depth,
            depth,
            cost,
            self._api_calls_used,
            self.max_api_calls or "∞",
        )

        return depth

    # ------------------------------------------------------------------ #
    # Read-only properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def api_calls_used(self) -> int:
        """Total API calls consumed in the current episode so far."""
        return self._api_calls_used

    @property
    def remaining_budget(self) -> Optional[int]:
        """
        Remaining API call budget, or None if no budget is set.
        """
        if self.max_api_calls is None:
            return None
        return max(self.max_api_calls - self._api_calls_used, 0)

    @property
    def is_budget_exhausted(self) -> bool:
        """True when no more API calls can be made in this episode."""
        return self._budget_exhausted

    @property
    def records(self) -> list[DepthRecord]:
        """Read-only snapshot of all DepthRecords so far in the episode."""
        return list(self._records)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _map_score_to_depth(score: float) -> int:
        """
        Apply the three-tier mapping rule.

            Score < 0.35        →  d = 1
            0.35 ≤ Score ≤ 0.70 →  d = 2
            Score > 0.70        →  d = 3
        """
        if score < _THRESHOLD_SHALLOW:
            return 1
        if score <= _THRESHOLD_DEEP:
            return 2
        return 3

    def _apply_budget(self, requested_depth: int) -> int:
        """
        Clamp the requested depth to what the remaining budget allows.

        If even depth 1 cannot be afforded, marks the budget as exhausted
        and returns 0.
        """
        if self.max_api_calls is None:
            return requested_depth

        remaining = self.max_api_calls - self._api_calls_used

        # Try to honour the requested depth; fall back to shallower tiers
        for depth in range(requested_depth, 0, -1):
            cost = self.depth_costs.get(depth, depth)
            if cost <= remaining:
                if depth < requested_depth:
                    logger.info(
                        "ACC: budget constraint reduced depth %d → %d "
                        "(remaining budget: %d calls).",
                        requested_depth,
                        depth,
                        remaining,
                    )
                return depth

        # Budget fully exhausted
        self._budget_exhausted = True
        logger.warning(
            "ACC: budget exhausted after %d API calls (cap=%d).",
            self._api_calls_used,
            self.max_api_calls,
        )
        return 0
