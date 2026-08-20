"""
Dataset utilities for RLM-0 training.

Loads RLMTrajectory JSON files from disk, renders them to grammar spans,
and exposes an IterableDataset-friendly list. Also computes r_hat (mean
rollbacks among successful trajectories) for the rollback-budget penalty,
and supports blending a small fraction of general instruction data to
mitigate catastrophic forgetting (spec §5).
"""

import os
import glob
import random
from typing import List, Optional

from RLM.utils.trajectory import RLMTrajectory
from .render import render_trajectory, RenderedTrajectory


def load_trajectories(directory: str, verified_only: bool = True) -> List[RLMTrajectory]:
    out = []
    for path in glob.glob(os.path.join(os.path.expanduser(directory), "*.json")):
        try:
            t = RLMTrajectory.load(path)
        except Exception:
            continue
        if verified_only and not t.verified:
            continue
        out.append(t)
    return out


def compute_r_hat(trajectories: List[RLMTrajectory]) -> float:
    succ = [t for t in trajectories if t.final_outcome]
    if not succ:
        return 1.0
    return sum(t.total_rollbacks for t in succ) / len(succ)


class TrajectoryDataset:
    """Render-on-load dataset of RenderedTrajectory; pairs with RLMCollator."""

    def __init__(self, trajectories: List[RLMTrajectory], **render_kwargs):
        self.rendered: List[RenderedTrajectory] = [
            render_trajectory(t, **render_kwargs) for t in trajectories
        ]

    def __len__(self):
        return len(self.rendered)

    def __getitem__(self, i):
        return self.rendered[i]


class ReplayMixDataset:
    """
    Blend trajectory examples with a fraction of general-instruction examples.

    `general` items must already be encoded dicts (input_ids/labels/...),
    or RenderedTrajectory; RLMCollator handles both. General items should
    carry traj_weight=1.0 and is_success irrelevant (no UL/rollback spans).
    """

    def __init__(self, traj_ds, general_items: List, replay_frac: float = 0.04,
                 seed: int = 0):
        self.traj_ds = traj_ds
        self.general = general_items
        self.replay_frac = replay_frac
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.traj_ds)

    def __getitem__(self, i):
        if self.general and self.rng.random() < self.replay_frac:
            return self.rng.choice(self.general)
        return self.traj_ds[i]
