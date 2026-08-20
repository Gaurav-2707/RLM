"""
RLM-0 training package.

Turns objectively-verified `RLMTrajectory` objects into a reward-weighted
supervised fine-tuning corpus for Llama 3.1 8B (QLoRA), and provides the
P1-P4 "recursion proof" evaluation harness.

Modules:
  tokens     - control-token vocabulary + tokenizer surgery
  render     - RLMTrajectory -> grammar string + span metadata
  collator   - masked, reward/role-weighted collator (labels, tok_weights, ul_mask)
  loss       - three-term objective (CE + unlikelihood + rollback budget)
  data       - trajectory loading, r_hat, replay-mix dataset
  train      - QLoRA driver (checkpoint on best P3, forgetting tripwire)
  eval       - P1-P4 recursion proof protocol
"""

from .tokens import CONTROL_TOKENS, add_control_tokens, conf_to_token, CONF_BIN_EDGES
from .render import render_trajectory, RenderedTrajectory
from .collator import RLMCollator, encode_trajectory
from .loss import recursion_loss, LossWeights
from .data import (load_trajectories, compute_r_hat, TrajectoryDataset,
                   ReplayMixDataset)

__all__ = [
    "CONTROL_TOKENS",
    "add_control_tokens",
    "conf_to_token",
    "CONF_BIN_EDGES",
    "render_trajectory",
    "RenderedTrajectory",
    "RLMCollator",
    "encode_trajectory",
    "recursion_loss",
    "LossWeights",
    "load_trajectories",
    "compute_r_hat",
    "TrajectoryDataset",
    "ReplayMixDataset",
]
