"""
GridWorld Experiment Runner.

Orchestrates controlled experiments across four compute modes:
    - baseline:       d=0, random actions (no LLM)
    - fixed_shallow:  d=1, single LLM call per step
    - fixed_deep:     d=3, three LLM calls per step
    - adaptive:       rule-based ACC selects depth per step

Produces structured CSV logs for analysis:
    - steps.csv:    per-step records (episode, step, pos, action, complexity, depth, reward, ...)
    - episodes.csv: per-episode summaries (success, total_steps, total_depth, ...)
    - config.json:  full experiment configuration for reproducibility
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from RLM.environments.gridworld import GridWorld
from RLM.experiments.action_parser import parse_action
from RLM.acc import AdaptiveComputeController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One row in steps.csv."""
    episode: int
    step: int
    pos_x: int
    pos_y: int
    action: str
    action_was_parsed: bool
    complexity: float
    depth: int
    reward: float
    cumulative_depth: int
    depth_budget_remaining: int
    done: bool
    success: bool


@dataclass
class EpisodeResult:
    """One row in episodes.csv."""
    episode: int
    mode: str
    seed: int
    success: bool
    total_steps: int
    total_depth: int
    total_reward: float
    budget_exhausted: bool
    max_total_depth: int
    wall_time_seconds: float
    parse_failure_count: int
    depth_distribution: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GridWorld-specific LLM prompt
# ---------------------------------------------------------------------------

_GRIDWORLD_SYSTEM_PROMPT = (
    "You are a navigation agent in a grid world. Your ONLY job is to choose "
    "the best movement direction to reach the goal.\n\n"
    "You MUST respond with EXACTLY ONE of these actions: up, down, left, right\n\n"
    "Output your chosen action on its own line, like:\n"
    "Action: right\n\n"
    "Do NOT output anything else. Just the action."
)

_GRIDWORLD_REASONING_PROMPT = (
    "You are a navigation agent in a grid world. Analyze the current state "
    "and choose the best movement direction to reach the goal.\n\n"
    "Think step by step about which direction moves you closer to the goal "
    "while avoiding obstacles.\n\n"
    "After your reasoning, you MUST output your chosen action in this format:\n"
    "Action: <direction>\n\n"
    "where <direction> is one of: up, down, left, right"
)


def _build_prompt(state_text: str, depth: int) -> list[dict[str, str]]:
    """Build LLM prompt messages for the current state."""
    system = _GRIDWORLD_REASONING_PROMPT if depth >= 2 else _GRIDWORLD_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Current state:\n{state_text}\n\nChoose your action:"},
    ]


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: GridWorld,
    controller: Callable[[float], int],
    llm_client: Optional[object] = None,
    max_total_depth: int = 150,
    episode_id: int = 0,
    mode: str = "baseline",
    seed: int = 0,
) -> tuple[EpisodeResult, list[StepRecord]]:
    """
    Run a single GridWorld episode.

    Parameters
    ----------
    env : GridWorld
        The environment instance (already reset with the desired seed).
    controller : Callable[[float], int]
        Maps complexity score → depth.
        Baseline: lambda _: 0
        Fixed:    lambda _: d
        Adaptive: acc.select_depth
    llm_client : object or None
        An object with a .completion(messages) method.
        None for baseline mode (random actions).
    max_total_depth : int
        Per-episode depth budget cap.
    episode_id : int
        Episode index for logging.
    mode : str
        Mode name for logging.
    seed : int
        Seed used for this episode.

    Returns
    -------
    tuple[EpisodeResult, list[StepRecord]]
    """
    rng = random.Random(seed)
    step_records: list[StepRecord] = []
    depth_budget_remaining = max_total_depth
    cumulative_depth = 0
    total_reward = 0.0
    parse_failures = 0
    depth_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    t_start = time.time()

    for step_idx in range(env.max_steps):
        state_text = env.get_state_text()
        complexity = env.get_complexity_score()
        valid_actions = env.get_valid_actions()

        # Select depth via controller
        raw_depth = controller(complexity)

        # Budget enforcement
        depth = min(raw_depth, depth_budget_remaining)

        if depth > 0 and llm_client is not None:
            # LLM-based action selection
            prompt = _build_prompt(state_text, depth)

            # Multi-call for deeper reasoning: chain depth calls
            # For depth > 1, we append prior reasoning into context
            response = ""
            for d in range(depth):
                if d > 0:
                    # Append prior response as context for deeper reasoning
                    prompt.append({"role": "assistant", "content": response})
                    prompt.append({
                        "role": "user",
                        "content": (
                            "Reconsider your choice. Think more carefully about "
                            "obstacles and distance. What is the best action? "
                            "Respond with:\nAction: <direction>"
                        ),
                    })
                try:
                    response = llm_client.completion(prompt)
                except Exception as e:
                    logger.error("LLM call failed at step %d depth %d: %s", step_idx, d, e)
                    response = ""
                    break

            action, was_parsed = parse_action(response, valid_actions, rng)
            if not was_parsed:
                parse_failures += 1

            depth_budget_remaining -= depth
            cumulative_depth += depth
        else:
            # Random action (baseline or budget exhausted)
            action = rng.choice(valid_actions) if valid_actions else "right"
            was_parsed = False  # N/A for random
            depth = 0

        depth_dist[depth] = depth_dist.get(depth, 0) + 1

        # Step the environment
        result = env.step(action)
        total_reward += result.reward

        # Record
        step_records.append(StepRecord(
            episode=episode_id,
            step=step_idx,
            pos_x=env.agent_pos[0],
            pos_y=env.agent_pos[1],
            action=action,
            action_was_parsed=was_parsed,
            complexity=round(complexity, 6),
            depth=depth,
            reward=round(result.reward, 4),
            cumulative_depth=cumulative_depth,
            depth_budget_remaining=depth_budget_remaining,
            done=result.done,
            success=result.success,
        ))

        if result.done:
            break

    wall_time = time.time() - t_start

    episode_result = EpisodeResult(
        episode=episode_id,
        mode=mode,
        seed=seed,
        success=result.success if step_records else False,
        total_steps=len(step_records),
        total_depth=cumulative_depth,
        total_reward=round(total_reward, 4),
        budget_exhausted=(depth_budget_remaining <= 0),
        max_total_depth=max_total_depth,
        wall_time_seconds=round(wall_time, 3),
        parse_failure_count=parse_failures,
        depth_distribution=depth_dist,
    )

    return episode_result, step_records


# ---------------------------------------------------------------------------
# Multi-episode experiment for one mode
# ---------------------------------------------------------------------------

def run_experiment_mode(
    mode: str,
    seeds: list[int],
    llm_client: Optional[object] = None,
    grid_size: int = 10,
    obstacle_fraction: float = 0.20,
    max_steps: Optional[int] = None,
    max_total_depth: int = 150,
    difficulty: Optional[str] = None,
    acc_budget: Optional[int] = None,
) -> tuple[list[EpisodeResult], list[StepRecord]]:
    """
    Run all episodes for a single experimental mode.

    Parameters
    ----------
    mode : str
        One of "baseline", "fixed_shallow", "fixed_deep", "adaptive".
    seeds : list[int]
        One seed per episode. Length determines number of episodes.
    llm_client : object or None
        LLM client with .completion() method. None for baseline.
    grid_size, obstacle_fraction : int/float
        Environment configuration (overridden by difficulty if provided).
    max_steps : int or None
        Hard episode step limit.  None → use the difficulty preset default
        (easy=60, medium=100, hard=150).  Pass an int to override, e.g. 5
        for a cheap smoke test.
    max_total_depth : int
        Per-episode depth budget cap.
    difficulty : str or None
        If provided, "easy"/"medium"/"hard" preset.
    acc_budget : int or None
        Budget for the ACC (adaptive mode only).

    Returns
    -------
    tuple[list[EpisodeResult], list[StepRecord]]
    """
    # Build controller for this mode
    if mode == "baseline":
        controller: Callable[[float], int] = lambda _: 0
    elif mode == "fixed_shallow":
        controller = lambda _: 1
    elif mode == "fixed_deep":
        controller = lambda _: 3
    elif mode == "adaptive":
        acc = AdaptiveComputeController(max_api_calls=acc_budget)
        acc.new_episode()  # will be reset per-episode below
        controller = acc.select_depth
    else:
        raise ValueError(f"Unknown mode: {mode}")

    all_episodes: list[EpisodeResult] = []
    all_steps: list[StepRecord] = []

    for i, seed in enumerate(seeds):
        logger.info("Mode=%s | Episode %d/%d | seed=%d", mode, i + 1, len(seeds), seed)

        # Create fresh environment per episode.
        # If max_steps is explicitly given, build the env with difficulty preset
        # first (to get grid_size / obstacle_fraction) then override max_steps.
        env = GridWorld(
            grid_size=grid_size,
            obstacle_fraction=obstacle_fraction,
            max_steps=max_steps or 100,   # temporary; overridden below if difficulty given
            seed=seed,
            difficulty=difficulty,
        )
        # Apply explicit max_steps override AFTER the difficulty preset has run
        if max_steps is not None:
            env.max_steps = max_steps

        # Reset ACC episode tracking for adaptive mode
        if mode == "adaptive":
            acc.new_episode()

        episode_result, step_records = run_episode(
            env=env,
            controller=controller,
            llm_client=llm_client if mode != "baseline" else None,
            max_total_depth=max_total_depth,
            episode_id=i,
            mode=mode,
            seed=seed,
        )

        all_episodes.append(episode_result)
        all_steps.extend(step_records)

        logger.info(
            "  → success=%s, steps=%d, depth=%d, reward=%.3f",
            episode_result.success,
            episode_result.total_steps,
            episode_result.total_depth,
            episode_result.total_reward,
        )

    return all_episodes, all_steps


# ---------------------------------------------------------------------------
# CSV / JSON I/O
# ---------------------------------------------------------------------------

def _write_steps_csv(records: list[StepRecord], path: Path) -> None:
    """Write per-step records to CSV."""
    if not records:
        return
    fieldnames = list(StepRecord.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "episode": r.episode,
                "step": r.step,
                "pos_x": r.pos_x,
                "pos_y": r.pos_y,
                "action": r.action,
                "action_was_parsed": r.action_was_parsed,
                "complexity": r.complexity,
                "depth": r.depth,
                "reward": r.reward,
                "cumulative_depth": r.cumulative_depth,
                "depth_budget_remaining": r.depth_budget_remaining,
                "done": r.done,
                "success": r.success,
            })


def _write_episodes_csv(results: list[EpisodeResult], path: Path) -> None:
    """Write per-episode summaries to CSV."""
    if not results:
        return
    fieldnames = [
        "episode", "mode", "seed", "success", "total_steps",
        "total_depth", "total_reward", "budget_exhausted",
        "max_total_depth", "wall_time_seconds", "parse_failure_count",
        "depth_distribution",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["depth_distribution"] = json.dumps(row["depth_distribution"])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Complexity histogram validation (Phase 1.5 gate)
# ---------------------------------------------------------------------------

def validate_complexity_distribution(
    n_episodes: int = 10,
    grid_size: int = 10,
    obstacle_fraction: float = 0.20,
    max_steps: int = 100,
    difficulty: Optional[str] = None,
    output_path: Optional[str] = None,
    seed_start: int = 0,
) -> dict:
    """
    Run baseline episodes and validate the complexity score distribution.

    Returns a dict with pass/fail status and statistics.
    Optionally saves a histogram plot.
    """
    all_scores = []

    for i in range(n_episodes):
        seed = seed_start + i
        env = GridWorld(
            grid_size=grid_size,
            obstacle_fraction=obstacle_fraction,
            max_steps=max_steps,
            seed=seed,
            difficulty=difficulty,
        )
        rng = random.Random(seed)

        for _ in range(max_steps):
            all_scores.append(env.get_complexity_score())
            valid = env.get_valid_actions()
            if not valid:
                break
            action = rng.choice(valid)
            result = env.step(action)
            if result.done:
                break

    scores_arr = np.array(all_scores)
    std = float(np.std(scores_arr))
    mean = float(np.mean(scores_arr))

    # Check tier coverage
    tier_shallow = float(np.mean(scores_arr < 0.35))
    tier_medium = float(np.mean((scores_arr >= 0.35) & (scores_arr <= 0.70)))
    tier_deep = float(np.mean(scores_arr > 0.70))
    tiers_covered = sum(1 for t in [tier_shallow, tier_medium, tier_deep] if t > 0.01)

    # Check no bin dominates (approximate: check tier fractions)
    max_tier_frac = max(tier_shallow, tier_medium, tier_deep)

    passed = (tiers_covered >= 2) and (std > 0.08) and (max_tier_frac < 0.80)

    report = {
        "passed": passed,
        "n_scores": len(all_scores),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(float(scores_arr.min()), 4),
        "max": round(float(scores_arr.max()), 4),
        "tier_shallow_frac": round(tier_shallow, 4),
        "tier_medium_frac": round(tier_medium, 4),
        "tier_deep_frac": round(tier_deep, 4),
        "tiers_covered": tiers_covered,
        "max_tier_fraction": round(max_tier_frac, 4),
    }

    # Generate histogram if output path provided
    if output_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(all_scores, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
            ax.axvline(0.35, color="orange", linestyle="--", label="Shallow/Medium (0.35)")
            ax.axvline(0.70, color="red", linestyle="--", label="Medium/Deep (0.70)")
            ax.set_xlabel("Complexity Score", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(
                f"Complexity Score Distribution ({n_episodes} episodes)\n"
                f"μ={mean:.3f}, σ={std:.3f}, tiers={tiers_covered}/3, "
                f"{'PASS' if passed else 'FAIL'}",
                fontsize=11,
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            report["histogram_path"] = output_path
            logger.info("Complexity histogram saved to %s", output_path)
        except ImportError:
            logger.warning("matplotlib not available — skipping histogram plot.")

    return report


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_full_experiment(
    modes: Optional[list[str]] = None,
    n_episodes: int = 30,
    grid_size: int = 10,
    obstacle_fraction: float = 0.20,
    max_steps: Optional[int] = None,
    max_total_depth: int = 150,
    difficulty: Optional[str] = None,
    acc_budget: Optional[int] = None,
    output_dir: str = "results",
    llm_client: Optional[object] = None,
    seed_start: int = 0,
) -> dict:
    """
    Run the full 4-mode experiment and save results.

    Parameters
    ----------
    modes : list[str] or None
        Modes to run. Default: all four.
    n_episodes : int
        Episodes per mode.
    max_steps : int or None
        Hard step limit per episode.  None = use difficulty preset.
        Pass a small number (e.g. 5) for cheap smoke tests.
    output_dir : str
        Directory for CSV/JSON outputs.
    llm_client : object or None
        LLM client with .completion(). None runs baseline only.
    seed_start : int
        Start of the seed sequence.

    Returns
    -------
    dict
        Summary statistics per mode.
    """
    if modes is None:
        modes = ["baseline", "fixed_shallow", "fixed_deep", "adaptive"]

    # Generate identical seed sequence for all modes
    seeds = list(range(seed_start, seed_start + n_episodes))

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "modes": modes,
        "n_episodes": n_episodes,
        "seeds": seeds,
        "grid_size": grid_size,
        "obstacle_fraction": obstacle_fraction,
        "max_steps": max_steps,
        "max_total_depth": max_total_depth,
        "difficulty": difficulty,
        "acc_budget": acc_budget,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = {}

    for mode in modes:
        logger.info("=" * 60)
        logger.info("Running mode: %s (%d episodes)", mode, n_episodes)
        logger.info("=" * 60)

        # Skip LLM modes if no client
        effective_client = llm_client
        if mode == "baseline":
            effective_client = None
        elif llm_client is None and mode != "baseline":
            logger.warning(
                "No LLM client provided — mode '%s' will use random actions.", mode
            )
            effective_client = None

        episodes, steps = run_experiment_mode(
            mode=mode,
            seeds=seeds,
            llm_client=effective_client,
            grid_size=grid_size,
            obstacle_fraction=obstacle_fraction,
            max_steps=max_steps,
            max_total_depth=max_total_depth,
            difficulty=difficulty,
            acc_budget=acc_budget,
        )

        # Write CSVs per mode
        _write_steps_csv(steps, out / f"steps_{mode}.csv")
        _write_episodes_csv(episodes, out / f"episodes_{mode}.csv")

        # Compute summary
        n = len(episodes)
        successes = sum(1 for e in episodes if e.success)
        summary[mode] = {
            "success_rate": round(successes / n, 4) if n > 0 else 0.0,
            "avg_steps": round(sum(e.total_steps for e in episodes) / n, 2) if n > 0 else 0,
            "avg_total_depth": round(sum(e.total_depth for e in episodes) / n, 2) if n > 0 else 0,
            "avg_reward": round(sum(e.total_reward for e in episodes) / n, 4) if n > 0 else 0,
            "budget_exhausted_count": sum(1 for e in episodes if e.budget_exhausted),
            "avg_parse_failures": round(
                sum(e.parse_failure_count for e in episodes) / n, 2
            ) if n > 0 else 0,
        }

        logger.info("Mode %s summary: %s", mode, json.dumps(summary[mode], indent=2))

    # Write combined summary
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
