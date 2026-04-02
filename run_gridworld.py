"""
run_gridworld.py
================
Top-level entry point for GridWorld adaptive compute experiments.

Usage:
    python run_gridworld.py                                # baseline only (free)
    python run_gridworld.py --smoke                       # 2 episodes, 5 steps (~$0.001)
    python run_gridworld.py --mode fixed_shallow --episodes 3 --max-steps 10
    python run_gridworld.py --all --episodes 30           # full experiment
    python run_gridworld.py --validate                    # complexity histogram only
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # reads .env from project root

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_gridworld")


def main():
    parser = argparse.ArgumentParser(description="GridWorld Adaptive Compute Experiments")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["baseline", "fixed_shallow", "fixed_deep", "adaptive"],
        help="Run a single mode. Default: baseline only.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 4 modes.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run complexity histogram validation only (Phase 1.5 gate).",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per mode.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (overrides difficulty preset). "
                             "Use 5-10 for cheap smoke tests.")
    parser.add_argument("--difficulty", type=str, default="medium", help="easy/medium/hard.")
    parser.add_argument("--max-depth-budget", type=int, default=150, help="Per-episode depth budget.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model string.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed.")
    parser.add_argument("--smoke", action="store_true",
                        help="Ultra-cheap smoke test: 2 episodes, 5 steps. ~$0.001.")

    args = parser.parse_args()

    # --smoke sets safe testing defaults so you don't burn credits
    if args.smoke:
        args.episodes = 2
        args.max_steps = 5
        args.max_depth_budget = 15
        logger.info("🔬 Smoke-test mode: episodes=2, max_steps=5, budget=15 (~$0.001)")

    from RLM.experiments.gridworld_runner import (
        run_full_experiment,
        validate_complexity_distribution,
    )

    # -----------------------------------------------------------------
    # Phase 1.5: Complexity histogram validation
    # -----------------------------------------------------------------
    if args.validate:
        logger.info("=" * 60)
        logger.info("Running complexity histogram validation (Phase 1.5)")
        logger.info("=" * 60)

        import os
        os.makedirs(args.output_dir, exist_ok=True)

        report = validate_complexity_distribution(
            n_episodes=10,
            difficulty=args.difficulty,
            output_path=f"{args.output_dir}/complexity_histogram.png",
            seed_start=args.seed_start,
        )
        print(json.dumps(report, indent=2))

        if report["passed"]:
            logger.info("✅ PASSED — complexity distribution is suitable for experiments.")
        else:
            logger.warning("❌ FAILED — complexity distribution needs tuning. See histogram.")
            sys.exit(1)
        return

    # -----------------------------------------------------------------
    # Determine modes to run
    # -----------------------------------------------------------------
    if args.all:
        modes = ["baseline", "fixed_shallow", "fixed_deep", "adaptive"]
    elif args.mode:
        modes = [args.mode]
    else:
        modes = ["baseline"]

    # -----------------------------------------------------------------
    # Initialize LLM client (only if needed)
    # -----------------------------------------------------------------
    llm_client = None
    needs_llm = any(m != "baseline" for m in modes)

    if needs_llm:
        try:
            from RLM.utils.llm import LLMClient
            llm_client = LLMClient(model=args.model)
            logger.info("LLM client initialized: model=%s", args.model)
        except Exception as e:
            logger.warning(
                "Could not initialize LLM client (%s). "
                "Non-baseline modes will use random actions.", e
            )

    # -----------------------------------------------------------------
    # Run experiment
    # -----------------------------------------------------------------
    logger.info("Running modes: %s | %d episodes each", modes, args.episodes)

    summary = run_full_experiment(
        modes=modes,
        n_episodes=args.episodes,
        max_steps=args.max_steps,          # None → use difficulty preset default
        max_total_depth=args.max_depth_budget,
        difficulty=args.difficulty,
        output_dir=args.output_dir,
        llm_client=llm_client,
        seed_start=args.seed_start,
    )

    print("\n" + "=" * 60)
    print("  Experiment Complete")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
