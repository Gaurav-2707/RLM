"""
Analysis module for GridWorld experiments.

Loads structured CSV logs, computes aggregate metrics, and generates
publication-ready matplotlib figures.

Core plots:
    1. Success rate comparison (bar chart, 4 modes)
    2. Compute cost vs. success rate (grouped bar)
    3. Steps-to-goal distribution (box plots)
    4. Depth distribution (adaptive mode, stacked bar)

Usage:
    python -m RLM.experiments.analysis results/
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ModeData:
    """Loaded data for one experimental mode."""
    mode: str
    episodes: list[dict]
    steps: list[dict]


def load_mode_data(results_dir: str, mode: str) -> Optional[ModeData]:
    """Load episodes and steps CSVs for a single mode."""
    rdir = Path(results_dir)
    ep_path = rdir / f"episodes_{mode}.csv"
    st_path = rdir / f"steps_{mode}.csv"

    if not ep_path.exists():
        return None

    episodes = []
    with open(ep_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse types
            row["success"] = row["success"] == "True"
            row["total_steps"] = int(row["total_steps"])
            row["total_depth"] = int(row["total_depth"])
            row["total_reward"] = float(row["total_reward"])
            row["budget_exhausted"] = row["budget_exhausted"] == "True"
            row["wall_time_seconds"] = float(row["wall_time_seconds"])
            row["parse_failure_count"] = int(row["parse_failure_count"])
            if "depth_distribution" in row:
                row["depth_distribution"] = json.loads(row["depth_distribution"])
            episodes.append(row)

    steps = []
    if st_path.exists():
        with open(st_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["step"] = int(row["step"])
                row["complexity"] = float(row["complexity"])
                row["depth"] = int(row["depth"])
                row["reward"] = float(row["reward"])
                row["cumulative_depth"] = int(row["cumulative_depth"])
                row["depth_budget_remaining"] = int(row["depth_budget_remaining"])
                row["success"] = row["success"] == "True"
                row["done"] = row["done"] == "True"
                steps.append(row)

    return ModeData(mode=mode, episodes=episodes, steps=steps)


def load_all_modes(
    results_dir: str,
    modes: Optional[list[str]] = None,
) -> dict[str, ModeData]:
    """Load data for all available modes."""
    if modes is None:
        modes = ["baseline", "fixed_shallow", "fixed_deep", "adaptive"]

    data = {}
    for mode in modes:
        md = load_mode_data(results_dir, mode)
        if md is not None:
            data[mode] = md
    return data


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(data: dict[str, ModeData]) -> dict[str, dict]:
    """
    Compute aggregate metrics per mode.

    Returns dict[mode_name, metrics_dict].
    """
    metrics = {}
    for mode, md in data.items():
        eps = md.episodes
        n = len(eps)
        if n == 0:
            continue

        successes = [e for e in eps if e["success"]]
        n_success = len(successes)

        metrics[mode] = {
            "n_episodes": n,
            "success_rate": n_success / n,
            "avg_steps": np.mean([e["total_steps"] for e in eps]),
            "std_steps": np.std([e["total_steps"] for e in eps]),
            "avg_steps_on_success": (
                np.mean([e["total_steps"] for e in successes])
                if successes else float("nan")
            ),
            "avg_total_depth": np.mean([e["total_depth"] for e in eps]),
            "std_total_depth": np.std([e["total_depth"] for e in eps]),
            "avg_reward": np.mean([e["total_reward"] for e in eps]),
            "avg_parse_failures": np.mean([e["parse_failure_count"] for e in eps]),
            "budget_exhausted_rate": sum(
                1 for e in eps if e["budget_exhausted"]
            ) / n,
        }

        # Depth distribution (aggregate across all episodes)
        if md.steps:
            all_depths = [s["depth"] for s in md.steps]
            unique, counts = np.unique(all_depths, return_counts=True)
            metrics[mode]["depth_distribution"] = dict(zip(
                [int(u) for u in unique],
                [int(c) for c in counts],
            ))
            metrics[mode]["avg_complexity"] = np.mean(
                [s["complexity"] for s in md.steps]
            )

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_style():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# Map mode names to display-friendly labels and colors
_MODE_LABELS = {
    "baseline": "Baseline\n(Random)",
    "fixed_shallow": "Fixed\nShallow (d=1)",
    "fixed_deep": "Fixed\nDeep (d=3)",
    "adaptive": "Adaptive\n(ACC)",
}

_MODE_COLORS = {
    "baseline": "#95a5a6",
    "fixed_shallow": "#3498db",
    "fixed_deep": "#e74c3c",
    "adaptive": "#2ecc71",
}


def plot_success_rate(
    metrics: dict[str, dict],
    output_path: Optional[str] = None,
) -> None:
    """Bar chart comparing success rates across modes."""
    import matplotlib.pyplot as plt
    _setup_style()

    modes = [m for m in _MODE_LABELS if m in metrics]
    rates = [metrics[m]["success_rate"] * 100 for m in modes]
    colors = [_MODE_COLORS[m] for m in modes]
    labels = [_MODE_LABELS[m] for m in modes]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.5)

    # Value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold",
        )

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Compute Mode")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 100)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


def plot_compute_vs_success(
    metrics: dict[str, dict],
    output_path: Optional[str] = None,
) -> None:
    """Grouped bar chart: compute cost vs. success rate."""
    import matplotlib.pyplot as plt
    _setup_style()

    modes = [m for m in _MODE_LABELS if m in metrics]
    labels = [_MODE_LABELS[m] for m in modes]
    success = [metrics[m]["success_rate"] * 100 for m in modes]
    compute = [metrics[m]["avg_total_depth"] for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax1 = plt.subplots()
    bars1 = ax1.bar(x - width/2, success, width, label="Success Rate (%)",
                    color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Success Rate (%)", color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, compute, width, label="Avg Total Depth",
                    color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Avg Total Depth (Compute)", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Compute Cost vs. Performance")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


def plot_steps_distribution(
    data: dict[str, ModeData],
    output_path: Optional[str] = None,
) -> None:
    """Box plot of steps-to-completion across modes."""
    import matplotlib.pyplot as plt
    _setup_style()

    modes = [m for m in _MODE_LABELS if m in data]
    steps_data = []
    labels = []
    colors = []

    for mode in modes:
        steps = [e["total_steps"] for e in data[mode].episodes]
        steps_data.append(steps)
        labels.append(_MODE_LABELS[mode])
        colors.append(_MODE_COLORS[mode])

    fig, ax = plt.subplots()
    bp = ax.boxplot(steps_data, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Total Steps per Episode")
    ax.set_title("Steps Distribution by Compute Mode")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


def plot_depth_distribution(
    data: dict[str, ModeData],
    output_path: Optional[str] = None,
) -> None:
    """Bar chart of depth tier distribution for adaptive mode."""
    import matplotlib.pyplot as plt
    _setup_style()

    if "adaptive" not in data:
        return

    steps = data["adaptive"].steps
    if not steps:
        return

    depths = [s["depth"] for s in steps]
    unique, counts = np.unique(depths, return_counts=True)

    depth_colors = {0: "#95a5a6", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}

    fig, ax = plt.subplots()
    bars = ax.bar(
        [f"d={d}" for d in unique],
        counts,
        color=[depth_colors.get(d, "#333") for d in unique],
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, count in zip(bars, counts):
        pct = count / sum(counts) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom",
        )

    ax.set_ylabel("Number of Steps")
    ax.set_xlabel("Depth Tier")
    ax.set_title("Adaptive Mode — Depth Distribution")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir: str, output_dir: Optional[str] = None) -> None:
    """Load data and generate all analysis plots."""
    import matplotlib
    matplotlib.use("Agg")

    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    data = load_all_modes(results_dir)
    if not data:
        print(f"No data found in {results_dir}")
        return

    metrics = compute_metrics(data)

    # Print metrics summary
    print("\n" + "=" * 60)
    print("  Experiment Metrics Summary")
    print("=" * 60)
    for mode, m in metrics.items():
        print(f"\n  {mode}:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Generate plots
    plot_success_rate(metrics, os.path.join(output_dir, "success_rate.png"))
    print(f"\n  Saved: {output_dir}/success_rate.png")

    plot_compute_vs_success(metrics, os.path.join(output_dir, "compute_vs_success.png"))
    print(f"  Saved: {output_dir}/compute_vs_success.png")

    plot_steps_distribution(data, os.path.join(output_dir, "steps_distribution.png"))
    print(f"  Saved: {output_dir}/steps_distribution.png")

    if "adaptive" in data:
        plot_depth_distribution(data, os.path.join(output_dir, "depth_distribution.png"))
        print(f"  Saved: {output_dir}/depth_distribution.png")

    # Save metrics JSON
    # Convert numpy types for JSON serialization
    serializable_metrics = {}
    for mode, m in metrics.items():
        serializable_metrics[mode] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else
                int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in m.items()
        }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"  Saved: {output_dir}/metrics.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    generate_all_plots(results_dir)
