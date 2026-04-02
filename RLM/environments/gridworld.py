"""
GridWorld Environment for Adaptive Compute Research.

A deterministic, seed-controlled grid navigation benchmark designed
for evaluating LLM-based agents with varying reasoning depths.

The environment produces a per-step complexity score that varies
meaningfully across states, enabling the Adaptive Compute Controller
to demonstrate non-trivial depth allocation.

Design principles:
    - Deterministic transitions (no stochasticity in dynamics)
    - Full reproducibility via numpy Generator seeding
    - Configurable difficulty (grid size, obstacle density)
    - Natural-language state representation optimized for LLM consumption
    - Complexity score = f(distance, obstacle_density, local_branching)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONS: list[str] = ["up", "down", "left", "right"]

_DELTAS: dict[str, tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}


class CellType(enum.IntEnum):
    """Integer codes stored in the grid array."""
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    GOAL = 3


class Difficulty(enum.Enum):
    """Pre-calibrated difficulty presets."""
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


_DIFFICULTY_PRESETS: dict[Difficulty, dict] = {
    Difficulty.EASY:   {"grid_size": 6,  "obstacle_fraction": 0.10, "max_steps": 60},
    Difficulty.MEDIUM: {"grid_size": 10, "obstacle_fraction": 0.20, "max_steps": 100},
    Difficulty.HARD:   {"grid_size": 15, "obstacle_fraction": 0.30, "max_steps": 150},
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Returned by GridWorld.step()."""
    observation: str          # natural-language state text
    reward: float             # +1.0 goal, -0.01 per step, -0.1 wall bump
    done: bool                # episode terminated?
    success: bool             # reached goal?
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------

class GridWorld:
    """
    A 2-D grid navigation environment.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid.
    obstacle_fraction : float
        Fraction of non-start/goal cells to fill with obstacles.
    max_steps : int
        Hard step limit per episode.
    seed : int or None
        Seed for the numpy random Generator (full reproducibility).
    difficulty : Difficulty or str or None
        If provided, overrides grid_size / obstacle_fraction / max_steps
        with a calibrated preset.
    complexity_weights : tuple[float, float, float]
        (w_distance, w_density, w_branching) for the complexity score.
        Normalized internally.  Default equal weighting.
    """

    def __init__(
        self,
        grid_size: int = 10,
        obstacle_fraction: float = 0.20,
        max_steps: int = 100,
        seed: Optional[int] = None,
        difficulty: Optional[Difficulty | str] = None,
        complexity_weights: tuple[float, float, float] = (0.33, 0.33, 0.34),
    ) -> None:
        # Apply difficulty preset if given
        if difficulty is not None:
            if isinstance(difficulty, str):
                difficulty = Difficulty(difficulty.lower())
            preset = _DIFFICULTY_PRESETS[difficulty]
            grid_size = preset["grid_size"]
            obstacle_fraction = preset["obstacle_fraction"]
            max_steps = preset["max_steps"]

        if grid_size < 3:
            raise ValueError("grid_size must be >= 3")
        if not (0.0 <= obstacle_fraction < 1.0):
            raise ValueError("obstacle_fraction must be in [0, 1)")

        self.grid_size = grid_size
        self.obstacle_fraction = obstacle_fraction
        self.max_steps = max_steps
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Normalize complexity weights
        w_total = sum(complexity_weights)
        self._w_dist = complexity_weights[0] / w_total
        self._w_dens = complexity_weights[1] / w_total
        self._w_branch = complexity_weights[2] / w_total

        # Episode state (initialized on reset)
        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.agent_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = (grid_size - 1, grid_size - 1)
        self.obstacles: set[tuple[int, int]] = set()
        self.current_step: int = 0
        self._done: bool = False
        self._success: bool = False
        self._total_obstacles: int = 0

        # Auto-reset on construction
        self.reset()

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> str:
        """
        Reset the environment to a fresh episode.

        Parameters
        ----------
        seed : int or None
            If provided, re-seeds the RNG for this episode.

        Returns
        -------
        str
            The initial natural-language state observation.
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self._done = False
        self._success = False

        self._build_grid()
        return self.get_state_text()

    def _build_grid(self) -> None:
        """Procedurally generate the grid with obstacles."""
        n = self.grid_size
        self.grid = np.zeros((n, n), dtype=np.int8)
        self.obstacles = set()

        # Fix start and goal positions
        self.agent_pos = (0, 0)
        self.goal_pos = (n - 1, n - 1)

        # Compute number of obstacle cells
        total_cells = n * n
        reserved = {self.agent_pos, self.goal_pos}
        available_cells = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if (r, c) not in reserved
        ]
        num_obstacles = int(self.obstacle_fraction * len(available_cells))

        # Place obstacles randomly
        if num_obstacles > 0 and len(available_cells) > 0:
            chosen_indices = self._rng.choice(
                len(available_cells), size=min(num_obstacles, len(available_cells)),
                replace=False,
            )
            for idx in chosen_indices:
                pos = available_cells[idx]
                self.grid[pos[0], pos[1]] = CellType.OBSTACLE
                self.obstacles.add(pos)

        self._total_obstacles = len(self.obstacles)

        # Mark agent and goal
        self.grid[self.agent_pos[0], self.agent_pos[1]] = CellType.AGENT
        self.grid[self.goal_pos[0], self.goal_pos[1]] = CellType.GOAL

    # ------------------------------------------------------------------ #
    # Core step                                                            #
    # ------------------------------------------------------------------ #

    def step(self, action: str) -> StepResult:
        """
        Execute one action in the environment.

        Parameters
        ----------
        action : str
            One of "up", "down", "left", "right".

        Returns
        -------
        StepResult
            Contains observation, reward, done flag, success flag, info dict.
        """
        if self._done:
            return StepResult(
                observation=self.get_state_text(),
                reward=0.0,
                done=True,
                success=self._success,
                info={"reason": "episode_already_done"},
            )

        action = action.strip().lower()
        valid_actions = self.get_valid_actions()

        # Compute reward and new position
        if action not in ACTIONS:
            # Invalid action string — treat as no-op with penalty
            reward = -0.1
            reason = "invalid_action"
        elif action not in valid_actions:
            # Valid action string but blocked (wall/obstacle)
            reward = -0.1
            reason = "blocked"
        else:
            # Execute movement
            dr, dc = _DELTAS[action]
            old_r, old_c = self.agent_pos
            new_r, new_c = old_r + dr, old_c + dc

            # Clear old position
            self.grid[old_r, old_c] = CellType.EMPTY
            self.agent_pos = (new_r, new_c)
            self.grid[new_r, new_c] = CellType.AGENT

            reason = "moved"
            reward = -0.01  # small step cost

        self.current_step += 1

        # Check goal
        if self.agent_pos == self.goal_pos:
            self._done = True
            self._success = True
            reward = 1.0
            reason = "goal_reached"

        # Check step limit
        if self.current_step >= self.max_steps and not self._done:
            self._done = True
            self._success = False
            reason = "max_steps_exceeded"

        return StepResult(
            observation=self.get_state_text(),
            reward=reward,
            done=self._done,
            success=self._success,
            info={
                "reason": reason,
                "step": self.current_step,
                "agent_pos": self.agent_pos,
                "manhattan_distance": self.manhattan_distance(),
            },
        )

    # ------------------------------------------------------------------ #
    # State and observation                                                #
    # ------------------------------------------------------------------ #

    def get_state_text(self) -> str:
        """
        Return a natural-language description of the current state,
        designed for LLM consumption.
        """
        r, c = self.agent_pos
        gr, gc = self.goal_pos
        dist = self.manhattan_distance()
        valid = self.get_valid_actions()

        lines = [
            f"Grid: {self.grid_size}x{self.grid_size} with {self._total_obstacles} obstacles.",
            f"Position: row={r}, col={c}.",
            f"Goal: row={gr}, col={gc}.",
            f"Manhattan distance to goal: {dist}.",
            f"Valid actions: {valid}.",
        ]
        return " ".join(lines)

    def get_valid_actions(self) -> list[str]:
        """Return the list of actions that lead to a valid (non-blocked) cell."""
        r, c = self.agent_pos
        n = self.grid_size
        valid = []
        for action, (dr, dc) in _DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in self.obstacles:
                valid.append(action)
        return valid

    def manhattan_distance(self) -> int:
        """Manhattan distance from agent to goal."""
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
            self.agent_pos[1] - self.goal_pos[1]
        )

    # ------------------------------------------------------------------ #
    # Complexity score                                                     #
    # ------------------------------------------------------------------ #

    def get_complexity_score(self) -> float:
        """
        Compute a complexity score in [0, 1] for the current state.

        Score = w_dist * normalized_distance
              + w_dens * obstacle_density
              + w_branch * (1 - local_branching_factor)

        Components:
            normalized_distance:
                manhattan_distance / max_possible_distance.
                Farther from goal → more complex.

            obstacle_density:
                total_obstacles / (grid_size^2 - 2).
                More obstacles → more complex.

            local_branching_factor:
                |valid_actions| / 4.
                Fewer valid moves → more constrained → more complex.
                We use (1 - branching) so that lower branching = higher score.
        """
        max_dist = 2 * (self.grid_size - 1)
        norm_distance = self.manhattan_distance() / max_dist if max_dist > 0 else 0.0

        total_placeable = self.grid_size ** 2 - 2
        obstacle_density = self._total_obstacles / total_placeable if total_placeable > 0 else 0.0

        branching = len(self.get_valid_actions()) / 4.0
        inverse_branching = 1.0 - branching

        score = (
            self._w_dist * norm_distance
            + self._w_dens * obstacle_density
            + self._w_branch * inverse_branching
        )
        return round(min(max(score, 0.0), 1.0), 6)

    # ------------------------------------------------------------------ #
    # Structured state (for future RL extension)                           #
    # ------------------------------------------------------------------ #

    def get_structured_state(self) -> dict:
        """
        Return a structured state dictionary suitable for RL observation
        spaces. Contains numeric features, not text.
        """
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "manhattan_distance": self.manhattan_distance(),
            "valid_actions": self.get_valid_actions(),
            "complexity_score": self.get_complexity_score(),
            "obstacle_density": self._total_obstacles / max(self.grid_size ** 2 - 2, 1),
            "step": self.current_step,
            "max_steps": self.max_steps,
            "done": self._done,
        }

    # ------------------------------------------------------------------ #
    # Rendering (for debugging)                                            #
    # ------------------------------------------------------------------ #

    def render_ascii(self) -> str:
        """Return a human-readable ASCII grid for debugging."""
        symbols = {
            CellType.EMPTY: ".",
            CellType.OBSTACLE: "#",
            CellType.AGENT: "A",
            CellType.GOAL: "G",
        }
        rows = []
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row_chars.append("A")
                elif (r, c) == self.goal_pos:
                    row_chars.append("G")
                elif (r, c) in self.obstacles:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            rows.append(" ".join(row_chars))
        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"GridWorld(size={self.grid_size}, obstacles={self._total_obstacles}, "
            f"agent={self.agent_pos}, goal={self.goal_pos}, "
            f"step={self.current_step}/{self.max_steps})"
        )
