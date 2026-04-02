"""
tests/test_gridworld.py
=======================
Comprehensive tests for the GridWorld environment.

Validates:
    - Deterministic reset/step with fixed seed
    - Obstacle placement (never on start/goal)
    - Valid action masking correctness
    - Complexity score bounds and non-triviality
    - State text required information
    - Difficulty presets
    - Episode termination conditions (goal + max_steps)
    - Step result structure

Run with:
    pytest tests/test_gridworld.py -v
"""

import pytest
import numpy as np
from RLM.environments.gridworld import GridWorld, Difficulty, ACTIONS, StepResult


# ===========================================================================
# Determinism & Reproducibility
# ===========================================================================

class TestDeterminism:

    def test_same_seed_same_grid(self):
        env1 = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=42)
        env2 = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=42)
        assert env1.obstacles == env2.obstacles
        assert env1.agent_pos == env2.agent_pos
        assert env1.goal_pos == env2.goal_pos

    def test_same_seed_same_trajectory(self):
        actions = ["right", "right", "down", "down"]
        env1 = GridWorld(grid_size=6, obstacle_fraction=0.05, seed=99)
        env2 = GridWorld(grid_size=6, obstacle_fraction=0.05, seed=99)
        for a in actions:
            r1 = env1.step(a)
            r2 = env2.step(a)
            assert r1.reward == r2.reward
            assert env1.agent_pos == env2.agent_pos

    def test_different_seed_different_grid(self):
        env1 = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=1)
        env2 = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=2)
        # Extremely unlikely to have identical obstacle sets
        assert env1.obstacles != env2.obstacles

    def test_reset_with_new_seed(self):
        env = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=42)
        obs1 = env.obstacles.copy()
        env.reset(seed=99)
        obs2 = env.obstacles.copy()
        assert obs1 != obs2

    def test_reset_same_seed_reproduces(self):
        env = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=42)
        obs_first = env.obstacles.copy()
        env.reset(seed=42)
        assert env.obstacles == obs_first


# ===========================================================================
# Grid Construction
# ===========================================================================

class TestGridConstruction:

    def test_no_obstacle_on_start(self):
        for seed in range(20):
            env = GridWorld(grid_size=10, obstacle_fraction=0.3, seed=seed)
            assert env.agent_pos not in env.obstacles

    def test_no_obstacle_on_goal(self):
        for seed in range(20):
            env = GridWorld(grid_size=10, obstacle_fraction=0.3, seed=seed)
            assert env.goal_pos not in env.obstacles

    def test_start_position(self):
        env = GridWorld(grid_size=8, seed=0)
        assert env.agent_pos == (0, 0)

    def test_goal_position(self):
        env = GridWorld(grid_size=8, seed=0)
        assert env.goal_pos == (7, 7)

    def test_obstacle_count_approximate(self):
        env = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=42)
        expected = int(0.2 * (100 - 2))  # 19
        assert abs(len(env.obstacles) - expected) <= 1

    def test_zero_obstacles(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        assert len(env.obstacles) == 0

    def test_minimum_grid_size(self):
        env = GridWorld(grid_size=3, seed=0)
        assert env.grid_size == 3

    def test_invalid_grid_size_raises(self):
        with pytest.raises(ValueError, match="grid_size"):
            GridWorld(grid_size=2)

    def test_invalid_obstacle_fraction_raises(self):
        with pytest.raises(ValueError, match="obstacle_fraction"):
            GridWorld(obstacle_fraction=1.0)


# ===========================================================================
# Valid Action Masking
# ===========================================================================

class TestActionMasking:

    def test_corner_start_has_limited_actions(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        valid = env.get_valid_actions()
        # At (0,0), can go down or right, not up or left
        assert "up" not in valid
        assert "left" not in valid
        assert "down" in valid
        assert "right" in valid

    def test_center_open_grid_all_valid(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        # Move to center (2,2)
        env.step("down"); env.step("down")
        env.step("right"); env.step("right")
        valid = env.get_valid_actions()
        assert len(valid) == 4

    def test_obstacle_blocks_action(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        # Manually place obstacle at (0,1)
        env.obstacles.add((0, 1))
        env.grid[0, 1] = 1  # OBSTACLE
        valid = env.get_valid_actions()
        assert "right" not in valid

    def test_valid_actions_subset_of_all_actions(self):
        env = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=42)
        valid = env.get_valid_actions()
        for a in valid:
            assert a in ACTIONS


# ===========================================================================
# Step Execution
# ===========================================================================

class TestStep:

    def test_step_returns_step_result(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        result = env.step("right")
        assert isinstance(result, StepResult)

    def test_valid_move_updates_position(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        assert env.agent_pos == (0, 0)
        env.step("right")
        assert env.agent_pos == (0, 1)
        env.step("down")
        assert env.agent_pos == (1, 1)

    def test_step_cost_is_negative(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        result = env.step("right")
        assert result.reward == pytest.approx(-0.01)

    def test_blocked_move_penalty(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        result = env.step("up")  # at (0,0), up is blocked by wall
        assert result.reward == pytest.approx(-0.1)
        assert env.agent_pos == (0, 0)  # didn't move

    def test_invalid_action_string(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        result = env.step("teleport")
        assert result.reward == pytest.approx(-0.1)

    def test_goal_reached_reward(self):
        env = GridWorld(grid_size=3, obstacle_fraction=0.0, seed=0)
        # Navigate to goal at (2,2) from (0,0)
        env.step("right"); env.step("right")
        env.step("down"); env.step("down")
        assert env._success is True
        assert env._done is True

    def test_goal_reward_is_positive(self):
        env = GridWorld(grid_size=3, obstacle_fraction=0.0, seed=0)
        env.step("right"); env.step("right")
        env.step("down")
        result = env.step("down")  # reach (2,2)
        assert result.reward == pytest.approx(1.0)
        assert result.success is True
        assert result.done is True

    def test_max_steps_terminates(self):
        env = GridWorld(grid_size=10, obstacle_fraction=0.0, max_steps=5, seed=0)
        for _ in range(5):
            result = env.step("right")
        assert result.done is True
        assert result.success is False

    def test_step_after_done_is_noop(self):
        env = GridWorld(grid_size=3, obstacle_fraction=0.0, max_steps=2, seed=0)
        env.step("right"); env.step("right")  # uses up 2 steps
        result = env.step("down")  # should be noop
        assert result.done is True
        assert result.reward == 0.0


# ===========================================================================
# Complexity Score
# ===========================================================================

class TestComplexityScore:

    def test_score_in_unit_interval(self):
        for seed in range(20):
            env = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=seed)
            score = env.get_complexity_score()
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for seed {seed}"

    def test_score_at_goal_is_lower(self):
        """Agent at goal should have low distance component → lower score."""
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        score_start = env.get_complexity_score()
        # Move to goal
        env.step("right"); env.step("right"); env.step("right"); env.step("right")
        env.step("down"); env.step("down"); env.step("down"); env.step("down")
        score_near_goal = env.get_complexity_score()
        assert score_near_goal < score_start

    def test_score_varies_across_steps(self):
        """Score should change as agent moves (distance changes)."""
        env = GridWorld(grid_size=10, obstacle_fraction=0.1, seed=42)
        scores = []
        for _ in range(10):
            scores.append(env.get_complexity_score())
            valid = env.get_valid_actions()
            if valid:
                env.step(valid[0])
        unique_scores = set(scores)
        assert len(unique_scores) > 1, "Complexity score should vary across steps"

    def test_higher_obstacle_density_increases_score(self):
        env_low = GridWorld(grid_size=10, obstacle_fraction=0.05, seed=42)
        env_high = GridWorld(grid_size=10, obstacle_fraction=0.35, seed=42)
        # At start position, distance is same; density differs
        # Note: obstacles may also affect branching, but density component is higher
        score_low = env_low.get_complexity_score()
        score_high = env_high.get_complexity_score()
        assert score_high > score_low

    def test_score_not_constant_across_seeds(self):
        """Different seeds should produce different start scores."""
        scores = set()
        for seed in range(10):
            env = GridWorld(grid_size=10, obstacle_fraction=0.2, seed=seed)
            scores.add(env.get_complexity_score())
        assert len(scores) > 1


# ===========================================================================
# State Text
# ===========================================================================

class TestStateText:

    def test_state_text_contains_position(self):
        env = GridWorld(grid_size=5, seed=0)
        text = env.get_state_text()
        assert "row=0" in text
        assert "col=0" in text

    def test_state_text_contains_goal(self):
        env = GridWorld(grid_size=5, seed=0)
        text = env.get_state_text()
        assert "Goal" in text

    def test_state_text_contains_distance(self):
        env = GridWorld(grid_size=5, seed=0)
        text = env.get_state_text()
        assert "distance" in text.lower()

    def test_state_text_contains_valid_actions(self):
        env = GridWorld(grid_size=5, seed=0)
        text = env.get_state_text()
        assert "Valid actions" in text

    def test_state_text_contains_obstacles(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.2, seed=0)
        text = env.get_state_text()
        assert "obstacle" in text.lower()


# ===========================================================================
# Difficulty Presets
# ===========================================================================

class TestDifficultyPresets:

    def test_easy_preset(self):
        env = GridWorld(difficulty="easy", seed=0)
        assert env.grid_size == 6
        assert env.max_steps == 60

    def test_medium_preset(self):
        env = GridWorld(difficulty="medium", seed=0)
        assert env.grid_size == 10
        assert env.max_steps == 100

    def test_hard_preset(self):
        env = GridWorld(difficulty="hard", seed=0)
        assert env.grid_size == 15
        assert env.max_steps == 150

    def test_enum_difficulty(self):
        env = GridWorld(difficulty=Difficulty.HARD, seed=0)
        assert env.grid_size == 15

    def test_harder_has_more_obstacles(self):
        easy = GridWorld(difficulty="easy", seed=42)
        hard = GridWorld(difficulty="hard", seed=42)
        assert len(hard.obstacles) > len(easy.obstacles)


# ===========================================================================
# Structured State (RL hook)
# ===========================================================================

class TestStructuredState:

    def test_structured_state_keys(self):
        env = GridWorld(grid_size=5, seed=0)
        state = env.get_structured_state()
        expected_keys = {
            "agent_pos", "goal_pos", "manhattan_distance",
            "valid_actions", "complexity_score", "obstacle_density",
            "step", "max_steps", "done",
        }
        assert expected_keys.issubset(state.keys())

    def test_structured_state_types(self):
        env = GridWorld(grid_size=5, seed=0)
        state = env.get_structured_state()
        assert isinstance(state["agent_pos"], tuple)
        assert isinstance(state["complexity_score"], float)
        assert isinstance(state["valid_actions"], list)


# ===========================================================================
# Manhattan Distance
# ===========================================================================

class TestManhattanDistance:

    def test_start_distance(self):
        env = GridWorld(grid_size=5, seed=0)
        # From (0,0) to (4,4) → distance = 8
        assert env.manhattan_distance() == 8

    def test_distance_decreases_toward_goal(self):
        env = GridWorld(grid_size=5, obstacle_fraction=0.0, seed=0)
        d0 = env.manhattan_distance()
        env.step("right")
        d1 = env.manhattan_distance()
        assert d1 < d0

    def test_at_goal_distance_zero(self):
        env = GridWorld(grid_size=3, obstacle_fraction=0.0, seed=0)
        env.step("right"); env.step("right")
        env.step("down"); env.step("down")
        assert env.manhattan_distance() == 0
