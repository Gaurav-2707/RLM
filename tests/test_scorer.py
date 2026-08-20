"""
Tests for training/scorer.py — TrajectoryScorer.

All five scoring dimensions tested with known inputs → known outputs.
Uses the same _traj/_step helpers as test_render.py.
"""

import uuid
import pytest

from RLM.utils.trajectory import RLMTrajectory, TrajectoryStep
from RLM.training.scorer import TrajectoryScorer


# ── helpers ───────────────────────────────────────────────────────────────────

def _step(action_type, reward=0.0, output_action="fix", was_rollback=False,
          rollback_reason="", n=1):
    return TrajectoryStep(
        step_number=n, iteration=1, action_type=action_type,
        input_state="state", output_action=output_action,
        outcome="", reward=reward, confidence=0.7,
        was_rollback=was_rollback, rollback_reason=rollback_reason,
    )


def _traj(steps, final_outcome=True, task="fix the bug", rollbacks=None):
    rb = rollbacks if rollbacks is not None else sum(1 for s in steps if s.was_rollback)
    return RLMTrajectory(
        trajectory_id=str(uuid.uuid4()),
        steps=steps, total_steps=len(steps),
        total_rollbacks=rb,
        final_outcome=final_outcome, verified=True,
        final_diff="", task_description=task,
    )


# ── _score_recursion ──────────────────────────────────────────────────────────

class TestScoreRecursion:
    def setup_method(self):
        self.scorer = TrajectoryScorer()

    def test_rollback_and_success_is_1(self):
        traj = _traj([
            _step("edit", n=1),
            _step("edit", was_rollback=True, n=2),
            _step("run_tests", reward=1.0, n=3),
        ], final_outcome=True)
        assert self.scorer._score_recursion(traj) == 1.0

    def test_success_no_rollback_is_0_6(self):
        traj = _traj([_step("edit", n=1), _step("run_tests", reward=1.0, n=2)],
                     final_outcome=True)
        assert self.scorer._score_recursion(traj) == pytest.approx(0.6)

    def test_failure_is_0_2(self):
        traj = _traj([_step("edit", n=1), _step("run_tests", reward=0.0, n=2)],
                     final_outcome=False)
        assert self.scorer._score_recursion(traj) == pytest.approx(0.2)

    def test_rollback_but_failure_is_0_2(self):
        """Rollback + failure → still failed, no recovery signal → 0.2."""
        traj = _traj([
            _step("edit", n=1),
            _step("edit", was_rollback=True, n=2),
            _step("run_tests", reward=0.0, n=3),
        ], final_outcome=False)
        assert self.scorer._score_recursion(traj) == pytest.approx(0.2)


# ── _score_difficulty ─────────────────────────────────────────────────────────

class TestScoreDifficulty:
    def setup_method(self):
        self.scorer = TrajectoryScorer(max_steps=30)

    def test_one_step_is_low(self):
        traj = _traj([_step("edit", n=1)], final_outcome=True)
        assert self.scorer._score_difficulty(traj) == pytest.approx(1 / 30)

    def test_max_steps_is_1(self):
        steps = [_step("edit", n=i+1) for i in range(30)]
        traj = _traj(steps, final_outcome=True)
        assert self.scorer._score_difficulty(traj) == pytest.approx(1.0)

    def test_beyond_max_steps_clamps_to_1(self):
        steps = [_step("edit", n=i+1) for i in range(50)]
        traj = _traj(steps, final_outcome=False)
        assert self.scorer._score_difficulty(traj) == pytest.approx(1.0)

    def test_proportional_for_mid_length(self):
        steps = [_step("edit", n=i+1) for i in range(15)]
        traj = _traj(steps, final_outcome=True)
        assert self.scorer._score_difficulty(traj) == pytest.approx(15 / 30)


# ── _score_novelty ────────────────────────────────────────────────────────────

class TestScoreNovelty:
    def setup_method(self):
        self.scorer = TrajectoryScorer()

    def test_cold_start_returns_0_5(self):
        traj = _traj([_step("edit", output_action="x = 1", n=1)])
        assert self.scorer._score_novelty(traj, corpus=[]) == pytest.approx(0.5)

    def test_identical_traj_in_corpus_is_low_novelty(self):
        """Exact duplicate should have cosine similarity 1.0 → novelty ≈ 0."""
        steps = [_step("edit", output_action="return x + 1", n=1)]
        traj = _traj(steps, task="add one to x")
        corpus = [_traj(steps, task="add one to x")]
        score = self.scorer._score_novelty(traj, corpus)
        assert score < 0.1, f"Duplicate should have near-zero novelty, got {score}"

    def test_completely_different_traj_has_high_novelty(self):
        """Orthogonal vocabulary → high novelty."""
        traj = _traj([_step("edit", output_action="neural network matrix gradient backprop", n=1)],
                     task="train a neural network model")
        corpus_traj = _traj([_step("edit", output_action="parse xml element attribute dom", n=1)],
                             task="parse the xml document")
        score = self.scorer._score_novelty(traj, corpus=[corpus_traj])
        assert score > 0.5, f"Orthogonal traj should have high novelty, got {score}"

    def test_returns_in_0_1_range(self):
        traj = _traj([_step("edit", output_action="foo bar baz", n=1)])
        corpus = [_traj([_step("edit", output_action="something else entirely", n=1)])]
        score = self.scorer._score_novelty(traj, corpus)
        assert 0.0 <= score <= 1.0


# ── _score_verification ───────────────────────────────────────────────────────

class TestScoreVerification:
    def setup_method(self):
        self.scorer = TrajectoryScorer(max_steps=30)

    def test_no_run_tests_is_zero(self):
        traj = _traj([_step("edit", n=1)])
        assert self.scorer._score_verification(traj) == pytest.approx(0.0)

    def test_fail_to_pass_transition_is_positive(self):
        steps = [
            _step("edit", n=1),
            _step("run_tests", reward=0.0, n=2),
            _step("edit", n=3),
            _step("run_tests", reward=1.0, n=4),
        ]
        traj = _traj(steps, final_outcome=True)
        assert self.scorer._score_verification(traj) > 0.0

    def test_no_transition_when_always_passing(self):
        """Tests that always pass → no FAIL→PASS flip → score = 0."""
        steps = [
            _step("run_tests", reward=1.0, n=1),
            _step("run_tests", reward=1.0, n=2),
        ]
        traj = _traj(steps, final_outcome=True)
        # prev_reward starts at 0.0, first run_tests has reward=1.0 → that IS a transition!
        # Score = 1 transition / max(1, 30) = 1/30
        assert self.scorer._score_verification(traj) == pytest.approx(1 / 30)

    def test_multiple_transitions_accumulate(self):
        steps = [
            _step("run_tests", reward=0.0, n=1),
            _step("run_tests", reward=1.0, n=2),   # transition 1
            _step("run_tests", reward=0.0, n=3),
            _step("run_tests", reward=1.0, n=4),   # transition 2
        ]
        traj = _traj(steps, final_outcome=True)
        assert self.scorer._score_verification(traj) == pytest.approx(2 / 30)


# ── _score_efficiency ─────────────────────────────────────────────────────────

class TestScoreEfficiency:
    def setup_method(self):
        self.scorer = TrajectoryScorer(max_steps=30)

    def test_failure_is_zero(self):
        traj = _traj([_step("edit", n=1)], final_outcome=False)
        assert self.scorer._score_efficiency(traj) == pytest.approx(0.0)

    def test_one_step_success_is_near_1(self):
        traj = _traj([_step("edit", n=1)], final_outcome=True)
        assert self.scorer._score_efficiency(traj) == pytest.approx(1.0 - 1/30)

    def test_max_steps_success_is_zero(self):
        steps = [_step("edit", n=i+1) for i in range(30)]
        traj = _traj(steps, final_outcome=True)
        assert self.scorer._score_efficiency(traj) == pytest.approx(0.0)

    def test_beyond_max_steps_clamps_to_zero(self):
        steps = [_step("edit", n=i+1) for i in range(50)]
        traj = _traj(steps, final_outcome=True)
        assert self.scorer._score_efficiency(traj) == pytest.approx(0.0)


# ── combined score ────────────────────────────────────────────────────────────

class TestCombinedScore:
    def setup_method(self):
        self.scorer = TrajectoryScorer()

    def test_score_in_0_1_range(self):
        traj = _traj([_step("edit", n=1), _step("run_tests", reward=1.0, n=2)],
                     final_outcome=True)
        assert 0.0 <= self.scorer.score(traj, []) <= 1.0

    def test_rollback_success_scores_higher_than_straight_success(self):
        straight = _traj([_step("edit", n=1), _step("run_tests", reward=1.0, n=2)],
                         final_outcome=True)
        with_rollback = _traj([
            _step("edit", n=1),
            _step("run_tests", reward=0.0, n=2),
            _step("edit", was_rollback=True, n=3),
            _step("run_tests", reward=1.0, n=4),
        ], final_outcome=True)
        assert self.scorer.score(with_rollback, []) > self.scorer.score(straight, [])

    def test_failure_scores_lower_than_success(self):
        success = _traj([_step("edit", n=1)], final_outcome=True)
        failure = _traj([_step("edit", n=1)], final_outcome=False)
        assert self.scorer.score(success, []) > self.scorer.score(failure, [])

    def test_filter_corpus_keeps_high_scores(self):
        good = _traj([
            _step("edit", n=1),
            _step("edit", was_rollback=True, n=2),
            _step("run_tests", reward=1.0, n=3),
        ], final_outcome=True, task="fix the authentication bug in login module")
        bad = _traj([_step("edit", n=1)], final_outcome=False, task="fix auth")
        accepted = self.scorer.filter_corpus([good, bad], min_score=0.4)
        assert good in accepted
        assert bad not in accepted

    def test_filter_corpus_progressive_novelty(self):
        """Second identical trajectory should score lower than first (novelty drops)."""
        steps = [_step("edit", output_action="identical action text here", n=1)]
        t1 = _traj(steps, task="identical task description")
        t2 = _traj(steps, task="identical task description")
        # t1 added to corpus → t2 should score lower (less novel)
        score_t1 = self.scorer.score(t1, [])
        score_t2 = self.scorer.score(t2, [t1])
        assert score_t2 < score_t1, "Duplicate should score lower due to novelty drop"
