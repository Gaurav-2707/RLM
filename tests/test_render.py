"""
Regression tests for training/render.py.

These tests guard against the two known bugs that poisoned training data:
  BUG1: verdict tokens emitted on non-run_tests steps (outcome string truthy)
  BUG2: empty <reason></reason> blocks on every non-reason step

And verify the core rendering invariants the loss depends on.
"""

import uuid
import pytest

from RLM.utils.trajectory import RLMTrajectory, TrajectoryStep
from RLM.training.render import (
    render_trajectory, compute_traj_weight, _last_action_index, Span
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _step(action_type, outcome="", reward=0.0, confidence=0.5,
          output_action="some action", was_rollback=False, rollback_reason="",
          n=1):
    return TrajectoryStep(
        step_number=n, iteration=1, action_type=action_type,
        input_state="def foo(): pass", output_action=output_action,
        outcome=outcome, reward=reward, confidence=confidence,
        was_rollback=was_rollback, rollback_reason=rollback_reason,
    )


def _traj(steps, final_outcome=True):
    return RLMTrajectory(
        trajectory_id=str(uuid.uuid4()),
        steps=steps,
        total_steps=len(steps),
        total_rollbacks=sum(1 for s in steps if s.was_rollback),
        final_outcome=final_outcome,
        verified=True,
        final_diff="",
        task_description="test task",
    )


def _spans_by_role(rt, role):
    return [s for s in rt.spans if s.role == role]


# ── BUG1 regression ───────────────────────────────────────────────────────────

class TestBug1VerdictDoubleEmit:
    """BUG1: step.outcome is a string, always truthy → double-emitted verdicts."""

    def test_verdict_only_on_run_tests_step(self):
        """Only run_tests steps should produce a <test> verdict span."""
        steps = [
            _step("edit",     outcome="some non-empty output", n=1),
            _step("reason",   outcome="reasoning text here",   n=2),
            _step("run_tests", outcome="1 passed", reward=1.0, n=3),
        ]
        rt = render_trajectory(_traj(steps))
        verdict_spans = _spans_by_role(rt, "verdict")
        assert len(verdict_spans) == 1, (
            f"Expected exactly 1 verdict span, got {len(verdict_spans)}. "
            f"BUG1 regression: step.outcome truthy on non-test steps."
        )

    def test_verdict_content_reflects_reward(self):
        """PASS verdict when reward > 0, FAIL when reward <= 0."""
        pass_step = _step("run_tests", outcome="1 passed", reward=1.0, n=1)
        fail_step = _step("run_tests", outcome="0 passed, 1 failed", reward=0.0, n=2)
        rt = render_trajectory(_traj([pass_step, fail_step], final_outcome=False))
        verdicts = _spans_by_role(rt, "verdict")
        assert verdicts[0].text == "PASS"
        assert verdicts[1].text == "FAIL"

    def test_no_verdict_when_no_run_tests(self):
        """Trajectory with no run_tests steps → zero verdict spans."""
        steps = [
            _step("reason", outcome="lots of output here", n=1),
            _step("edit",   outcome="diff output text",    n=2),
        ]
        rt = render_trajectory(_traj(steps, final_outcome=False))
        assert len(_spans_by_role(rt, "verdict")) == 0

    def test_multiple_run_tests_get_multiple_verdicts(self):
        """Two run_tests steps → exactly two verdict spans."""
        steps = [
            _step("run_tests", outcome="1 failed", reward=0.0, n=1),
            _step("edit",      outcome="patched",              n=2),
            _step("run_tests", outcome="1 passed", reward=1.0, n=3),
        ]
        rt = render_trajectory(_traj(steps))
        assert len(_spans_by_role(rt, "verdict")) == 2


# ── BUG2 regression ───────────────────────────────────────────────────────────

class TestBug2EmptyReasonBlocks:
    """BUG2: every non-reason step emitted empty trainable <reason> spans."""

    def test_no_empty_reason_spans(self):
        """No reason span should ever have empty text."""
        steps = [
            _step("reason", output_action="I should look at the tests", n=1),
            _step("edit",   output_action="def foo(): return 42",       n=2),
            _step("run_tests", reward=1.0, n=3),
        ]
        rt = render_trajectory(_traj(steps))
        empty = [s for s in rt.spans if s.role == "reason" and s.text == ""]
        assert empty == [], f"Found {len(empty)} empty reason span(s) — BUG2 regression"

    def test_reason_span_only_on_reason_step(self):
        """A reason span should exist only when action_type == 'reason'."""
        steps = [
            _step("edit",     n=1),
            _step("reason",   output_action="thinking...", n=2),
            _step("run_tests", reward=0.0, n=3),
        ]
        rt = render_trajectory(_traj(steps, final_outcome=False))
        reason_spans = _spans_by_role(rt, "reason")
        # Should be exactly 1: from the reason step
        assert len(reason_spans) == 1
        assert reason_spans[0].text == "thinking..."

    def test_no_reason_spans_when_no_reason_steps(self):
        """Pure edit/test trajectory should produce zero reason spans."""
        steps = [
            _step("edit",     output_action="x = 1", n=1),
            _step("run_tests", reward=1.0,           n=2),
        ]
        rt = render_trajectory(_traj(steps))
        assert len(_spans_by_role(rt, "reason")) == 0


# ── Trainability invariants ───────────────────────────────────────────────────

class TestTrainabilityInvariants:
    """World-supplied spans must not be trainable; model spans must be."""

    def test_state_spans_not_trainable(self):
        rt = render_trajectory(_traj([_step("edit", n=1)]))
        state_spans = _spans_by_role(rt, "state")
        assert all(not s.trainable for s in state_spans), \
            "State spans must not be trainable (world-supplied context)"

    def test_verdict_spans_not_trainable(self):
        rt = render_trajectory(_traj([_step("run_tests", reward=1.0, n=1)]))
        verdict_spans = _spans_by_role(rt, "verdict")
        assert all(not s.trainable for s in verdict_spans), \
            "Verdict spans must not be trainable (world-supplied oracle)"

    def test_action_spans_are_trainable(self):
        rt = render_trajectory(_traj([_step("edit", output_action="x=1", n=1)]))
        action_spans = _spans_by_role(rt, "action")
        assert all(s.trainable for s in action_spans)

    def test_conf_spans_are_trainable(self):
        rt = render_trajectory(_traj([_step("edit", n=1)]))
        conf_spans = _spans_by_role(rt, "conf")
        assert all(s.trainable for s in conf_spans)

    def test_task_struct_not_trainable(self):
        rt = render_trajectory(_traj([_step("edit", n=1)]))
        # <task> span is struct and should not be trainable
        struct_spans = [s for s in rt.spans if s.role == "struct"]
        task_span = next((s for s in struct_spans if s.text.startswith("<task>")), None)
        assert task_span is not None
        assert not task_span.trainable


# ── UL mask ───────────────────────────────────────────────────────────────────

class TestULMask:
    """UL flag marks only the final failed action on failed trajectories."""

    def test_ul_on_final_failed_action(self):
        """Failed trajectory → exactly one is_ul span on the last edit."""
        steps = [
            _step("edit", output_action="attempt 1", n=1),
            _step("run_tests", reward=0.0, n=2),
            _step("edit", output_action="attempt 2", n=3),
            _step("run_tests", reward=0.0, n=4),
        ]
        rt = render_trajectory(_traj(steps, final_outcome=False))
        ul_spans = [s for s in rt.spans if s.is_ul]
        assert len(ul_spans) == 1
        assert ul_spans[0].text == "attempt 2"

    def test_no_ul_on_success_trajectory(self):
        """Success trajectory → zero is_ul spans anywhere."""
        steps = [
            _step("edit", output_action="fix", n=1),
            _step("run_tests", reward=1.0, n=2),
        ]
        rt = render_trajectory(_traj(steps, final_outcome=True))
        ul_spans = [s for s in rt.spans if s.is_ul]
        assert ul_spans == [], f"No UL spans on success trajectory, got {len(ul_spans)}"


# ── Trajectory weight ─────────────────────────────────────────────────────────

class TestTrajectoryWeight:
    """compute_traj_weight should reward success and penalize long trajectories."""

    def test_success_weight_greater_than_failure(self):
        success = _traj([_step("edit", n=1)], final_outcome=True)
        failure = _traj([_step("edit", n=1)], final_outcome=False)
        assert compute_traj_weight(success) > compute_traj_weight(failure)

    def test_shorter_success_has_higher_weight(self):
        """Efficiency bonus: fewer steps = higher weight (up to 1.5x)."""
        short = _traj([_step("edit", n=1)], final_outcome=True)
        long_steps = [_step("edit", n=i+1) for i in range(20)]
        long = _traj(long_steps, final_outcome=True)
        assert compute_traj_weight(short) >= compute_traj_weight(long)

    def test_weight_always_positive(self):
        for outcome in (True, False):
            traj = _traj([_step("edit", n=1)], final_outcome=outcome)
            assert compute_traj_weight(traj) > 0


# ── Rollback rendering ────────────────────────────────────────────────────────

class TestRollbackRendering:
    def test_rollback_span_emitted_when_was_rollback(self):
        rb_step = _step("edit", was_rollback=True,
                        rollback_reason="tests failed badly", n=1)
        rt = render_trajectory(_traj([rb_step], final_outcome=False))
        rb_spans = _spans_by_role(rt, "rollback")
        assert len(rb_spans) == 1
        assert "tests failed badly" in rb_spans[0].text
        assert rb_spans[0].trainable

    def test_no_rollback_span_when_not_rollback(self):
        rt = render_trajectory(_traj([_step("edit", n=1)]))
        assert len(_spans_by_role(rt, "rollback")) == 0
