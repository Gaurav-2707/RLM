"""
Render an RLMTrajectory into the recursive grammar as a list of spans.

Each span carries:
  - text:       the raw string
  - trainable:  whether the loss is computed over it (model-generated spans)
  - role:       "struct" | "state" | "reason" | "action" | "verdict" |
                "rollback" | "conf" | "final"
  - is_ul:      True only on the *final failed action* (unlikelihood target)

World-supplied spans (state, test verdict) are NOT trainable: they are
context. Decision-bearing spans (action, rollback, conf) are trainable and
later upweighted by the collator.
"""

from dataclasses import dataclass
from typing import List

from RLM.utils.trajectory import RLMTrajectory, TrajectoryStep
from .tokens import conf_to_token

STATE_TAIL_CHARS = 2000  # truncate noisy state/diff blocks to their tail


@dataclass
class Span:
    text: str
    trainable: bool
    role: str
    is_ul: bool = False


@dataclass
class RenderedTrajectory:
    spans: List[Span]
    traj_weight: float
    is_success: bool
    num_rollbacks: int
    task_id: str


def _truncate_tail(s: str, n: int = STATE_TAIL_CHARS) -> str:
    s = s or ""
    return s if len(s) <= n else "...(truncated)...\n" + s[-n:]


def compute_traj_weight(
    traj: RLMTrajectory,
    max_steps: int = 20,
    base_success: float = 1.0,
    base_failure: float = 0.25,
    gamma: float = 0.5,
) -> float:
    """w_traj = base(outcome) * efficiency_bonus  (see spec §1.5)."""
    base = base_success if traj.final_outcome else base_failure
    steps = max(1, traj.total_steps)
    eff = 1.0 + gamma * (1.0 - min(steps, max_steps) / max_steps)
    eff = max(0.5, min(1.5, eff))
    return base * eff


def _last_action_index(traj: RLMTrajectory) -> int:
    """Index of the final action-bearing step (for UL on failed trajectories)."""
    for i in range(len(traj.steps) - 1, -1, -1):
        if traj.steps[i].action_type not in ("run_tests", "rollback"):
            return i
    return len(traj.steps) - 1


def render_trajectory(traj: RLMTrajectory, **weight_kwargs) -> RenderedTrajectory:
    spans: List[Span] = []
    is_success = bool(traj.final_outcome)
    ul_idx = _last_action_index(traj) if not is_success else -1

    spans.append(Span(f"<task> {traj.task_description.strip()} </task>\n", False, "struct"))

    for i, step in enumerate(traj.steps):
        spans.append(Span(f"<step>\n", False, "struct"))

        # world-supplied state (context, masked)
        spans.append(Span("<state> ", False, "struct"))
        spans.append(Span(_truncate_tail(step.input_state), False, "state"))
        spans.append(Span(" </state>\n", False, "struct"))

        # model reasoning (trainable) — only emit on reason steps, never empty blocks
        if step.action_type == "reason":
            spans.append(Span("<reason> ", True, "struct"))
            spans.append(Span(step.output_action.strip(), True, "reason"))
            spans.append(Span(" </reason>\n", True, "struct"))

        # rollback marker (trainable decision token)
        if step.was_rollback:
            spans.append(Span(f'<rollback> {step.rollback_reason.strip()} </rollback>\n',
                              True, "rollback"))

        # action (trainable decision-bearing); UL flag on final failed action
        if step.action_type not in ("reason", "run_tests", "rollback"):
            spans.append(Span("<action> ", True, "action"))
            spans.append(Span(step.output_action.strip(), True, "action",
                              is_ul=(i == ul_idx)))
            spans.append(Span(" </action>\n", True, "action"))

        # test verdict (world-supplied, masked) — only on actual test-run steps
        if step.action_type == "run_tests":
            verdict = "PASS" if step.reward > 0 else "FAIL"
            spans.append(Span("<test> ", False, "struct"))
            spans.append(Span(verdict, False, "verdict"))
            spans.append(Span(" </test>\n", False, "struct"))

        # confidence bucket (trainable decision token)
        spans.append(Span("<conf> ", True, "struct"))
        spans.append(Span(conf_to_token(step.confidence), True, "conf"))
        spans.append(Span(" </conf>\n", True, "struct"))

        spans.append(Span("</step>\n", False, "struct"))

    outcome = "SUCCESS" if is_success else "FAILURE"
    spans.append(Span(f"<final> {outcome}\n", True, "final"))

    return RenderedTrajectory(
        spans=spans,
        traj_weight=compute_traj_weight(traj, **weight_kwargs),
        is_success=is_success,
        num_rollbacks=traj.total_rollbacks,
        task_id=traj.trajectory_id,
    )
