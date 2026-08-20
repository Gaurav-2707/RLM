"""
P1-P4 Recursion Proof evaluation harness (spec §4.2).

The harness is model-agnostic: pass a `ModelRunner` that knows how to
(a) roll out a task to a verified PASS/FAIL and (b) read the model's
rollback-token probability at a given prefix. Tasks are described with the
same RLMTrajectory grammar so eval matches training.

Metrics produced:
  - Recovery Rate (RR)            : recover-from-first-failure rate
  - P1 flip_response_rate         : reacts to a forced FAIL verdict
  - P2 novel_error_RR             : RR on unseen error types
  - P3 feedback_value             : RR_with - RR_without (verdict ablated)
  - P4 rollback_necessity_RR      : solves tasks that require a rollback
  - rollback precision/recall, ECE
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any, Optional

from .tokens import conf_to_token, CONF_TOKENS, CONF_BIN_EDGES


# --------------------------------------------------------------------------
# Runner interface — implemented by the training script against the live model
# --------------------------------------------------------------------------
class ModelRunner:
    """Adapter the eval harness drives. Implement against your model+env."""

    def rollout(self, task: "EvalTask", feedback: bool = True,
                force_first_verdict: Optional[str] = None) -> "Rollout":
        """Run the model on a task until verified terminal or step budget.

        feedback=False  -> hide <test> verdicts (P3 ablation).
        force_first_verdict in {"PASS","FAIL"} -> override first verdict (P1).
        """
        raise NotImplementedError

    def rollback_prob(self, prefix_grammar: str) -> float:
        """P(<rollback>) at the next position given a rendered prefix."""
        raise NotImplementedError


@dataclass
class EvalTask:
    task_id: str
    task_description: str
    repo_path: str
    test_command: str
    error_type: str = "generic"
    novel_error: bool = False          # P2
    requires_rollback: bool = False    # P4
    # oracle labels: step index -> is this state genuinely broken?
    broken_states: Dict[int, bool] = field(default_factory=dict)


@dataclass
class Rollout:
    task_id: str
    first_action_failed: bool
    final_pass: bool
    num_steps: int
    rollbacks: List[int] = field(default_factory=list)   # step indices rolled back
    # per-step (confidence_bucket_token, step_passed) for calibration
    conf_events: List[tuple] = field(default_factory=list)
    reacted_to_flip: bool = False      # P1


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def recovery_rate(rollouts: List[Rollout]) -> float:
    recov = [r for r in rollouts if r.first_action_failed]
    if not recov:
        return float("nan")
    return sum(r.final_pass for r in recov) / len(recov)


def rollback_precision_recall(rollouts: List[Rollout],
                              tasks: Dict[str, EvalTask]) -> Dict[str, float]:
    tp = fp = fn = 0
    for r in rollouts:
        broken = tasks[r.task_id].broken_states
        rolled = set(r.rollbacks)
        for idx, was_broken in broken.items():
            in_rolled = idx in rolled
            if in_rolled and was_broken:
                tp += 1
            elif in_rolled and not was_broken:
                fp += 1
            elif (not in_rolled) and was_broken:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"rollback_precision": prec, "rollback_recall": rec}


def expected_calibration_error(rollouts: List[Rollout], n_buckets: int = 5) -> float:
    # midpoint confidence per bucket token
    edges = [0.0] + CONF_BIN_EDGES + [1.0]
    mids = {CONF_TOKENS[i]: (edges[i] + edges[i + 1]) / 2 for i in range(len(CONF_TOKENS))}
    bins: Dict[str, List[int]] = {t: [] for t in CONF_TOKENS}
    for r in rollouts:
        for tok, passed in r.conf_events:
            if tok in bins:
                bins[tok].append(1 if passed else 0)
    total = sum(len(v) for v in bins.values())
    if total == 0:
        return float("nan")
    ece = 0.0
    for tok, outcomes in bins.items():
        if not outcomes:
            continue
        acc = sum(outcomes) / len(outcomes)
        ece += (len(outcomes) / total) * abs(acc - mids[tok])
    return ece


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------
def recursion_proof(runner: ModelRunner, tasks: List[EvalTask]) -> Dict[str, Any]:
    task_map = {t.task_id: t for t in tasks}

    with_fb = [runner.rollout(t, feedback=True) for t in tasks]
    without_fb = [runner.rollout(t, feedback=False) for t in tasks]

    # P1: force a FAIL verdict on a known-good first action, expect a reaction
    flip_tasks = [t for t in tasks if not t.requires_rollback]
    flips = [runner.rollout(t, feedback=True, force_first_verdict="FAIL")
             for t in flip_tasks]
    flip_rate = (sum(r.reacted_to_flip for r in flips) / len(flips)
                 if flips else float("nan"))

    # P2: novel-error subset
    novel = [r for r, t in zip(with_fb, tasks) if t.novel_error]
    # P4: rollback-necessity subset
    p4 = [r for r, t in zip(with_fb, tasks) if t.requires_rollback]

    rr_with = recovery_rate(with_fb)
    rr_without = recovery_rate(without_fb)

    metrics = {
        "recovery_rate": rr_with,
        "P1_flip_response_rate": flip_rate,
        "P2_novel_error_RR": recovery_rate(novel) if novel else float("nan"),
        "P3_feedback_value": (rr_with - rr_without)
        if not _isnan(rr_with) and not _isnan(rr_without) else float("nan"),
        "P4_rollback_necessity_RR": recovery_rate(p4) if p4 else float("nan"),
        "median_steps": _median([r.num_steps for r in with_fb]),
    }
    metrics.update(rollback_precision_recall(with_fb, task_map))
    metrics["ece"] = expected_calibration_error(with_fb)
    metrics["passes_recursion_bar"] = _passes_bar(metrics)
    return metrics


# --- pass/fail gate from spec §4.2 ---
def _passes_bar(m: Dict[str, Any]) -> bool:
    def ok(v):
        return v is not None and not _isnan(v)
    return (
        ok(m["P3_feedback_value"]) and m["P3_feedback_value"] >= 0.25 and
        ok(m["rollback_precision"]) and m["rollback_precision"] >= 0.70 and
        ok(m["ece"]) and m["ece"] <= 0.10 and
        ok(m["P1_flip_response_rate"]) and m["P1_flip_response_rate"] >= 0.80
    )


def _isnan(x):
    return x != x


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
