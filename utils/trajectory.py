"""
RLM Trajectory Format — The Training Currency of Recursive Labs.

Every time RLM Runtime runs a task and tests verify the outcome,
a trajectory is produced. These trajectories are the ground-truth labeled
dataset we use to train foundational RLM models.

Why this matters:
  - OpenAI's training signal: human preference (RLHF) — subjective, expensive
  - Our training signal: compiler/test verified outcomes — objective, free
  - Ground truth beats human preference for agentic tasks

Schema is versioned. Every training run pins to a schema version.
"""

import json
import os
import uuid
import time
import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORIES_DIR = os.path.expanduser("~/.rlm/traces")


@dataclass
class TrajectoryStep:
    """A single step in an RLM execution: observe → reason → act → outcome."""
    step_number: int
    iteration: int
    action_type: str          # "reason", "edit_file", "run_tests", "rollback", "memory_retrieve"
    input_state: str          # what the model saw
    output_action: str        # what the model did (code block or reasoning)
    outcome: str              # result (test output, file diff, memory retrieved)
    reward: float             # execution-grounded reward (-1.0 to 1.0)
    confidence: float         # model's reported confidence (0.0 to 1.0)
    was_rollback: bool = False
    rollback_reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class RLMTrajectory:
    """
    A complete verified execution trajectory.

    This is the atomic unit of training data for RLM foundational models.
    Only trajectories with verified=True are used for training.
    """
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = TRAJECTORY_SCHEMA_VERSION

    # What model produced this
    model_provider: str = ""
    model_name: str = ""
    rlm_runtime_version: str = "0.1.0"

    # The task
    task_description: str = ""
    repo_path: str = ""
    test_command: str = ""

    # The execution
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_steps: int = 0
    total_rollbacks: int = 0

    # The outcome — objectively verified
    final_outcome: bool = False       # True = tests passed
    verified: bool = False            # True = outcome confirmed by compiler/tests
    final_reward: float = 0.0
    final_diff: str = ""              # unified diff of changes made

    # Contribution
    contributed: bool = False         # user opted in for training upload
    anonymized: bool = False          # PII stripped

    # Timestamps
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    runtime_seconds: float = 0.0

    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)
        self.total_steps = len(self.steps)
        if step.was_rollback:
            self.total_rollbacks += 1

    def finalize(self, success: bool, final_diff: str = ""):
        self.final_outcome = success
        self.verified = True           # caller only calls this after test verification
        self.final_diff = final_diff
        self.completed_at = time.time()
        self.runtime_seconds = self.completed_at - self.started_at
        self.final_reward = 1.0 if success else -0.5

    def anonymize(self):
        """Strip PII before contributing. Removes API keys, emails, internal hostnames."""
        patterns = [
            (r'sk-[A-Za-z0-9]{20,}', '[API_KEY]'),          # OpenAI keys
            (r'AIza[A-Za-z0-9_\-]{35}', '[API_KEY]'),        # Google keys
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
            (r'https?://[a-zA-Z0-9\-\.]+\.internal[^\s]*', '[INTERNAL_URL]'),
            (r'Bearer\s+[A-Za-z0-9\-_\.]+', 'Bearer [TOKEN]'),
        ]
        def _clean(text: str) -> str:
            for pattern, replacement in patterns:
                text = re.sub(pattern, replacement, text)
            return text

        for step in self.steps:
            step.input_state = _clean(step.input_state)
            step.output_action = _clean(step.output_action)
            step.outcome = _clean(step.outcome)

        self.task_description = _clean(self.task_description)
        self.final_diff = _clean(self.final_diff)
        self.repo_path = "[REPO_PATH]"
        self.anonymized = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, directory: str = TRAJECTORIES_DIR):
        """Persist trajectory to disk as JSONL."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.trajectory_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "RLMTrajectory":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps = [TrajectoryStep(**s) for s in data.pop("steps", [])]
        t = cls(**data)
        t.steps = steps
        return t


class TrajectoryCollector:
    """
    Collects trajectories from RLM Runtime runs.

    When contribute_traces=True, anonymizes and queues for upload.
    When contribute_traces=False, saves locally only (for future training).

    Users who contribute get: free tier access, priority features,
    leaderboard credit. We get: ground-truth labeled training data.
    """

    def __init__(self, contribute: bool = False, save_dir: str = TRAJECTORIES_DIR):
        self.contribute = contribute
        self.save_dir = save_dir
        self._active: Optional[RLMTrajectory] = None

    def start(
        self,
        model_provider: str,
        model_name: str,
        task_description: str,
        repo_path: str = "",
        test_command: str = "",
    ) -> RLMTrajectory:
        self._active = RLMTrajectory(
            model_provider=model_provider,
            model_name=model_name,
            task_description=task_description,
            repo_path=repo_path,
            test_command=test_command,
            contributed=self.contribute,
        )
        return self._active

    @property
    def active(self) -> Optional[RLMTrajectory]:
        return self._active

    def record_step(
        self,
        step_number: int,
        iteration: int,
        action_type: str,
        input_state: str,
        output_action: str,
        outcome: str,
        reward: float,
        confidence: float,
        was_rollback: bool = False,
        rollback_reason: str = "",
    ):
        if self._active is None:
            return
        step = TrajectoryStep(
            step_number=step_number,
            iteration=iteration,
            action_type=action_type,
            input_state=input_state,
            output_action=output_action,
            outcome=outcome,
            reward=reward,
            confidence=confidence,
            was_rollback=was_rollback,
            rollback_reason=rollback_reason,
        )
        self._active.add_step(step)

    def finish(self, success: bool, final_diff: str = "") -> Optional[str]:
        """Finalize trajectory, save to disk, and optionally contribute."""
        if self._active is None:
            return None
        self._active.finalize(success, final_diff)
        if self.contribute:
            self._active.anonymize()
        path = self._active.save(self.save_dir)
        # Fire-and-forget upload — never blocks the caller
        if self.contribute and self._active.verified:
            try:
                from RLM.utils.upload import upload_trajectory_async
                upload_trajectory_async(self._active)
            except Exception:
                pass   # graceful degrade: upload failure never crashes the REPL
        self._active = None
        return path


def load_all_trajectories(directory: str = TRAJECTORIES_DIR) -> List[RLMTrajectory]:
    """Load all saved trajectories from disk."""
    if not os.path.exists(directory):
        return []
    trajectories = []
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            try:
                t = RLMTrajectory.load(os.path.join(directory, fname))
                trajectories.append(t)
            except Exception:
                pass
    return trajectories


def trajectory_stats(directory: str = TRAJECTORIES_DIR) -> Dict[str, Any]:
    """Quick summary stats for the local trajectory corpus."""
    trajs = load_all_trajectories(directory)
    if not trajs:
        return {"total": 0}
    verified = [t for t in trajs if t.verified]
    successful = [t for t in verified if t.final_outcome]
    contributed = [t for t in trajs if t.contributed]
    return {
        "total": len(trajs),
        "verified": len(verified),
        "successful": len(successful),
        "success_rate": len(successful) / len(verified) if verified else 0.0,
        "with_rollbacks": sum(1 for t in trajs if t.total_rollbacks > 0),
        "contributed": len(contributed),
        "avg_steps": sum(t.total_steps for t in trajs) / len(trajs),
        "avg_rollbacks": sum(t.total_rollbacks for t in trajs) / len(trajs),
        "models_seen": list({t.model_name for t in trajs}),
    }
