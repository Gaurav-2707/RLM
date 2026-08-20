"""
Trajectory Quality Scorer — filters the corpus before training.

We cannot blindly train on every contributed trajectory. Low-quality
trajectories (trivially easy tasks, redundant copies of the same bug type,
trajectories where the model got lucky rather than being correct) add noise
and slow convergence.

The scorer evaluates five dimensions:

  1. Recursion signal  (0.35) — trajectories with rollback→recovery are more
     valuable than straight wins. A trajectory that failed, rolled back, and
     recovered demonstrates the exact behavior we're training.

  2. Task difficulty   (0.25) — more iterations before success = harder problem
     = richer supervision. One-shot fixes are worth less.

  3. Novelty          (0.20) — cosine distance from the existing corpus on
     (task_description + action text). Novel bug types > the 100th off-by-one.

  4. Verification depth (0.15) — number of distinct test assertions that
     changed from FAIL→PASS. More tests = harder problem = better signal.

  5. Efficiency        (0.05) — terse successful trajectories over verbose ones.
     Wandering models produce noisy demonstrations.

Usage:
    scorer = TrajectoryScorer()
    scores = [scorer.score(t, corpus) for t in new_trajectories]
    good   = scorer.filter_corpus(new_trajectories, corpus, min_score=0.4)
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

from RLM.utils.trajectory import RLMTrajectory


# ---------------------------------------------------------------------------
# Weights — must sum to 1.0
# ---------------------------------------------------------------------------
_W_RECURSION    = 0.35
_W_DIFFICULTY   = 0.25
_W_NOVELTY      = 0.20
_W_VERIFICATION = 0.15
_W_EFFICIENCY   = 0.05

assert abs(_W_RECURSION + _W_DIFFICULTY + _W_NOVELTY + _W_VERIFICATION + _W_EFFICIENCY - 1.0) < 1e-9


class TrajectoryScorer:
    """
    Scores trajectories for training value.

    Parameters
    ----------
    max_steps : int
        Cap for the difficulty score computation. Steps beyond this are
        treated as equivalent (very hard).
    novelty_vocab_size : int
        Number of top-frequency terms to keep in the TF-IDF vocabulary.
    """

    def __init__(self, max_steps: int = 30, novelty_vocab_size: int = 2000):
        self.max_steps = max_steps
        self.novelty_vocab_size = novelty_vocab_size
        self._vocab: Optional[dict] = None       # term → idf, built lazily
        self._corpus_vecs: List[dict] = []       # tf-idf vectors for corpus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, traj: RLMTrajectory,
              corpus: List[RLMTrajectory]) -> float:
        """
        Score a single trajectory against the existing corpus.

        Returns a float in [0, 1]. Higher = more training value.
        Cold start: when corpus is empty, returns 0.5 uniform.

        Parameters
        ----------
        traj   : The trajectory to score.
        corpus : The already-accepted training corpus (for novelty comparison).
        """
        recursion_score    = self._score_recursion(traj)
        difficulty_score   = self._score_difficulty(traj)
        novelty_score      = self._score_novelty(traj, corpus)
        verification_score = self._score_verification(traj)
        efficiency_score   = self._score_efficiency(traj)

        combined = (
            _W_RECURSION    * recursion_score
            + _W_DIFFICULTY   * difficulty_score
            + _W_NOVELTY      * novelty_score
            + _W_VERIFICATION * verification_score
            + _W_EFFICIENCY   * efficiency_score
        )
        return max(0.0, min(1.0, combined))

    def filter_corpus(
        self,
        candidates: List[RLMTrajectory],
        existing_corpus: Optional[List[RLMTrajectory]] = None,
        min_score: float = 0.4,
    ) -> List[RLMTrajectory]:
        """
        Return only candidates that score >= min_score.

        Parameters
        ----------
        candidates      : New trajectories to evaluate.
        existing_corpus : Already-accepted training data (can be None/empty).
        min_score       : Rejection threshold. Default 0.4.
        """
        corpus = existing_corpus or []
        accepted = []
        for traj in candidates:
            if self.score(traj, corpus) >= min_score:
                accepted.append(traj)
                corpus = corpus + [traj]   # progressive novelty update
        return accepted

    # ------------------------------------------------------------------
    # Individual dimension scorers
    # (each returns float in [0, 1])
    # ------------------------------------------------------------------

    def _score_recursion(self, traj: RLMTrajectory) -> float:
        """
        Trajectories with rollback→recovery score highest.
        Straight successful solves score medium.
        Failed trajectories with no recovery score lowest.
        """
        if traj.total_rollbacks > 0 and traj.final_outcome:
            return 1.0
        elif traj.final_outcome:
            return 0.6
        else:
            return 0.2

    def _score_difficulty(self, traj: RLMTrajectory) -> float:
        """
        Score = min(total_steps, max_steps) / max_steps.
        More iterations before success = harder problem = more signal.
        """
        return min(traj.total_steps, self.max_steps) / self.max_steps

    def _score_novelty(self, traj: RLMTrajectory,
                       corpus: List[RLMTrajectory]) -> float:
        """
        1 - max_cosine_similarity against the nearest corpus neighbour (TF vectors).
        Returns 0.5 on cold start (empty corpus).
        """
        if not corpus:
            return 0.5
        traj_vec = self._tf(self._tokenize(self._text(traj)))
        if not traj_vec:
            return 0.5
        max_sim = max(
            self._cosine_sim(traj_vec, self._tf(self._tokenize(self._text(c))))
            for c in corpus
        )
        return max(0.0, min(1.0, 1.0 - max_sim))

    def _score_verification(self, traj: RLMTrajectory) -> float:
        """
        Count FAIL→PASS transitions across run_tests steps (reward flipped from
        ≤0 to >0).  Normalize by max_steps.
        """
        transitions = 0
        prev_reward = 0.0
        for step in traj.steps:
            if step.action_type == "run_tests":
                if prev_reward <= 0.0 < step.reward:
                    transitions += 1
                prev_reward = step.reward
        return min(1.0, transitions / max(1, self.max_steps))

    def _score_efficiency(self, traj: RLMTrajectory) -> float:
        """
        1.0 - (total_steps / max_steps), clamped to [0, 1].
        0.0 for failed trajectories (we don't reward terse failures).
        """
        if not traj.final_outcome:
            return 0.0
        return max(0.0, min(1.0, 1.0 - traj.total_steps / self.max_steps))

    # ------------------------------------------------------------------
    # TF-IDF helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text(traj: RLMTrajectory) -> str:
        """Extract all relevant text from a trajectory for vectorization."""
        parts = [traj.task_description or ""]
        for step in traj.steps:
            if step.output_action:
                parts.append(step.output_action)
        return " ".join(parts).lower()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"[a-z0-9_]+", text.lower())

    @staticmethod
    def _tf(tokens: List[str]) -> dict:
        counts: dict = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = max(1, len(tokens))
        return {t: c / total for t, c in counts.items()}

    @staticmethod
    def _cosine_sim(a: dict, b: dict) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
