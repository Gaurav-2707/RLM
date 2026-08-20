"""
TestDrivenEnv: The core TDRL Gymnasium environment for the RLM architecture.

This environment models the Software Engineering loop as a Markov Decision
Process (MDP) where an RL Agent learns when to EDIT code, VERIFY via tests,
or ROLLBACK to a safe state using Episodic Memory.
"""

import os
import json
import re
import subprocess
import difflib
import glob
import random
from typing import Tuple, Optional, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rich.console import Console
from rich.text import Text

console = Console()


def _log(msg: str, style: str = "white") -> None:
    """Print a styled log line via the rich Console.

    Args:
        msg: The message to print.
        style: A rich style string (e.g. "bold red", "green", "cyan").
    """
    console.print(Text(msg, style=style))


class TestDrivenEnv(gym.Env):
    """Test-Driven Reinforcement Learning (TDRL) Gymnasium environment.

    Models the Software Engineering development loop as an MDP:
      - State:   [step, test_ratio, last_status, confidence, ctx_len] + 384-D error embedding
      - Actions: 0=RUN_TESTS, 1=EDIT_FILE, 2=ROLLBACK
      - Reward:  Execution-grounded via compiler / test suite output

    The core Episodic Memory system snapshots the codebase to RAM before
    every LLM edit. If a subsequent test run fails, action=2 (ROLLBACK)
    physically restores the last known-good state from that snapshot.

    Args:
        repo_path: Absolute or relative path to the target codebase.
        test_command: Shell command used to run the test suite (e.g. "pytest").
        reward_type: One of "binary", "granular", or "both" (hybrid).
        max_steps: Hard step limit before the episode terminates.
    """

    def __init__(
        self,
        repo_path: str,
        test_command: str = "pytest",
        reward_type: str = "both",
        max_steps: int = 20,
        inject_synthetic_bug: bool = False,
        memory_system: Optional[Any] = None,
    ):
        super().__init__()

        self.repo_path = repo_path
        self.test_command = test_command
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.inject_synthetic_bug = inject_synthetic_bug

        # Optional ACC / Memory System (graceful degradation if not installed)
        try:
            from RLM.acc.controller import AdaptiveComputeController
            from RLM.memory.system import EpisodicMemorySystem
            self.memory_system = memory_system or EpisodicMemorySystem(capacity=50)
            self.acc = AdaptiveComputeController()
        except ImportError:
            self.memory_system = memory_system
            self.acc = None

        # Action space: RUN_TESTS=0, EDIT_FILE=1, ROLLBACK=2
        self.action_space = spaces.Discrete(3)

        # Observation: 5 scalars + 384-D sentence-transformer embedding
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(389,), dtype=np.float32
        )

        # Scalar state variables
        self.current_step = 0
        self.tests_passed_ratio = 0.0
        self.last_action_status = 0.0
        self.current_confidence = 0.5
        self.context_len = 0.1
        self.test_output = ""
        self.failed_edits = []
        self.file_snapshots: dict = {}
        self.last_edit: Optional[dict] = None
        self.execution_context: Dict[str, Any] = {}
        self.trace_events = []

        # Load the sentence-transformer once at init
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ─────────────────────────────── Observation ───────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build the 389-D observation vector.

        Concatenates 5 scalar state features with a 384-D sentence-transformer
        embedding of the most recent test output.

        Returns:
            np.ndarray: Shape (389,), dtype float32.
        """
        scalars = np.array(
            [
                self.current_step / self.max_steps,
                self.tests_passed_ratio,
                self.last_action_status,
                self.current_confidence,
                self.context_len,
            ],
            dtype=np.float32,
        )

        if not self.test_output:
            text_embed = np.zeros(384, dtype=np.float32)
        else:
            text_embed = self.embedding_model.encode(
                self.test_output[:1000]
            ).astype(np.float32)

        return np.concatenate([scalars, text_embed])

    # ─────────────────────────────── Reset ────────────────────────────────

    def set_execution_context(self, context: Dict[str, Any]):
        """Attach planner, memory, and trace metadata to the next edit prompt."""
        self.execution_context = context or {}

    def _record_event(self, event_type: str, **payload):
        event = {
            "step": self.current_step,
            "type": event_type,
            **payload,
        }
        self.trace_events.append(event)
        return event

    def reset(self, seed=None, options=None):
        """Reset the environment for a new training episode.

        Injects a random synthetic bug into auth.py to ensure training
        generalises across multiple failure modes.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Unused; present for Gymnasium API compliance.

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and empty info dict.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.tests_passed_ratio = 0.0
        self.last_action_status = 0.0
        self.current_confidence = 0.5
        self.context_len = 0.1
        self.test_output = ""
        self.failed_edits = []
        self.file_snapshots = {}
        self.last_edit = None
        self.trace_events = []

        # Synthetic Training Variance: inject random bugs to prevent overfitting
        target_file = os.path.join(self.repo_path, "auth.py")
        if self.inject_synthetic_bug and os.path.exists(target_file):
            bug_type = random.choice(["syntax", "logic_greater", "logic_less", "clean"])
            content = "from db import save_user_session\n\ndef login(password_length: int) -> str:\n"

            if bug_type == "syntax":
                content += "    if password_length >= 8\n        security_level = 1\n"
            elif bug_type == "logic_greater":
                content += "    if password_length > 8:\n        security_level = 1\n"
            elif bug_type == "logic_less":
                content += "    if password_length < 8:\n        security_level = 1\n"
            else:
                content += "    if password_length >= 8:\n        security_level = 1\n"

            content += (
                "        return save_user_session(user_id=123, security_level=security_level)\n"
                "    else:\n        return \"Login Failed\"\n"
            )

            with open(target_file, "w") as f:
                f.write(content)
            self._record_event("synthetic_bug_injected", file=target_file, bug_type=bug_type)

        return self._get_observation(), {}

    # ─────────────────────────────── Test Runner ─────────────────────────

    def _run_tests(self) -> Tuple[float, float, int, int]:
        """Execute the test suite and compute a scalar reward.

        Returns:
            Tuple of (reward, passed_ratio, passed_count, total_count).
        """
        try:
            result = subprocess.run(
                self.test_command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            self.test_output = result.stdout + "\n" + result.stderr
            self._record_event(
                "tests_run",
                command=self.test_command,
                returncode=result.returncode,
                output=self.test_output[-4000:],
            )

            if result.returncode == 0:
                passed, total = 1, 1
            else:
                passed, total = 0, 1

            ratio = passed / total if total > 0 else 0.0

            if self.reward_type == "binary":
                reward = 10.0 if result.returncode == 0 else -1.0
            elif self.reward_type == "granular":
                reward = (ratio * 10.0) - 1.0
            else:  # hybrid
                reward = (ratio * 5.0) + (5.0 if result.returncode == 0 else -1.0)

            return reward, ratio, passed, total

        except Exception as exc:
            self.test_output = str(exc)
            self._record_event("tests_error", command=self.test_command, error=str(exc))
            return -2.0, 0.0, 0, 1

    # ─────────────────────────── LLM Edit Engine ─────────────────────────

    @staticmethod
    def _parse_llm_json(raw: str) -> dict:
        """Aggressively parse LLM output into a JSON dict.

        Handles markdown fences, wrapping arrays, non-string field values,
        and leading/trailing prose.

        Args:
            raw: Raw string from LLMClient.completion().

        Returns:
            dict: Parsed edit specification with "file", "search", "replace".

        Raises:
            ValueError: If a valid dict cannot be extracted.
        """
        if not isinstance(raw, str):
            raw = json.dumps(raw)

        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        # Regex-extract first {...} block
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(f"No JSON object in LLM response: {raw[:200]}")

        parsed = json.loads(match.group(0))

        # Unwrap accidental arrays
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed)}")

        # Ensure string values (guard against nested dicts/lists)
        for key in ("search", "replace", "file"):
            val = parsed.get(key, "")
            if not isinstance(val, str):
                parsed[key] = json.dumps(val)

        return parsed

    @staticmethod
    def _fuzzy_replace(content: str, search_block: str, replace_block: str) -> Optional[str]:
        """Apply a fuzzy search-and-replace using difflib SequenceMatcher.

        Finds the sliding window of lines with similarity > 0.8 and replaces
        it with the replacement block.

        Args:
            content: The full file contents as a string.
            search_block: The code block to find (exact or approximate).
            replace_block: The new code to substitute in.

        Returns:
            str: Updated file contents, or None if no match found.
        """
        def normalize(text: str) -> str:
            return " ".join(text.split())

        # Exact match first (fastest path)
        if search_block in content:
            return content.replace(search_block, replace_block, 1)

        search_norm = normalize(search_block)
        if not search_norm:
            return None

        lines = content.split("\n")
        search_lines = search_block.strip().split("\n")
        window_size = len(search_lines)

        best_ratio, best_idx = 0.0, -1
        for i in range(len(lines) - window_size + 1):
            window = "\n".join(lines[i : i + window_size])
            ratio = difflib.SequenceMatcher(None, normalize(window), search_norm).ratio()
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, i

        if best_ratio > 0.8 and best_idx >= 0:
            _log(f"   ↳ Fuzzy match score: {best_ratio:.3f}", "dim")
            new_lines = lines[:best_idx] + [replace_block] + lines[best_idx + window_size :]
            return "\n".join(new_lines)

        _log(f"   ↳ No fuzzy match found (best={best_ratio:.3f})", "dim yellow")
        return None

    def _apply_llm_edit(self) -> bool:
        """Request a structured JSON code edit from the LLM and apply it.

        Pipeline:
          1. Gather codebase context and latest test failure output.
          2. Inject episodic memory of previous failed attempts.
          3. Request a {"file", "search", "replace"} JSON patch from the LLM.
          4. Snapshot the file to RAM (Episodic Memory).
          5. Apply the patch via exact match or fuzzy difflib matcher.

        Returns:
            bool: True if the patch was successfully applied to disk.
        """
        try:
            from RLM.utils.llm import LLMClient

            # 1. Gather context
            py_files = glob.glob(os.path.join(self.repo_path, "*.py"))
            code_context = ""
            for pf in py_files:
                if not os.path.basename(pf).startswith("test_"):
                    with open(pf, "r") as f:
                        code_context += f"--- {os.path.basename(pf)} ---\n{f.read()}\n\n"

            # 2. Get latest test failure
            try:
                result = subprocess.run(
                    self.test_command,
                    shell=True,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                test_output = result.stdout + "\n" + result.stderr
            except Exception:
                test_output = "Test command failed to run."

            # 3. Build prompt with Episodic Memory injection
            prompt = (
                f"You are an autonomous debugging agent. The following codebase is failing its tests.\n\n"
                f"SYSTEM 2 EXECUTION PLAN:\n{self.execution_context.get('execution_plan', 'No plan available.')}\n\n"
                f"EPISODIC MEMORY WARNINGS:\n{self.execution_context.get('failure_warnings', 'No failure warnings.')}\n\n"
                f"RELEVANT PAST EXPERIENCE:\n{self.execution_context.get('retrieved_memories', 'No retrieved memories.')}\n\n"
                f"CODEBASE:\n{code_context}\n\n"
                f"TEST OUTPUT:\n{test_output}\n\n"
                f"Your task is to fix the bug using a search and replace block. "
                f"Return ONLY a raw JSON object with keys:\n"
                f'- "file": the basename of the file to edit (e.g. "auth.py")\n'
                f'- "search": the exact block of code to find (match indentation precisely)\n'
                f'- "replace": the corrected replacement block\n'
            )

            if self.failed_edits:
                prompt += "\nPREVIOUS FAILED ATTEMPTS — do NOT repeat these:\n"
                for edit in self.failed_edits[-3:]:
                    prompt += (
                        f"- search: '{edit.get('search', '')}' "
                        f"→ replace: '{edit.get('replace', '')}'\n"
                    )

            # 4. Call LLM
            client = LLMClient()
            response_text = client.completion(
                prompt, response_format={"type": "json_object"}
            )

            # 5. Parse
            try:
                edit = self._parse_llm_json(response_text)
            except (ValueError, json.JSONDecodeError) as exc:
                _log(f"   ↳ JSON parse failed: {exc}", "dim red")
                return False

            search_block = edit.get("search", "")
            replace_block = edit.get("replace", "")
            target_file_basename = edit.get("file", "")

            if not target_file_basename:
                _log("   ↳ LLM did not specify a target file.", "dim red")
                return False

            target_file = os.path.join(self.repo_path, target_file_basename)
            if not os.path.exists(target_file):
                _log(f"   ↳ Target file not found: {target_file}", "dim red")
                return False

            with open(target_file, "r") as f:
                content = f.read()

            # Episodic Memory Snapshot (before any write)
            self.file_snapshots[target_file] = content

            # 6. Apply patch
            new_content = self._fuzzy_replace(content, search_block, replace_block)
            if new_content is None:
                self._record_event(
                    "edit_apply_failed",
                    file=target_file_basename,
                    search=search_block,
                )
                return False

            with open(target_file, "w") as f:
                f.write(new_content)

            # Log to episodic memory for future prompt injection
            self.failed_edits.append(edit)
            self.last_edit = edit
            self._record_event(
                "edit_applied",
                file=target_file_basename,
                search=search_block,
                replace=replace_block,
            )
            return True

        except Exception as exc:
            _log(f"   ↳ LLM edit pipeline error: {exc}", "dim red")
            self._record_event("edit_error", error=str(exc))
            return False

    # ─────────────────────────────── Step ─────────────────────────────────

    def step(self, action: int):
        """Advance the environment by one step.

        Executes the selected action, computes reward, and returns the
        next observation. The episode terminates when all tests pass or
        max_steps is reached.

        Args:
            action: Integer action — 0=RUN_TESTS, 1=EDIT_FILE, 2=ROLLBACK.

        Returns:
            Tuple: (obs, reward, terminated, truncated, info)
        """
        self.current_step += 1
        done = False
        reward = -0.01  # Small step penalty to encourage efficiency
        info = {}

        if action == 0:  # RUN_TESTS
            test_reward, ratio, passed, total = self._run_tests()
            reward += test_reward
            self.tests_passed_ratio = ratio
            self.last_action_status = 1.0 if test_reward > 0 else -1.0
            if self.memory_system and self.last_edit:
                from RLM.memory.base import MemoryEntry
                score = 1.0 if ratio == 1.0 else -1.0
                entry = MemoryEntry(
                    state=self.test_output[-1000:],
                    reasoning=str(self.execution_context.get("execution_plan", ""))[:1000],
                    action=json.dumps(self.last_edit),
                    outcome="tests_passed" if ratio == 1.0 else "tests_failed",
                    outcome_score=score,
                )
                self.memory_system.add_memory(entry)
                self._record_event(
                    "memory_written",
                    memory_id=entry.entry_id,
                    outcome_score=score,
                    outcome="tests_passed" if ratio == 1.0 else "tests_failed",
                )
            if ratio == 1.0:
                done = True

        elif action == 1:  # EDIT_FILE
            success = self._apply_llm_edit()
            self.last_action_status = 0.5 if success else -1.0
            reward += 0.0  # Delayed reward — only via RUN_TESTS

        elif action == 2:  # ROLLBACK — TRUE EPISODIC MEMORY
            if self.file_snapshots:
                for file_path, snapshot_content in self.file_snapshots.items():
                    with open(file_path, "w") as f:
                        f.write(snapshot_content)
                self._record_event("rollback", files=list(self.file_snapshots.keys()))
            self.last_action_status = 0.0
            reward += -0.1  # Small rollback penalty

        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_observation()
        return obs, reward, done, False, info
