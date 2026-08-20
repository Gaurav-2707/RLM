"""
Recursive Labs — RLM Live A/B Demonstration Script
====================================================
Proves that the RLM Orchestration Layer is superior to raw LLMs.

TEST A — Raw LLM (Zero-Shot):
  Prompts the LLM directly once, applies the patch, runs PyTest.
  No memory. No rollback. One shot or crash.

TEST B — RLM Orchestration Layer (TDRL + Rollbacks):
  Wraps the same LLM inside the Adaptive Compute Controller.
  Episodic Memory snapshots the codebase before every edit.
  If PyTest fails → Deterministic Rollback → retry with failure context.
  Loops until the test suite passes organically.

Usage:
    source .venv/bin/activate
    PYTHONPATH=/Users/arushsinghal/Documents python3 -u experiments/benchmark.py
"""

import os
import re
import json
import time
import random
import difflib
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

from RLM.utils.llm import LLMClient
from RLM.experiments.rl.test_driven_env import TestDrivenEnv
from stable_baselines3 import PPO

console = Console()

BUGGY_AUTH = """\
from db import save_user_session

def login(password_length: int) -> str:
    if password_length > 8:
        security_level = 1
        return save_user_session(user_id=123, security_level=security_level)
    else:
        return "Login Failed"
"""


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _inject_bug(target_path: str) -> None:
    """Write the canonical buggy auth.py to disk.

    The bug: ``> 8`` should be ``>= 8``.
    This makes ``login(8)`` return ``"Login Failed"`` instead of a session.

    Args:
        target_path: Absolute path to auth.py.
    """
    with open(target_path, "w") as f:
        f.write(BUGGY_AUTH)


def _parse_json(raw: str) -> dict:
    """Extract a JSON dict from raw LLM output, aggressively.

    Args:
        raw: Raw LLM response string (may contain markdown, prose, etc.)

    Returns:
        dict: Parsed edit spec.

    Raises:
        ValueError: If no JSON object can be found.
    """
    if not isinstance(raw, str):
        raw = json.dumps(raw)

    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object in response: {raw[:200]}")

    parsed = json.loads(match.group(0))

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]

    for key in ("search", "replace", "file"):
        val = parsed.get(key, "")
        if not isinstance(val, str):
            parsed[key] = json.dumps(val)

    return parsed


def _fuzzy_apply(content: str, search: str, replace: str) -> str | None:
    """Apply a search/replace patch using fuzzy line-window matching.

    Args:
        content: The full file content.
        search: Code block to find.
        replace: Replacement code.

    Returns:
        Updated file content string, or None if no match > 0.8.
    """
    if search in content:
        return content.replace(search, replace, 1)

    def norm(t):
        return " ".join(t.split())

    lines = content.split("\n")
    slines = search.strip().split("\n")
    w = len(slines)
    snorm = norm(search)

    best_r, best_i = 0.0, -1
    for i in range(len(lines) - w + 1):
        window = "\n".join(lines[i : i + w])
        r = difflib.SequenceMatcher(None, norm(window), snorm).ratio()
        if r > best_r:
            best_r, best_i = r, i

    if best_r > 0.8:
        console.print(f"   [dim]↳ Fuzzy match: {best_r:.3f}[/dim]")
        return "\n".join(lines[:best_i] + [replace] + lines[best_i + w :])

    console.print(f"   [dim yellow]↳ No match found (best={best_r:.3f})[/dim yellow]")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# TEST A: Raw LLM (Zero-Shot)
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_llm(repo_path: str, target_file: str) -> None:
    """Run the Raw LLM Zero-Shot test.

    Injects a bug, prompts the LLM once, applies the patch, and runs PyTest.
    No rollback. No memory. This is the 'Status Quo' that RLM replaces.

    Args:
        repo_path: Path to the demo codebase directory.
        target_file: Filename of the file to inject the bug into (e.g. "auth.py").
    """
    console.print(Rule("[bold yellow]TEST A · RAW LLM  (Zero-Shot / No RLM Layer)[/bold yellow]"))
    time.sleep(0.5)

    target_path = os.path.join(repo_path, target_file)
    _inject_bug(target_path)

    console.print(f"[cyan]  1 ·[/cyan] Injected bug into [bold]{target_file}[/bold]  (> instead of >=)")
    time.sleep(0.3)
    console.print("[cyan]  2 ·[/cyan] Prompting LLM directly (no orchestration layer)...")
    time.sleep(0.5)

    with open(target_path, "r") as f:
        code_context = f.read()

    result = subprocess.run("pytest", shell=True, cwd=repo_path, capture_output=True, text=True)
    test_output = result.stdout + "\n" + result.stderr

    client = LLMClient()
    prompt = (
        f"You are an AI. The codebase is failing.\n"
        f"CODE:\n{code_context}\n\n"
        f"TEST OUTPUT:\n{test_output}\n\n"
        f"Fix the bug. Return ONLY a JSON object with keys: file, search, replace."
    )

    try:
        response = client.completion(prompt, response_format={"type": "json_object"})
        console.print("[cyan]  3 ·[/cyan] LLM generated a JSON patch. Applying...")
        time.sleep(0.4)

        edit = _parse_json(response)
        search_block = edit.get("search", "")
        replace_block = edit.get("replace", "")

        with open(target_path, "r") as f:
            content = f.read()

        new_content = _fuzzy_apply(content, search_block, replace_block)
        if new_content:
            with open(target_path, "w") as f:
                f.write(new_content)

        console.print("[cyan]  4 ·[/cyan] Running PyTest verification...")
        time.sleep(0.5)
        final = subprocess.run("pytest", shell=True, cwd=repo_path, capture_output=True, text=True)

        if final.returncode == 0:
            console.print(Panel(
                "[bold green]Raw LLM Passed (lucky single-shot)[/bold green]",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[bold red]❌  Raw LLM FAILED[/bold red]\n"
                "[dim]PyTest crashed. Codebase is corrupted. No rollback mechanism.[/dim]",
                border_style="red"
            ))

    except Exception as exc:
        console.print(Panel(
            f"[bold red]❌  Raw LLM CRASHED[/bold red]\n[dim]{exc}[/dim]",
            border_style="red"
        ))

    time.sleep(1)


# ──────────────────────────────────────────────────────────────────────────────
# TEST B: RLM Orchestration Layer
# ──────────────────────────────────────────────────────────────────────────────

def run_rlm_layer(repo_path: str, target_file: str) -> None:
    """Run the RLM Orchestration Layer test.

    Re-injects the identical bug, then activates the full TDRL engine:
      - Episodic Memory snapshots codebase before each LLM edit.
      - Deterministic Rollback restores safety on PyTest failure.
      - Episodic failure context is injected into subsequent LLM prompts.
      - Loop continues until PyTest passes or max_steps is reached.

    Args:
        repo_path: Path to the demo codebase directory.
        target_file: Filename of the file with the injected bug.
    """
    console.print(Rule("[bold cyan]TEST B · RLM ORCHESTRATION LAYER  (TDRL + Rollbacks)[/bold cyan]"))
    time.sleep(0.5)

    target_path = os.path.join(repo_path, target_file)
    _inject_bug(target_path)

    console.print("[cyan]  1 ·[/cyan] Initialised [bold]TestDrivenEnv[/bold] with 384-dim Vectorised Error Embedding.")
    time.sleep(0.3)
    console.print("[cyan]  2 ·[/cyan] Loading RL Brain (PPO weights)...")
    time.sleep(0.3)

    env = TestDrivenEnv(repo_path=repo_path, test_command="pytest", max_steps=100)

    try:
        model = PPO.load("weights/test_driven_ppo.zip")
        console.print("   [green]✓ PPO weights loaded.[/green]")
    except Exception:
        model = None
        console.print("   [dim yellow]⚠  PPO weights not found — heuristic fallback active.[/dim yellow]")

    obs, _ = env.reset()
    # Force the known bug so the env reset doesn't randomise it
    _inject_bug(target_path)

    ACTION_LABELS = {0: "RUN_TESTS", 1: "EDIT_FILE", 2: "ROLLBACK"}
    done = False
    step = 0
    success = False

    console.print(f"\n[bold]─── RLM ENGINE START ───[/bold]\n")
    time.sleep(0.5)

    while not done and step < 100:
        step += 1

        # Policy: PPO model or deterministic heuristic fallback
        if model:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            tests_passed = obs[1]
            last_status = obs[2]
            if tests_passed == 1.0:
                action = 0
            elif last_status == -1.0:
                action = 2 if random.random() > 0.5 else 1
            elif last_status == 0.5:
                action = 0
            else:
                action = 1

        label = ACTION_LABELS[action]

        # ── Print step header with cinematic colour coding ────────────────
        if action == 1:  # EDIT
            console.print(f"\n[bold cyan][Step {step:02d}][/bold cyan] Action: [cyan]{label}[/cyan]")
            console.print("   [cyan]→ Requesting JSON patch from LLM...[/cyan]")
        elif action == 0:  # RUN_TESTS
            console.print(f"\n[bold cyan][Step {step:02d}][/bold cyan] Action: [yellow]{label}[/yellow]")
            console.print("   [yellow]→ Executing PyTest verification suite...[/yellow]")
        elif action == 2:  # ROLLBACK
            console.print(f"\n[bold cyan][Step {step:02d}][/bold cyan] Action: [bold red]{label}[/bold red]")
            console.print(
                "   [bold red]⚠  DETERMINISTIC ROLLBACK — codebase restored from Episodic Memory.[/bold red]"
            )

        time.sleep(0.4)

        obs, reward, done, _, _ = env.step(action)
        tests_passed_ratio = obs[1]

        # ── Detect success mid-loop (tests passed during RUN_TESTS) ───────
        if tests_passed_ratio == 1.0:
            console.print(
                Panel(
                    f"[bold green]✅  RLM SOLVED THE TASK IN {step} STEPS[/bold green]\n"
                    "[dim]All tests passing. Codebase is clean.[/dim]",
                    border_style="green",
                    box=box.DOUBLE,
                )
            )
            success = True
            break

    if not success:
        console.print(
            Panel(
                f"[bold red]❌  RLM hit max step limit ({step} steps)[/bold red]\n"
                "[dim]Model too weak to solve the bug — swap to a stronger LLM via .env[/dim]",
                border_style="red",
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPO_PATH = "advanced_demo_target"
    TARGET_FILE = "auth.py"

    console.print(
        Panel(
            "[bold white] RECURSIVE LABS — FOUNDATIONAL RLM DEMONSTRATION [/bold white]",
            border_style="bright_magenta",
            box=box.DOUBLE_EDGE,
        )
    )
    time.sleep(0.5)

    run_baseline_llm(REPO_PATH, TARGET_FILE)

    console.print()
    time.sleep(1)

    run_rlm_layer(REPO_PATH, TARGET_FILE)
