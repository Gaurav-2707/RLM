"""
SWE-bench Lite Evaluation Harness — P0 YC credibility number.

Runs IntegratedRLM against the princeton-nlp/SWE-bench_Lite dataset and measures
FAIL→PASS resolve rate vs a raw LLM baseline on the SAME base model.

The honest framing: we are NOT claiming to beat OpenDevin or SWE-agent at raw
SWE-bench numbers. We measure the delta: IntegratedRLM(model) - raw(model).
That delta is the RLM Runtime value proposition. This is a same-base-model
ablation: the only variable that changes between the two arms is whether the
RLM Runtime wraps the model. Same weights, same API, same prompts where possible.

Usage:
    # Quick smoke test (5 instances)
    uv run python -m experiments.swe_bench_eval --limit 5 --model gpt-4o

    # Full Lite eval (300 instances)
    uv run python -m experiments.swe_bench_eval --limit 300 --model gpt-4o \
        --output experiments/results/swe_bench_results.json

    # Baseline only (to establish baseline numbers without burning quota)
    uv run python -m experiments.swe_bench_eval --limit 50 --baseline-only

Output JSON schema:
    {
        "meta": {
            "model": "gpt-4o",
            "limit": 50,
            "timestamp": "...",
            "swe_bench_split": "test",
            "framing": "same-base-model ablation: raw(model) vs RLM(model)"
        },
        "results": [
            {
                "instance_id": "astropy__astropy-1234",
                "repo": "astropy/astropy",
                "baseline_resolved": false,
                "rlm_resolved": true,
                "baseline_patch": "...",
                "rlm_patch": "...",
                "baseline_steps": 1,
                "rlm_steps": 7,
                "rlm_rollbacks": 2,
                "error": null
            },
            ...
        ],
        "summary": {
            "total": 50,
            "baseline_resolved": 8,
            "rlm_resolved": 15,
            "baseline_resolve_rate": 0.16,
            "rlm_resolve_rate": 0.30,
            "delta": 0.14,
            "rlm_only_resolved": 9,   # instances RLM fixed that baseline missed
            "baseline_only_resolved": 2,
        }
    }

Dependencies (install separately — not in core RLM deps):
    uv pip install "RLM[swe-bench]"   # pulls datasets + gitpython
    # OR:  uv pip install datasets gitpython

Architecture notes:
    - Each instance runs in a fresh tempdir (no state leakage)
    - Baseline: raw LLM completion with the issue description as prompt
    - RLM:      IntegratedRLM with enable_acc=True, enable_engine=True,
                repo_path=checkout_dir, test_command=instance_test_command
    - Patch extraction: git diff HEAD after RLM finishes
    - Verification: apply patch to fresh checkout, run instance test_cmd
    - Trajectory metadata (steps/rollbacks): recovered from the saved trajectory
      JSON, since IntegratedRLM clears the active trajectory after completion.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Where IntegratedRLM's TrajectoryCollector writes trajectory JSON files.
_TRACES_DIR = os.path.expanduser("~/.rlm/traces")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120,
         input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output, never raising on non-zero exit."""
    return subprocess.run(
        cmd, cwd=cwd, timeout=timeout, input=input_text,
        capture_output=True, text=True, check=False,
    )


def _extract_diff(text: str) -> str:
    """
    Pull a unified diff out of an LLM response.

    Tries, in order:
      1. A ```diff ... ``` or ```patch ... ``` fenced block
      2. Any fenced block that looks like a diff (contains 'diff --git' or '--- ')
      3. A raw diff region starting at the first 'diff --git' / '--- ' line
    Returns "" if nothing diff-like is found.
    """
    if not text:
        return ""

    # 1 & 2: fenced code blocks
    fence_re = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
    for block in fence_re.findall(text):
        if "diff --git" in block or block.lstrip().startswith(("--- ", "+++ ")):
            return block.strip("\n") + "\n"

    # 3: raw diff region
    for marker in ("diff --git", "--- a/", "--- "):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip("\n") + "\n"

    return ""


def _git(cmd: List[str], cwd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return _run(["git"] + cmd, cwd=cwd, timeout=timeout)


def _apply_patch(patch_text: str, repo_dir: str) -> bool:
    """
    Apply a unified diff to repo_dir. Tries `git apply` first (handles git-style
    diffs), then falls back to `patch -p1`. Returns True on success.
    """
    if not patch_text.strip():
        return False
    if not patch_text.endswith("\n"):
        patch_text += "\n"   # git apply rejects patches without a trailing newline
    # Plain `git apply` first (best for git-diff output), then 3-way, then patch.
    # Deliberately avoid `--reject`: it applies partially and would produce
    # false-positive verifications.
    for cmd in (["git", "apply", "-"],
                ["git", "apply", "--3way", "-"],
                ["patch", "-p1", "--forward"]):
        proc = _run(cmd, cwd=repo_dir, input_text=patch_text)
        if proc.returncode == 0:
            return True
    return False


def _latest_trajectory_since(t_start: float) -> Optional[Dict[str, Any]]:
    """
    Find the trajectory JSON written most recently after t_start and load it.
    Used to recover total_steps / total_rollbacks after IntegratedRLM.completion()
    (which clears its in-memory active trajectory).
    """
    try:
        candidates = []
        for path in glob.glob(os.path.join(_TRACES_DIR, "*.json")):
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime >= t_start - 1.0:  # 1s slack for clock granularity
                candidates.append((mtime, path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        with open(candidates[0][1]) as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SWE-bench Lite dataset loader
# ---------------------------------------------------------------------------

def _load_dataset(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    """
    Load SWE-bench Lite from HuggingFace.

    Each instance dict has:
        instance_id, repo, base_commit, problem_statement,
        hints_text, test_patch, patch (gold),
        FAIL_TO_PASS (list of test ids), PASS_TO_PASS (list),
        environment_setup_commit
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets not installed. Run: uv pip install \"RLM[swe-bench]\" "
            "(or: uv pip install datasets gitpython)"
        ) from exc

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    instances = [dict(row) for row in ds]
    if limit:
        instances = instances[:limit]
    return instances


# ---------------------------------------------------------------------------
# Per-instance environment setup
# ---------------------------------------------------------------------------

def _clone_at_base(instance: Dict, dest: str) -> Optional[str]:
    """
    Clone the instance repo into `dest`, checkout base_commit.
    Returns the repo dir path, or None on failure.
    """
    repo = instance["repo"]
    url = f"https://github.com/{repo}.git"
    proc = _run(["git", "clone", "--quiet", url, dest], timeout=600)
    if proc.returncode != 0:
        logger.warning("clone failed for %s: %s", repo, proc.stderr[-300:])
        return None
    co = _git(["checkout", "--quiet", instance["base_commit"]], cwd=dest)
    if co.returncode != 0:
        logger.warning("checkout %s failed: %s", instance["base_commit"], co.stderr[-300:])
        return None
    return dest


def _setup_instance(instance: Dict, workdir: str) -> Optional[str]:
    """
    Clone the repo, checkout base_commit, apply test_patch (installs the
    FAIL_TO_PASS tests). Returns the checked-out repo dir, or None on failure.
    """
    repo_dir = os.path.join(workdir, instance["repo"].replace("/", "__"))
    if _clone_at_base(instance, repo_dir) is None:
        return None
    # Apply the test patch so the failing tests exist in the working tree.
    test_patch = instance.get("test_patch") or ""
    if test_patch.strip():
        if not _apply_patch(test_patch, repo_dir):
            logger.warning("test_patch failed to apply for %s", instance["instance_id"])
            # Not fatal for the baseline arm, but FAIL_TO_PASS may not exist.
    return repo_dir


def _instance_test_command(instance: Dict) -> str:
    """
    Build a pytest command targeting only the FAIL_TO_PASS tests (capped at 10
    nodeids for runtime). Uses `python -m pytest` so it picks up the repo's env.
    """
    fail_to_pass = instance.get("FAIL_TO_PASS") or []
    if isinstance(fail_to_pass, str):
        # Some dataset versions store it as a JSON-encoded string.
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except Exception:
            fail_to_pass = [fail_to_pass]
    tests = " ".join(t for t in fail_to_pass[:10] if t)
    if not tests:
        return "python -m pytest -x --tb=short -q"
    return f"python -m pytest {tests} -x --tb=short -q"


# ---------------------------------------------------------------------------
# Patch verification
# ---------------------------------------------------------------------------

def _verify_patch(instance: Dict, patch: str, workdir: str) -> bool:
    """
    Apply `patch` to a fresh checkout and run the FAIL_TO_PASS tests.

    Steps:
      1. Fresh clone at base_commit
      2. Apply test_patch (so the tests exist)
      3. Apply the candidate patch — if it fails to apply, return False
      4. Run FAIL_TO_PASS tests; return True iff they pass
    """
    if not patch or not patch.strip():
        return False

    verify_dir = os.path.join(workdir, "verify_" + instance["instance_id"].replace("/", "__"))
    if os.path.exists(verify_dir):
        shutil.rmtree(verify_dir, ignore_errors=True)

    if _clone_at_base(instance, verify_dir) is None:
        return False

    # Install the test patch first (the tests must exist before we run them).
    test_patch = instance.get("test_patch") or ""
    if test_patch.strip():
        if not _apply_patch(test_patch, verify_dir):
            logger.debug("verify: test_patch failed to apply for %s", instance["instance_id"])
            return False

    # Apply the candidate fix.
    if not _apply_patch(patch, verify_dir):
        logger.debug("verify: candidate patch failed to apply for %s", instance["instance_id"])
        return False

    test_cmd = _instance_test_command(instance)
    proc = _run(test_cmd.split(), cwd=verify_dir, timeout=300)
    resolved = proc.returncode == 0

    shutil.rmtree(verify_dir, ignore_errors=True)
    return resolved


# ---------------------------------------------------------------------------
# Baseline: raw LLM
# ---------------------------------------------------------------------------

def _build_baseline_prompt(instance: Dict) -> str:
    """Construct the single-shot prompt for the raw-LLM baseline."""
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "") or ""
    prompt = (
        "You are an expert software engineer. Below is a GitHub issue describing "
        "a bug in the repository `" + instance.get("repo", "") + "`.\n\n"
        "Produce a fix as a unified diff (git format). Output ONLY the diff inside "
        "a ```diff fenced code block. Do not include explanation outside the block.\n\n"
        "=== ISSUE ===\n" + problem + "\n"
    )
    if hints.strip():
        prompt += "\n=== HINTS ===\n" + hints + "\n"
    prompt += (
        "\nRespond with a single ```diff code block containing the patch that "
        "resolves the issue.\n"
    )
    return prompt


def _run_baseline(instance: Dict, repo_dir: str, model: str) -> Dict[str, Any]:
    """
    Raw LLM, single shot. Build a prompt, get a diff, verify it.
    Returns {"resolved", "patch", "steps", "error"}.
    """
    from RLM.utils.llm import LLMClient

    try:
        client = LLMClient(model=model)
        prompt = _build_baseline_prompt(instance)
        response = client.completion(prompt)
        patch = _extract_diff(response)
        if not patch:
            return {"resolved": False, "patch": "", "steps": 1, "error": "no_diff_in_response"}
        with tempfile.TemporaryDirectory() as vtmp:
            resolved = _verify_patch(instance, patch, vtmp)
        return {"resolved": resolved, "patch": patch, "steps": 1, "error": None}
    except Exception as exc:
        logger.warning("baseline error on %s: %s", instance["instance_id"], exc)
        return {"resolved": False, "patch": "", "steps": 1, "error": str(exc)}


# ---------------------------------------------------------------------------
# RLM: IntegratedRLM with TDRL
# ---------------------------------------------------------------------------

def _run_rlm(instance: Dict, repo_dir: str, model: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Run IntegratedRLM (ACC + engine + TDRL) on the instance. Extract the git
    diff it produced, verify against FAIL_TO_PASS in a clean checkout.
    Returns {"resolved", "patch", "steps", "rollbacks", "error"}.
    """
    from RLM.integrated_repl import IntegratedRLM

    t_start = time.time()
    try:
        test_cmd = _instance_test_command(instance)
        rlm = IntegratedRLM(
            model=model,
            enable_acc=True,
            enable_engine=True,
            enable_tdrl=True,
            repo_path=repo_dir,
            test_command=test_cmd,
        )
        rlm.completion(context=[], query=instance.get("problem_statement", ""))

        # The RLM edited files in repo_dir; capture the resulting diff.
        diff_proc = _git(["diff", "HEAD"], cwd=repo_dir)
        patch = diff_proc.stdout if diff_proc.returncode == 0 else ""

        resolved = False
        if patch.strip():
            with tempfile.TemporaryDirectory() as vtmp:
                resolved = _verify_patch(instance, patch, vtmp)

        # Recover steps / rollbacks from the saved trajectory JSON.
        steps, rollbacks = 0, 0
        traj = _latest_trajectory_since(t_start)
        if traj:
            steps = int(traj.get("total_steps", 0) or 0)
            rollbacks = int(traj.get("total_rollbacks", 0) or 0)

        return {
            "resolved": resolved, "patch": patch,
            "steps": steps, "rollbacks": rollbacks, "error": None,
        }
    except Exception as exc:
        logger.warning("rlm error on %s: %s", instance["instance_id"], exc)
        return {"resolved": False, "patch": "", "steps": 0, "rollbacks": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _summarize(results: List[Dict], baseline_only: bool) -> Dict[str, Any]:
    total = len(results)
    baseline_resolved = sum(1 for r in results if r.get("baseline_resolved"))
    rlm_resolved = sum(1 for r in results if r.get("rlm_resolved"))
    rlm_only = sum(1 for r in results
                   if r.get("rlm_resolved") and not r.get("baseline_resolved"))
    baseline_only_count = sum(1 for r in results
                              if r.get("baseline_resolved") and not r.get("rlm_resolved"))
    summary = {
        "total": total,
        "baseline_resolved": baseline_resolved,
        "baseline_resolve_rate": (baseline_resolved / total) if total else 0.0,
    }
    if not baseline_only:
        summary.update({
            "rlm_resolved": rlm_resolved,
            "rlm_resolve_rate": (rlm_resolved / total) if total else 0.0,
            "delta": ((rlm_resolved - baseline_resolved) / total) if total else 0.0,
            "rlm_only_resolved": rlm_only,
            "baseline_only_resolved": baseline_only_count,
        })
    return summary


def run_eval(
    limit: Optional[int] = None,
    model: str = "gpt-4o",
    output: str = "experiments/results/swe_bench_results.json",
    baseline_only: bool = False,
    split: str = "test",
    resume: bool = False,
) -> Dict:
    """
    Main evaluation loop. Loads SWE-bench Lite, runs the baseline and (unless
    baseline_only) the RLM arm on each instance, verifies patches, writes JSON.
    """
    instances = _load_dataset(split=split, limit=limit)

    # Resume support: skip instances already present in an existing output file.
    done_ids = set()
    prior_results: List[Dict] = []
    if resume and os.path.exists(output):
        try:
            with open(output) as fh:
                prior = json.load(fh)
            prior_results = prior.get("results", [])
            done_ids = {r["instance_id"] for r in prior_results}
            logger.info("resume: %d instances already evaluated", len(done_ids))
        except Exception:
            logger.warning("resume: could not read %s, starting fresh", output)

    # Progress bar (rich if available, else plain iteration).
    try:
        from rich.progress import track
        iterator = track(instances, description="Evaluating SWE-bench Lite")
    except Exception:
        iterator = instances

    results: List[Dict] = list(prior_results)
    for instance in iterator:
        iid = instance["instance_id"]
        if iid in done_ids:
            continue

        record: Dict[str, Any] = {
            "instance_id": iid,
            "repo": instance.get("repo", ""),
            "baseline_resolved": False,
            "rlm_resolved": False,
            "baseline_patch": "",
            "rlm_patch": "",
            "baseline_steps": 0,
            "rlm_steps": 0,
            "rlm_rollbacks": 0,
            "error": None,
        }

        with tempfile.TemporaryDirectory(prefix="swebench_") as tmp:
            repo_dir = _setup_instance(instance, tmp)
            if repo_dir is None:
                record["error"] = "setup_failed"
                results.append(record)
                _write_output(output, model, limit, split, results, baseline_only)
                continue

            # Baseline arm (runs against a pristine setup tempdir each time).
            base = _run_baseline(instance, repo_dir, model)
            record["baseline_resolved"] = base["resolved"]
            record["baseline_patch"] = base["patch"]
            record["baseline_steps"] = base["steps"]
            if base["error"]:
                record["error"] = f"baseline:{base['error']}"

            # RLM arm — needs a clean checkout (baseline verification used its
            # own tempdirs, but the RLM physically edits repo_dir, so re-setup).
            if not baseline_only:
                # Re-clone into a dedicated subdir so the RLM has a pristine tree
                # (the RLM physically edits files, unlike the baseline arm).
                rlm_dir = os.path.join(tmp, "rlm_" + iid.replace("/", "__"))
                if _clone_at_base(instance, rlm_dir) is not None:
                    tp = instance.get("test_patch") or ""
                    if tp.strip():
                        _apply_patch(tp, rlm_dir)
                    rlm_res = _run_rlm(instance, rlm_dir, model)
                    record["rlm_resolved"] = rlm_res["resolved"]
                    record["rlm_patch"] = rlm_res["patch"]
                    record["rlm_steps"] = rlm_res["steps"]
                    record["rlm_rollbacks"] = rlm_res["rollbacks"]
                    if rlm_res["error"] and not record["error"]:
                        record["error"] = f"rlm:{rlm_res['error']}"
                else:
                    if not record["error"]:
                        record["error"] = "rlm_setup_failed"

        results.append(record)
        # Incremental write so a crash mid-run doesn't lose progress.
        _write_output(output, model, limit, split, results, baseline_only)

    summary = _summarize(results, baseline_only)
    full = _write_output(output, model, limit, split, results, baseline_only)
    return full


def _write_output(output: str, model: str, limit: Optional[int], split: str,
                  results: List[Dict], baseline_only: bool) -> Dict:
    """Assemble the full result dict and write it to `output`. Returns the dict."""
    full = {
        "meta": {
            "model": model,
            "limit": limit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "swe_bench_split": split,
            "framing": "same-base-model ablation: raw(model) vs RLM(model)",
        },
        "results": results,
        "summary": _summarize(results, baseline_only),
    }
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(full, fh, indent=2)
    return full


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(model: str, summary: Dict, output: str, baseline_only: bool) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="SWE-bench Lite — same-base-model ablation")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("Model", model)
        table.add_row("Instances", str(summary.get("total", 0)))
        table.add_row("Baseline resolved",
                      f"{summary.get('baseline_resolved', 0)} "
                      f"({summary.get('baseline_resolve_rate', 0):.1%})")
        if not baseline_only:
            table.add_row("RLM resolved",
                          f"{summary.get('rlm_resolved', 0)} "
                          f"({summary.get('rlm_resolve_rate', 0):.1%})")
            table.add_row("Delta (RLM − base)", f"+{summary.get('delta', 0):.1%}")
            table.add_row("RLM-only solves", str(summary.get("rlm_only_resolved", 0)))
            table.add_row("Baseline-only solves", str(summary.get("baseline_only_resolved", 0)))
        console.print(table)
        console.print(f"[dim]Output: {output}[/dim]")
    except Exception:
        # Plain-text fallback
        print(f"\n{'='*60}\nSWE-bench Lite Results\n{'='*60}")
        print(f"Model:             {model}")
        print(f"Instances:         {summary.get('total', 0)}")
        print(f"Baseline resolved: {summary.get('baseline_resolved', 0)} "
              f"({summary.get('baseline_resolve_rate', 0):.1%})")
        if not baseline_only:
            print(f"RLM resolved:      {summary.get('rlm_resolved', 0)} "
                  f"({summary.get('rlm_resolve_rate', 0):.1%})")
            print(f"Delta (RLM-base):  +{summary.get('delta', 0):.1%}")
        print(f"Output:            {output}")


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite evaluation — RLM vs baseline resolve rate "
                    "(same-base-model ablation)."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max instances to evaluate (default: all 300)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model string (must be in utils/llm.py providers)")
    parser.add_argument("--output", type=str,
                        default="experiments/results/swe_bench_results.json",
                        help="Path for JSON output")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run the baseline (no RLM arm).")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "dev"], help="SWE-bench split")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        results = run_eval(
            limit=args.limit,
            model=args.model,
            output=args.output,
            baseline_only=args.baseline_only,
            split=args.split,
            resume=args.resume,
        )
        _print_summary(args.model, results.get("summary", {}), args.output, args.baseline_only)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results were written incrementally.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
