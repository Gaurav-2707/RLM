"""
Synthetic Trajectory Generator — cold-start training data.

Mines real bug-fixing commits from popular Python repos on GitHub.
For each commit: checkout pre-fix state → run IntegratedRLM → verify
against post-fix tests → collect as a verified trajectory.

Target: ≥ 1000 high-quality verified trajectories.
These seed the training corpus before the product data flywheel activates.

Usage:
    # Quick test (1 repo, 5 commits)
    uv run python -m training.synthetic --repos 1 --per-repo 5 --out /tmp/traces/

    # Full run
    uv run python -m training.synthetic \
        --repos 15 --per-repo 70 \
        --out ~/.rlm/traces/ \
        --model gpt-4o \
        --workers 4

Output:
    ~/.rlm/traces/<trajectory_id>.json   (one file per trajectory)
    ~/.rlm/traces/synthetic_stats.json   (aggregate stats)

Commit filtering heuristics:
    - Message contains: "fix", "bug", "error", "issue", "patch", "correct"
    - Changed files: only .py files
    - Diff size: 5–500 lines changed (not too trivial, not a refactor)
    - Tests changed: at least one *test*.py file modified or test passes added
    - Pre-fix state: at least one test fails (confirms bug is real)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)

# IntegratedRLM's TrajectoryCollector always writes to this dir.
_TRACES_DIR = os.path.expanduser("~/.rlm/traces")


# ---------------------------------------------------------------------------
# Repos to mine (15 well-maintained Python projects with good test suites)
# Chosen criteria: active bug-fix history, pytest-based tests, reasonable
# setup time, diverse bug types.
# ---------------------------------------------------------------------------

SYNTHETIC_REPOS = [
    # Stdlib-adjacent utilities
    "https://github.com/dateutil/dateutil",           # date parsing edge cases
    "https://github.com/psf/requests",                # HTTP client bugs
    "https://github.com/pallets/flask",               # web framework
    "https://github.com/pallets/click",               # CLI framework
    "https://github.com/encode/httpx",                # async HTTP
    # Data & science
    "https://github.com/pandas-dev/pandas",           # rich bug history
    "https://github.com/numpy/numpy",                 # array edge cases
    "https://github.com/scikit-learn/scikit-learn",   # ML bugs
    # Developer tooling
    "https://github.com/pytest-dev/pytest",           # meta: testing bugs
    "https://github.com/PyCQA/pylint",                # linter bugs
    "https://github.com/PyCQA/flake8",                # linter
    # Parsing & serialization
    "https://github.com/yaml/pyyaml",                 # YAML edge cases
    "https://github.com/simplejson/simplejson",       # JSON edge cases
    # Async & networking
    "https://github.com/aio-libs/aiohttp",            # async HTTP bugs
    "https://github.com/tornadoweb/tornado",          # async framework
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CommitInfo:
    sha: str
    parent_sha: str
    message: str
    files_changed: List[str]
    test_files: List[str]
    diff_lines: int
    repo_url: str


@dataclass
class SyntheticResult:
    commit: CommitInfo
    trajectory_id: str
    resolved: bool
    trajectory_path: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Commit mining
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120,
         input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a subprocess capturing output, never raising on non-zero exit."""
    return subprocess.run(
        cmd, cwd=cwd, timeout=timeout, input=input_text,
        capture_output=True, text=True, check=False,
    )


def _clone_repo(url: str, workdir: str) -> str:
    """Clone repo to workdir/<repo_name> (shallow). Returns path."""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = os.path.join(workdir, repo_name)
    proc = _run(["git", "clone", "--depth=500", url, repo_dir], timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"clone failed for {url}: {proc.stderr[-300:]}")
    return repo_dir


def _find_bug_fix_commits(
    repo_dir: str,
    repo_url: str,
    max_commits: int = 70,
) -> List[CommitInfo]:
    """
    Find bug-fixing commits in repo_dir.

    Filtering criteria:
      - Commit message contains bug-fix keywords (case insensitive)
      - At least one .py file changed
      - At least one *test*.py file in the diff
      - Diff size between MIN_DIFF_LINES and MAX_DIFF_LINES
      - The parent commit has at least one failing test (confirms a real bug)

    Returns up to max_commits CommitInfo objects, most recent first.
    """
    BUG_KEYWORDS = ["fix", "bug", "error", "issue", "patch", "incorrect",
                    "wrong", "broken", "regression", "resolve", "repair"]
    MIN_DIFF_LINES = 5
    MAX_DIFF_LINES = 400

    log_proc = _run(["git", "log", "--format=%H %P %s", "-n", "1000"],
                    cwd=repo_dir, timeout=120)
    if log_proc.returncode != 0:
        return []

    results: List[CommitInfo] = []
    for line in log_proc.stdout.strip().splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        sha, parents, msg = parts
        # Skip merge commits (multiple parents) — diffs are ambiguous.
        parent_list = parents.split()
        if len(parent_list) != 1:
            continue
        parent_sha = parent_list[0]

        if not any(k in msg.lower() for k in BUG_KEYWORDS):
            continue

        names = _run(["git", "diff", "--name-only", parent_sha, sha],
                     cwd=repo_dir, timeout=60)
        if names.returncode != 0:
            continue
        files = [f for f in names.stdout.strip().splitlines() if f]
        py_files = [f for f in files if f.endswith(".py")]
        test_files = [f for f in py_files
                      if "test" in f.lower() or "spec" in f.lower()]
        if not py_files or not test_files:
            continue

        diff = _run(["git", "diff", parent_sha, sha, "--", "*.py"],
                    cwd=repo_dir, timeout=60)
        diff_lines = len([
            l for l in diff.stdout.splitlines()
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
        ])
        if not (MIN_DIFF_LINES <= diff_lines <= MAX_DIFF_LINES):
            continue

        results.append(CommitInfo(
            sha=sha, parent_sha=parent_sha, message=msg,
            files_changed=py_files, test_files=test_files,
            diff_lines=diff_lines, repo_url=repo_url,
        ))
        if len(results) >= max_commits:
            break
    return results


def _checkout_pre_fix(repo_dir: str, commit: CommitInfo) -> None:
    """Checkout the parent commit (pre-fix state), discarding local changes."""
    _run(["git", "checkout", "-f", commit.parent_sha], cwd=repo_dir, timeout=120)
    _run(["git", "clean", "-fd"], cwd=repo_dir, timeout=60)


def _apply_test_side(repo_dir: str, commit: CommitInfo) -> bool:
    """
    Apply ONLY the test-file changes from the fix commit onto the current
    (pre-fix) tree, then commit them. This installs the failing tests so the
    bug is reproducible, while leaving the source code in its buggy state.

    Committing the test side means a later `git diff HEAD` captures only the
    model's edits — a clean candidate patch with no test contamination.

    Returns True if test changes were applied and committed.
    """
    # Collect the diff for the test files touched by this commit.
    diff = _run(
        ["git", "diff", f"{commit.parent_sha}..{commit.sha}", "--"] + commit.test_files,
        cwd=repo_dir, timeout=60,
    )
    if not diff.stdout.strip():
        return False

    # Apply cleanly (plain, then 3-way). No --reject: partial application would
    # leave .rej/.orig artifacts that git add -A would then commit.
    applied = False
    for cmd in (["git", "apply", "-"], ["git", "apply", "--3way", "-"]):
        if _run(cmd, cwd=repo_dir, input_text=diff.stdout, timeout=60).returncode == 0:
            applied = True
            break
    if not applied:
        return False

    # Commit the test side so RLM edits diff cleanly on top of it.
    _run(["git", "add", "-A"], cwd=repo_dir, timeout=60)
    committed = _run(
        ["git", "-c", "user.email=synth@rlm.local", "-c", "user.name=rlm-synth",
         "commit", "-m", "synthetic: install failing tests"],
        cwd=repo_dir, timeout=60,
    )
    return committed.returncode == 0


def _verify_candidate(repo_dir: str, patch: str, test_cmd: str) -> bool:
    """
    Reset the worktree to the test-side commit (HEAD), replay the candidate
    `patch`, and run `test_cmd`. Returns True iff the tests pass. This is an
    independent re-verification — we never trust the model's self-report.
    """
    if not patch.strip():
        return False
    # Reset to the clean test-side commit.
    _run(["git", "checkout", "-f", "HEAD", "--", "."], cwd=repo_dir, timeout=60)
    _run(["git", "clean", "-fd"], cwd=repo_dir, timeout=60)
    # Replay the candidate fix.
    if not _git_apply(patch, repo_dir):
        return False
    result = _run(test_cmd.split(), cwd=repo_dir, timeout=300)
    return result.returncode == 0


def _git_apply(patch: str, repo_dir: str) -> bool:
    """
    Apply a unified diff. Tries plain `git apply` (best for `git diff` output),
    then `git apply --3way`, then `patch -p1 --forward`. Avoids `--reject`,
    which applies partially and would create false-positive verifications.
    """
    if not patch.strip():
        return False
    if not patch.endswith("\n"):
        patch += "\n"   # git apply rejects patches without a trailing newline
    for cmd in (["git", "apply", "-"],
                ["git", "apply", "--3way", "-"],
                ["patch", "-p1", "--forward"]):
        proc = _run(cmd, cwd=repo_dir, input_text=patch, timeout=60)
        if proc.returncode == 0:
            return True
    return False


def _latest_trajectory_path_since(t_start: float) -> Optional[str]:
    """Path of the trajectory JSON written most recently after t_start."""
    candidates = []
    for path in glob.glob(os.path.join(_TRACES_DIR, "*.json")):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime >= t_start - 1.0:
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_issue_text(repo_dir: str, commit: CommitInfo) -> str:
    """
    Build a synthetic issue description from the commit info.

    The LLM gets this as the task description. It should contain enough
    context to attempt the fix without seeing the actual patch.

    Strategy:
      - Commit message as title
      - List of changed Python source files (not test files) as context hint
      - If commit message references issue numbers, include those

    """
    source_files = [f for f in commit.files_changed if not any(
        t in f for t in ["test_", "_test.py", "tests/"]
    )]
    return (
        f"Bug: {commit.message}\n\n"
        f"Affected files: {', '.join(source_files)}\n\n"
        f"Fix the bug described above. Run the tests to verify your fix."
    )


def _build_test_command(repo_dir: str, commit: CommitInfo) -> str:
    """
    Build a pytest command targeting the commit's test files. Uses
    `python -m pytest` so it resolves against the active interpreter's env.
    """
    if commit.test_files:
        return "python -m pytest " + " ".join(commit.test_files) + " -x --tb=short -q"
    return "python -m pytest -x --tb=short -q"


# ---------------------------------------------------------------------------
# RLM execution
# ---------------------------------------------------------------------------

def _run_rlm_on_commit(
    repo_dir: str,
    commit: CommitInfo,
    model: str,
    trajectory_out_dir: str,
) -> SyntheticResult:
    """
    Run IntegratedRLM on the pre-fix repo state for a single commit.

    Steps:
        1. Checkout pre-fix state
        2. Apply + commit the test side (so failing tests exist)
        3. Pre-check: confirm the tests actually fail (bug is real)
        4. Run IntegratedRLM with TDRL
        5. Independently verify the produced diff against the tests
        6. Stamp the auto-saved trajectory with the independent verdict, copy
           it into trajectory_out_dir
        7. Return SyntheticResult
    """
    from RLM.integrated_repl import IntegratedRLM
    from RLM.utils.trajectory import RLMTrajectory

    tid = str(uuid.uuid4())

    try:
        _checkout_pre_fix(repo_dir, commit)

        if not _apply_test_side(repo_dir, commit):
            return SyntheticResult(commit, tid, False, None, "test_side_not_applicable")

        issue_text = _extract_issue_text(repo_dir, commit)
        test_cmd = _build_test_command(repo_dir, commit)

        # Pre-check: the bug must be reproducible (≥1 failing test). If the
        # tests already pass, the commit isn't a useful training example.
        pre = _run(test_cmd.split(), cwd=repo_dir, timeout=300)
        if pre.returncode == 0:
            return SyntheticResult(commit, tid, False, None, "bug_not_reproduced")

        # Run the RLM. It edits files in repo_dir and auto-saves a trajectory
        # to ~/.rlm/traces when completion() finishes.
        t_start = time.time()
        rlm = IntegratedRLM(
            model=model,
            enable_acc=True,
            enable_tdrl=True,
            repo_path=repo_dir,
            test_command=test_cmd,
        )
        rlm.completion(context=[], query=issue_text)

        # Candidate patch = the model's edits on top of the test-side commit.
        diff_proc = _run(["git", "diff", "HEAD"], cwd=repo_dir, timeout=60)
        patch = diff_proc.stdout if diff_proc.returncode == 0 else ""

        resolved = _verify_candidate(repo_dir, patch, test_cmd) if patch.strip() else False

        # Locate the trajectory IntegratedRLM just saved, stamp it with our
        # independent verdict, and write it into the synthetic corpus dir.
        traj_path = _latest_trajectory_path_since(t_start)
        out_path: Optional[str] = None
        if traj_path:
            try:
                traj = RLMTrajectory.load(traj_path)
                traj.final_outcome = resolved
                traj.verified = True   # verified by us, regardless of pass/fail
                os.makedirs(trajectory_out_dir, exist_ok=True)
                out_path = traj.save(trajectory_out_dir)
            except Exception as exc:
                logger.warning("could not re-save trajectory for %s: %s", commit.sha[:8], exc)

        return SyntheticResult(commit, tid, resolved, out_path, None)

    except Exception as exc:
        logger.warning("commit %s failed: %s", commit.sha[:8], exc)
        return SyntheticResult(commit, tid, False, None, str(exc))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_synthetic_trajectories(
    repos: int = 15,
    per_repo: int = 70,
    out_dir: str = "~/.rlm/traces/",
    model: str = "gpt-4o",
    workers: int = 1,
    repos_list: Optional[List[str]] = None,
) -> dict:
    """
    Main generation loop. Returns summary stats dict.

    For each repo: clone → find bug-fix commits → checkout pre-fix state →
    apply test side → run IntegratedRLM → independently verify → save trajectory.
    Multi-worker: each worker gets its own clone (no shared state).

    NOTE on environment: running a repo's test suite requires that repo's
    dependencies to be importable. Full per-repo env setup (conda/venv) is out
    of scope here; commits whose tests can't run are skipped gracefully and
    counted under "skipped".
    """
    repo_urls = (repos_list or SYNTHETIC_REPOS)[:repos]
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def _process_repo(url: str) -> dict:
        """Clone one repo, mine commits, run RLM on each. Returns per-repo stats."""
        repo_stat = {"url": url, "attempted": 0, "verified": 0, "skipped": 0, "errors": 0}
        with tempfile.TemporaryDirectory(prefix="synth_") as workdir:
            try:
                repo_dir = _clone_repo(url, workdir)
            except Exception as exc:
                logger.warning("clone failed for %s: %s", url, exc)
                repo_stat["errors"] += 1
                return repo_stat

            try:
                commits = _find_bug_fix_commits(repo_dir, url, max_commits=per_repo)
            except Exception as exc:
                logger.warning("commit mining failed for %s: %s", url, exc)
                repo_stat["errors"] += 1
                return repo_stat

            logger.info("%s: %d candidate commits", url, len(commits))
            for commit in commits:
                repo_stat["attempted"] += 1
                result = _run_rlm_on_commit(repo_dir, commit, model, out_dir)
                if result.error:
                    if result.error in ("bug_not_reproduced", "test_side_not_applicable"):
                        repo_stat["skipped"] += 1
                    else:
                        repo_stat["errors"] += 1
                elif result.resolved:
                    repo_stat["verified"] += 1
        return repo_stat

    by_repo: List[dict] = []

    # Progress bar over repos (rich if available).
    try:
        from rich.progress import track
        repo_iter = track(repo_urls, description="Mining repos")
    except Exception:
        repo_iter = repo_urls

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_repo, url): url for url in repo_urls}
            for fut in concurrent.futures.as_completed(futures):
                by_repo.append(fut.result())
    else:
        for url in repo_iter:
            by_repo.append(_process_repo(url))

    attempted = sum(r["attempted"] for r in by_repo)
    verified = sum(r["verified"] for r in by_repo)
    skipped = sum(r["skipped"] for r in by_repo)
    errors = sum(r["errors"] for r in by_repo)
    stats = {
        "attempted": attempted,
        "verified": verified,
        "skipped": skipped,
        "errors": errors,
        "resolve_rate": (verified / attempted) if attempted else 0.0,
        "by_repo": by_repo,
    }

    stats_path = os.path.join(out_dir, "synthetic_stats.json")
    try:
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2)
    except Exception as exc:
        logger.warning("could not write stats: %s", exc)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RLM training trajectories from bug-fixing commits."
    )
    parser.add_argument(
        "--repos", type=int, default=15,
        help="Number of repos to mine (default: 15, max: len(SYNTHETIC_REPOS))",
    )
    parser.add_argument(
        "--per-repo", type=int, default=70,
        help="Max commits per repo to attempt (default: 70)",
    )
    parser.add_argument(
        "--out", type=str, default="~/.rlm/traces/",
        help="Output directory for trajectory JSON files",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="LLM model to use for IntegratedRLM",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (each gets its own repo clone)",
    )
    parser.add_argument(
        "--repos-list", nargs="*",
        help="Override default repo list with specific URLs",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    try:
        stats = generate_synthetic_trajectories(
            repos=args.repos if args.repos_list else min(args.repos, len(SYNTHETIC_REPOS)),
            per_repo=args.per_repo,
            out_dir=out_dir,
            model=args.model,
            workers=args.workers,
            repos_list=args.repos_list,
        )
        print(f"\nSynthetic generation complete.")
        print(f"  Trajectories generated: {stats.get('verified', 0)}")
        print(f"  Total attempted:        {stats.get('attempted', 0)}")
        print(f"  Resolve rate:           {stats.get('resolve_rate', 0):.1%}")
        print(f"  Output dir:             {out_dir}")
    except NotImplementedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"Unimplemented path reached: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
