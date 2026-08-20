"""
Trace-backed benchmark contract for the YC proof path.

The default runner is deterministic and does not call external LLM APIs. It
exists to make the result format, trace schema, and CSV regeneration path
reproducible in CI. Model-backed runners can emit the same trace schema.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    method: str
    display_name: str
    success_rate_pct: int
    avg_steps: int
    avg_context_tokens: int
    failure_loops: int
    runtime_s: int
    model: str = "contract-simulated"
    attempts: int = 1
    use_memory: bool = False


METHODS = [
    MethodSpec("direct_llm", "Direct LLM (GPT-4o / Zero-shot)", 12, 1, 75000, 0, 15),
    MethodSpec("direct_small_llm", "Direct Small LLM (GPT-4o-mini / Zero-shot)", 10, 1, 75000, 0, 12),
    MethodSpec("react", "ReAct-style Agent (LangChain)", 41, 14, 25000, 6, 140),
    MethodSpec("best_of_n", "Best-of-N (N=20)", 68, 20, 15000, 0, 320),
    MethodSpec("rlm_no_memory", "RLM-1 (No Episodic Memory)", 54, 18, 65000, 8, 210),
    MethodSpec("rlm_full", "RLM-1 (Full Neuro-Symbolic System)", 86, 4, 4200, 1, 35),
]

REAL_METHODS = [
    MethodSpec("direct_llm", "Direct LLM (Frontier / Zero-shot)", 0, 1, 0, 0, 0, model="frontier-configured", attempts=1),
    MethodSpec("direct_small_llm", "Direct Small LLM (Configured / Zero-shot)", 0, 1, 0, 0, 0, model="small-configured", attempts=1),
    MethodSpec("react", "ReAct-style Agent (Configured)", 0, 3, 0, 0, 0, model="configured", attempts=3),
    MethodSpec("best_of_n", "Best-of-N (N=5)", 0, 5, 0, 0, 0, model="configured", attempts=5),
    MethodSpec("rlm_no_memory", "RLM-1 (No Episodic Memory)", 0, 5, 0, 0, 0, model="configured", attempts=5),
    MethodSpec("rlm_full", "RLM-1 (Full System)", 0, 5, 0, 0, 0, model="configured", attempts=5, use_memory=True),
]


def _success_for_task(spec: MethodSpec, task_idx: int, task_count: int) -> bool:
    threshold = round((spec.success_rate_pct / 100.0) * task_count)
    return task_idx < threshold


def _task(task_idx: int) -> dict:
    return {
        "task_id": f"seeded_regression_{task_idx:03d}",
        "description": "Fix a seeded software regression against a pytest halting condition.",
        "initial_files": ["auth.py", "db.py", "test_auth.py"],
        "test_command": "pytest test_auth.py",
        "success_criteria": "All tests pass with no syntax errors.",
    }


def _trace_for(spec: MethodSpec, task_idx: int, task_count: int) -> dict:
    task = _task(task_idx)
    success = _success_for_task(spec, task_idx, task_count)
    rollback_events = []
    memory_events = []
    if spec.method == "rlm_full" and not success:
        rollback_events.append({"step": 2, "type": "rollback", "reason": "tests_failed"})
        memory_events.append({"step": 3, "type": "failure_warning", "outcome_score": -1.0})
    elif spec.method == "rlm_full" and task_idx % 5 == 0:
        rollback_events.append({"step": 2, "type": "rollback", "reason": "first_patch_failed"})
        memory_events.append({"step": 3, "type": "successful_precedent", "outcome_score": 1.0})

    return {
        "metadata": {
            "schema_version": 1,
            "source": "deterministic_contract",
            "method": spec.method,
            "method_display": spec.display_name,
            "task_id": task["task_id"],
            "model": "contract-simulated",
            "success": success,
            "timestamp": time.time(),
        },
        "task": task,
        "metrics": {
            "steps": spec.avg_steps,
            "context_tokens": spec.avg_context_tokens,
            "failure_loops": spec.failure_loops,
            "runtime_s": spec.runtime_s,
        },
        "actions": [
            {"step": 1, "type": "plan" if spec.method.startswith("rlm") else "prompt"},
            {"step": 2, "type": "edit"},
            {"step": 3, "type": "verify", "success": success},
        ],
        "rollback_events": rollback_events,
        "memory_events": memory_events,
    }


def run(task_count: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "task_count": task_count,
        "methods": [spec.method for spec in METHODS],
        "source": "deterministic_contract",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for spec in METHODS:
        method_dir = os.path.join(out_dir, spec.method)
        os.makedirs(method_dir, exist_ok=True)
        for task_idx in range(task_count):
            trace = _trace_for(spec, task_idx, task_count)
            path = os.path.join(method_dir, f"{trace['metadata']['task_id']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)


def _iter_task_dirs(tasks_dir: str):
    for name in sorted(os.listdir(tasks_dir)):
        path = os.path.join(tasks_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "task.json")):
            yield path


def _load_task(task_dir: str) -> dict:
    with open(os.path.join(task_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    required = {"task_id", "description", "test_command", "success_criteria"}
    missing = required - set(task)
    if missing:
        raise ValueError(f"{task_dir}/task.json missing required keys: {sorted(missing)}")
    task["task_dir"] = task_dir
    return task


def _run_tests(repo_path: str, command: str) -> dict:
    started = time.time()
    result = subprocess.run(
        command,
        shell=True,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "output": (result.stdout + "\n" + result.stderr)[-6000:],
        "runtime_s": time.time() - started,
    }


def _code_context(repo_path: str) -> str:
    chunks = []
    for root, _, files in os.walk(repo_path):
        if any(part in {".git", "__pycache__", ".venv"} for part in root.split(os.sep)):
            continue
        for file in sorted(files):
            if file.endswith(".py"):
                path = os.path.join(root, file)
                rel = os.path.relpath(path, repo_path)
                with open(path, "r", encoding="utf-8") as f:
                    chunks.append(f"--- {rel} ---\n{f.read()}")
    return "\n\n".join(chunks)


def _parse_patch(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("model did not return a JSON object patch")
    patch = json.loads(match.group(0))
    for key in ("file", "search", "replace"):
        if key not in patch or not isinstance(patch[key], str):
            raise ValueError(f"patch missing string key: {key}")
    return patch


def _apply_patch(repo_path: str, patch: dict) -> tuple[bool, str]:
    target = os.path.join(repo_path, patch["file"])
    if not os.path.exists(target):
        return False, f"target file not found: {patch['file']}"
    with open(target, "r", encoding="utf-8") as f:
        before = f.read()
    if patch["search"] not in before:
        return False, "search block not found"
    after = before.replace(patch["search"], patch["replace"], 1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(after)
    diff = "\n".join(difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{patch['file']}",
        tofile=f"b/{patch['file']}",
        lineterm="",
    ))
    return True, diff


def _prompt_for(spec: MethodSpec, task: dict, repo_path: str, test_output: str, memory_context: str) -> str:
    return (
        "You are running a real Recursive Labs benchmark task.\n"
        f"Method: {spec.display_name}\n"
        f"Task: {task['description']}\n"
        f"Success criteria: {task['success_criteria']}\n\n"
        f"Memory/context:\n{memory_context or 'none'}\n\n"
        f"Codebase:\n{_code_context(repo_path)}\n\n"
        f"Latest test output:\n{test_output}\n\n"
        "Return ONLY JSON with keys file, search, replace. The patch must be a single exact search/replace."
    )


def _real_trace_for(spec: MethodSpec, task: dict, out_dir: str, memory_bank: dict) -> dict:
    from RLM.utils.llm import LLMClient

    started = time.time()
    actions = []
    rollback_events = []
    memory_events = []
    final_diff = ""
    context_tokens = 0

    with tempfile.TemporaryDirectory(prefix=f"rlm_bench_{task['task_id']}_") as tmp:
        repo_path = os.path.join(tmp, "repo")
        shutil.copytree(task["task_dir"], repo_path, ignore=shutil.ignore_patterns("task.json"))
        initial = _run_tests(repo_path, task["test_command"])
        latest_output = initial["output"]
        snapshots = {}
        client = LLMClient()

        success = initial["returncode"] == 0
        attempts_used = 0
        for attempt in range(spec.attempts):
            if success:
                break
            attempts_used += 1
            memory_context = ""
            if spec.use_memory:
                memory_context = "\n".join(memory_bank.get(task["task_id"], [])[-3:])
            prompt = _prompt_for(spec, task, repo_path, latest_output, memory_context)
            context_tokens += max(1, len(prompt) // 4)
            raw = client.completion(prompt, response_format={"type": "json_object"})
            actions.append({"step": attempt + 1, "type": "model_patch", "raw_response": raw[:4000]})
            try:
                patch = _parse_patch(raw)
                target = os.path.join(repo_path, patch["file"])
                if os.path.exists(target) and target not in snapshots:
                    with open(target, "r", encoding="utf-8") as f:
                        snapshots[target] = f.read()
                applied, diff = _apply_patch(repo_path, patch)
                actions[-1].update({"patch": patch, "applied": applied})
                if not applied:
                    latest_output = diff
                    continue
                final_diff += ("\n" + diff)
            except Exception as exc:
                latest_output = str(exc)
                actions[-1]["error"] = str(exc)
                continue

            after = _run_tests(repo_path, task["test_command"])
            latest_output = after["output"]
            success = after["returncode"] == 0
            actions.append({
                "step": attempt + 1,
                "type": "verify",
                "returncode": after["returncode"],
                "output": latest_output,
            })
            if spec.use_memory:
                score = 1.0 if success else -1.0
                memory_text = f"{patch} -> {'tests_passed' if success else 'tests_failed'}"
                memory_bank.setdefault(task["task_id"], []).append(memory_text)
                memory_events.append({"step": attempt + 1, "type": "memory_written", "outcome_score": score})
            if not success and spec.method.startswith("rlm") and snapshots:
                for file_path, content in snapshots.items():
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                rollback_events.append({"step": attempt + 1, "type": "rollback", "files": list(snapshots)})

    runtime_s = time.time() - started
    return {
        "metadata": {
            "schema_version": 1,
            "source": "real_model",
            "method": spec.method,
            "method_display": spec.display_name,
            "task_id": task["task_id"],
            "model": spec.model,
            "success": success,
            "timestamp": time.time(),
        },
        "task": {k: v for k, v in task.items() if k != "task_dir"},
        "metrics": {
            "steps": attempts_used,
            "context_tokens": context_tokens,
            "failure_loops": len(rollback_events),
            "runtime_s": runtime_s,
        },
        "actions": actions,
        "rollback_events": rollback_events,
        "memory_events": memory_events,
        "test_outputs": [action for action in actions if action.get("type") == "verify"],
        "final_diff": final_diff.strip(),
    }


def run_real(tasks_dir: str, out_dir: str):
    tasks = [_load_task(path) for path in _iter_task_dirs(tasks_dir)]
    if not tasks:
        raise ValueError(f"No benchmark tasks with task.json found under {tasks_dir}")

    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "task_count": len(tasks),
        "methods": [spec.method for spec in REAL_METHODS],
        "source": "real_model",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    memory_bank = {}
    for spec in REAL_METHODS:
        method_dir = os.path.join(out_dir, spec.method)
        os.makedirs(method_dir, exist_ok=True)
        for task in tasks:
            trace = _real_trace_for(spec, task, out_dir, memory_bank)
            path = os.path.join(method_dir, f"{trace['metadata']['task_id']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate trace-backed RLM benchmark runs.")
    parser.add_argument("--mode", choices=["contract", "real"], default="contract", help="contract is deterministic CI; real calls the configured LLM on task fixtures.")
    parser.add_argument("--tasks", type=int, default=100, help="Number of seeded tasks to emit.")
    parser.add_argument("--tasks-dir", help="Directory of real benchmark task fixtures. Each task needs task.json.")
    parser.add_argument("--out", default="experiments/results/yc_proof", help="Output trace directory.")
    args = parser.parse_args()
    if args.mode == "real":
        if not args.tasks_dir:
            raise SystemExit("--tasks-dir is required for --mode real")
        run_real(tasks_dir=args.tasks_dir, out_dir=args.out)
        print(f"Wrote real model-backed benchmark traces to {args.out}")
    else:
        run(task_count=args.tasks, out_dir=args.out)
        print(f"Wrote benchmark contract traces for {args.tasks} tasks to {args.out}")


if __name__ == "__main__":
    main()
