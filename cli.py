"""
RLM CLI — rlm init | status | run | benchmark | contribute

The entry point for the RLM Runtime hero product.
One command to turn any LLM into a Recursive Language Model.

Usage:
    rlm init                        # scan repo, build graph, start MCP server
    rlm status                      # show memory, graph stats, trajectory count
    rlm run "fix the auth bug"      # run a task with full RLM Runtime
    rlm benchmark                   # score your LLM before and after the runtime
    rlm contribute --enable         # opt into trajectory contribution for training
    rlm contribute --disable
    rlm serve                       # start the FastAPI + MCP server
"""

import argparse
import json
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

console = Console()

CONFIG_PATH = os.path.expanduser("~/.rlm/config.json")
TRACES_DIR  = os.path.expanduser("~/.rlm/traces")


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"contribute_traces": False, "repo_path": None}


def _save_config(cfg: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_init(args):
    """Scan the repo, build the semantic graph, and start the MCP server."""
    repo = args.repo or os.getcwd()
    console.print(Panel(
        f"[bold cyan]RLM Runtime — Initializing[/bold cyan]\n"
        f"Repo: [green]{repo}[/green]",
        title="Recursive Labs"
    ))

    # 1. Build semantic context graph
    console.print("[cyan]Building semantic context graph...[/cyan]")
    try:
        from RLM.memory.graph import SemanticContextGraph
        graph = SemanticContextGraph()
        graph.build_from_directory(repo)
        node_count = graph.graph.number_of_nodes()
        edge_count = graph.graph.number_of_edges()
        graph_path = os.path.join(repo, ".rlm_graph.json")
        graph.save_to_disk(graph_path)
        console.print(f"  [green]✓[/green] Graph: {node_count} nodes, {edge_count} edges → {graph_path}")
    except Exception as e:
        console.print(f"  [red]✗ Graph build failed: {e}[/red]")

    # 2. Persist config
    cfg = _load_config()
    cfg["repo_path"] = repo
    _save_config(cfg)
    console.print(f"  [green]✓[/green] Config saved to {CONFIG_PATH}")

    # 3. Print MCP server instructions
    console.print("\n[bold]To start the MCP server (connect Claude Code / Cursor):[/bold]")
    console.print(f"  [yellow]rlm serve[/yellow]")
    console.print("\n[bold]To run a task:[/bold]")
    console.print(f"  [yellow]rlm run \"describe what to fix\"[/yellow]")
    console.print("\n[dim]RLM Runtime is ready. Any LLM you connect now has memory, rollback, and verified execution.[/dim]")


def cmd_status(args):
    """Show current memory, graph, and trajectory stats."""
    cfg = _load_config()
    repo = cfg.get("repo_path") or os.getcwd()

    table = Table(title="RLM Runtime Status", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Graph
    graph_path = os.path.join(repo, ".rlm_graph.json")
    if os.path.exists(graph_path):
        size_kb = os.path.getsize(graph_path) // 1024
        table.add_row("Context Graph", "✓ Ready", f"{size_kb} KB — {graph_path}")
    else:
        table.add_row("Context Graph", "✗ Not built", "Run: rlm init")

    # Memory
    memory_path = os.path.join(repo, "rlm_memory.json")
    if os.path.exists(memory_path):
        with open(memory_path) as f:
            memories = json.load(f)
        table.add_row("Episodic Memory", "✓ Ready", f"{len(memories)} memories stored")
    else:
        table.add_row("Episodic Memory", "○ Empty", "Will populate on first run")

    # Trajectories
    try:
        from RLM.utils.trajectory import trajectory_stats
        stats = trajectory_stats(TRACES_DIR)
        if stats["total"] > 0:
            rate = f"{stats['success_rate']:.0%}"
            table.add_row(
                "Trajectories",
                "✓ Collecting",
                f"{stats['total']} total · {stats['verified']} verified · {rate} success rate"
            )
        else:
            table.add_row("Trajectories", "○ Empty", "Will collect on first run")
    except Exception:
        table.add_row("Trajectories", "○ Empty", "Will collect on first run")

    # Contribution
    contribute = cfg.get("contribute_traces", False)
    contrib_status = "[green]✓ Enabled[/green]" if contribute else "[dim]Disabled[/dim]"
    table.add_row("Contribute Traces", contrib_status, "rlm contribute --enable to help train RLM-0")

    # LLM provider
    from RLM.utils.llm import DEFAULT_PROVIDER, DEFAULT_MODEL
    table.add_row("LLM Provider", "✓ Detected", f"{DEFAULT_PROVIDER} / {DEFAULT_MODEL}")

    console.print(table)


def cmd_run(args):
    """Run a task with full RLM Runtime on the current repo."""
    task = args.task
    repo = args.repo or _load_config().get("repo_path") or os.getcwd()
    test_cmd = args.test_command
    cfg = _load_config()

    console.print(Panel(
        f"[bold cyan]RLM Runtime — Executing Task[/bold cyan]\n"
        f"Task: [yellow]{task}[/yellow]\n"
        f"Repo: [green]{repo}[/green]\n"
        f"Tests: [green]{test_cmd or 'none'}[/green]",
        title="Recursive Labs"
    ))

    enable_tdrl = bool(test_cmd)

    try:
        from RLM.integrated_repl import IntegratedRLM

        rlm = IntegratedRLM(
            enable_acc=True,
            enable_memory=True,
            enable_engine=True,
            enable_tdrl=enable_tdrl,
            repo_path=repo if enable_tdrl else None,
            test_command=test_cmd,
            contribute_traces=cfg.get("contribute_traces", False),
        )

        # Build context from repo
        context = _build_repo_context(repo)

        console.print("[cyan]Running RLM Runtime...[/cyan]")
        start = time.time()
        answer = rlm.completion(context=context, query=task)
        elapsed = time.time() - start

        console.print(Panel(
            f"[bold green]Result[/bold green]\n{answer}\n\n"
            f"[dim]Runtime: {elapsed:.1f}s[/dim]",
            title="RLM Output"
        ))

    except Exception as e:
        console.print(f"[red]✗ Execution failed: {e}[/red]")
        raise


def cmd_benchmark(args):
    """Score your current LLM with and without RLM Runtime on 5 quick tasks."""
    console.print(Panel(
        "[bold cyan]RLM Capability Scoring[/bold cyan]\n"
        "Comparing your LLM alone vs. LLM + RLM Runtime\n"
        "[dim]Running 5 standardized agentic tasks...[/dim]",
        title="Recursive Labs Benchmark"
    ))

    # Inline task suite — fixed, reproducible, no external deps
    TASKS = [
        {
            "id": "off_by_one",
            "description": "Fix the off-by-one error in this sort function",
            "context": "def sort_list(lst):\n    for i in range(len(lst)):\n        for j in range(i, len(lst)):\n            if lst[i] > lst[j]:\n                lst[i], lst[j] = lst[j], lst[i]\n    return lst",
            "expected": "correct sort",
        },
        {
            "id": "null_check",
            "description": "Add null check to prevent AttributeError on None input",
            "context": "def get_user_name(user):\n    return user.name.upper()",
            "expected": "null guard",
        },
        {
            "id": "complexity",
            "description": "What is the time complexity of this function and why?",
            "context": "def find_duplicates(lst):\n    result = []\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] == lst[j] and lst[i] not in result:\n                result.append(lst[i])\n    return result",
            "expected": "O(n^2) or O(n3)",
        },
        {
            "id": "refactor",
            "description": "Refactor this to use a dictionary for O(n) lookup instead of O(n^2)",
            "context": "def has_pair_with_sum(lst, target):\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] + lst[j] == target:\n                return True\n    return False",
            "expected": "hash set",
        },
        {
            "id": "bug_trace",
            "description": "This function returns wrong results for empty input. Find and fix the bug.",
            "context": "def average(numbers):\n    return sum(numbers) / len(numbers)",
            "expected": "division by zero guard",
        },
    ]

    from RLM.utils.llm import LLMClient, DEFAULT_PROVIDER, DEFAULT_MODEL

    table = Table(title=f"Benchmark: {DEFAULT_PROVIDER}/{DEFAULT_MODEL}", show_header=True, header_style="bold")
    table.add_column("Task", style="cyan", width=16)
    table.add_column("Raw LLM", justify="center")
    table.add_column("+ RLM Runtime", justify="center")

    baseline_score = 0
    rlm_score = 0

    for task in TASKS:
        # Raw LLM call
        try:
            llm = LLMClient()
            raw_answer = llm.completion(
                f"Task: {task['description']}\n\nCode:\n{task['context']}\n\nProvide a brief answer."
            )
            raw_pass = task["expected"].lower() in raw_answer.lower()
            baseline_score += int(raw_pass)
        except Exception:
            raw_pass = False

        # RLM Runtime call
        try:
            from RLM.integrated_repl import IntegratedRLM
            rlm = IntegratedRLM(enable_acc=True, enable_memory=False, enable_engine=True)
            rlm_answer = rlm.completion(context=task["context"], query=task["description"])
            rlm_pass = task["expected"].lower() in rlm_answer.lower()
            rlm_score += int(rlm_pass)
        except Exception:
            rlm_pass = False

        raw_cell  = "[green]PASS[/green]" if raw_pass  else "[red]FAIL[/red]"
        rlm_cell  = "[green]PASS[/green]" if rlm_pass  else "[red]FAIL[/red]"
        table.add_row(task["id"], raw_cell, rlm_cell)

    n = len(TASKS)
    table.add_row(
        "[bold]SCORE[/bold]",
        f"[bold]{baseline_score}/{n}[/bold]",
        f"[bold]{rlm_score}/{n}[/bold]",
    )

    console.print(table)
    uplift = rlm_score - baseline_score
    if uplift > 0:
        console.print(f"\n[bold green]+{uplift} tasks solved by RLM Runtime that raw LLM missed.[/bold green]")
        console.print("[dim]The runtime is working. Every task the wrapper solves is a training example for RLM-0.[/dim]")
    else:
        console.print("\n[yellow]No uplift detected on this task suite. Try with enable_engine=True or a more complex benchmark.[/yellow]")


def cmd_contribute(args):
    """Enable or disable trajectory contribution for RLM-0 training."""
    cfg = _load_config()
    if args.enable:
        cfg["contribute_traces"] = True
        _save_config(cfg)
        console.print(Panel(
            "[bold green]Trajectory contribution ENABLED[/bold green]\n\n"
            "Every verified RLM execution will be anonymized and contributed\n"
            "to the RLM-0 training corpus.\n\n"
            "[dim]• API keys, emails, and internal URLs are stripped automatically\n"
            "• Only trajectories where tests verify the outcome are sent\n"
            "• You can disable at any time with: rlm contribute --disable[/dim]",
            title="Thank you for training RLM-0"
        ))
    elif args.disable:
        cfg["contribute_traces"] = False
        _save_config(cfg)
        console.print("[yellow]Trajectory contribution disabled. Traces are still saved locally.[/yellow]")
    else:
        status = "[green]enabled[/green]" if cfg.get("contribute_traces") else "[dim]disabled[/dim]"
        console.print(f"Contribution status: {status}")
        console.print("Use --enable or --disable to change.")


def cmd_serve(args):
    """Start the RLM FastAPI + MCP server."""
    console.print("[cyan]Starting RLM Runtime server...[/cyan]")
    console.print("  API:  http://127.0.0.1:8000")
    console.print("  MCP:  stdio (for Claude Code / Cursor)")
    console.print("  Docs: http://127.0.0.1:8000/docs")
    console.print("\n[dim]Press Ctrl+C to stop.[/dim]\n")
    try:
        import uvicorn
        uvicorn.run("RLM.api.main:app", host="127.0.0.1", port=8000, reload=True)
    except ImportError:
        console.print("[red]uvicorn not installed. Run: uv pip install uvicorn[/red]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_repo_context(repo_path: str, max_chars: int = 8000) -> str:
    """Read Python source files from repo up to max_chars for context."""
    parts = []
    total = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__", ".git", "node_modules"}]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, repo_path)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    src = f.read()
                chunk = f"\n# {rel}\n{src}\n"
                if total + len(chunk) > max_chars:
                    break
                parts.append(chunk)
                total += len(chunk)
            except Exception:
                pass
    return "".join(parts) or "No source files found."


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="RLM Runtime — turn any LLM into a Recursive Language Model",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Scan repo, build graph, prepare runtime")
    p_init.add_argument("--repo", default=None, help="Repo path (default: cwd)")
    p_init.set_defaults(func=cmd_init)

    # status
    p_status = sub.add_parser("status", help="Show memory, graph, and trajectory stats")
    p_status.set_defaults(func=cmd_status)

    # run
    p_run = sub.add_parser("run", help="Run a task with full RLM Runtime")
    p_run.add_argument("task", help="Task description")
    p_run.add_argument("--repo", default=None)
    p_run.add_argument("--test-command", default=None, help="Test command to verify edits (enables TDRL)")
    p_run.set_defaults(func=cmd_run)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Score LLM before and after RLM Runtime")
    p_bench.set_defaults(func=cmd_benchmark)

    # contribute
    p_contrib = sub.add_parser("contribute", help="Opt in/out of trajectory contribution")
    p_contrib.add_argument("--enable", action="store_true")
    p_contrib.add_argument("--disable", action="store_true")
    p_contrib.set_defaults(func=cmd_contribute)

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI + MCP server")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
