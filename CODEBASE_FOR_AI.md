# RLM Codebase Brief — For AI Assistants

This document is a compressed map of the entire codebase. Read this instead
of exploring files individually. Everything here is accurate as of the latest
commit. File paths are relative to the project root.

---

## Package Structure

The project root (`/path/to/RLM/`) IS the `RLM` Python package — it has
`__init__.py` at the root. Import as `from RLM.xxx import ...`. The venv
is at `.venv/` and uses uv. A `sitecustomize.py` in the venv's site-packages
adds the parent directory to sys.path so `import RLM` resolves correctly.

```
RLM/                        ← Python package root (also project root)
├── __init__.py             ← exports: from .rlm import RLM
├── cli.py                  ← `rlm` CLI entry point (6 commands)
├── integrated_repl.py      ← IntegratedRLM — the hero product
├── repl.py                 ← base REPLEnv
├── rlm_repl.py             ← RLM_REPL (base class for IntegratedRLM)
├── rlm.py                  ← RLM base class
├── main.py                 ← alternative entry point
├── acc/                    ← Adaptive Compute Controller
├── api/                    ← FastAPI + MCP server
├── baselines/              ← ReAct, Best-of-N for comparison
├── engine/                 ← System2 planner + RLM engine
├── experiments/            ← benchmarks, RL training, analysis scripts
├── logger/                 ← logging utilities
├── memory/                 ← episodic memory + semantic graph
├── tests/                  ← test suite (74 passing)
├── theory/                 ← optimal stopping theory
├── training/               ← QLoRA training pipeline (Opus-designed)
├── utils/                  ← LLM client, trajectory, prompts, tracing
├── weights/                ← trained model weights (DQN, PPO)
├── demo_target/            ← simple Python repo for demo/benchmark
├── advanced_demo_target/   ← complex demo target (auth + DB)
├── docs/                   ← ARCHITECTURE_FAILURE_TAXONOMY.md
└── pyproject.toml          ← build config, deps
```

---

## Key Files — What Each Does

### integrated_repl.py — `IntegratedRLM`
The hero product. Subclasses `RLM_REPL`, wires in ACC + memory + engine.

```python
IntegratedRLM(
    model=None,              # LLM model string e.g. "gpt-4o"
    enable_acc=False,        # Adaptive Compute Controller
    enable_memory=False,     # episodic memory
    enable_engine=False,     # System2 planner
    enable_tdrl=False,       # test-driven RL (needs repo_path + test_command)
    repo_path=None,
    test_command=None,
    contribute_traces=False, # opt-in trajectory upload
    memory_capacity=200,
)
# Key method:
rlm.completion(context: str | list, query: str) -> str
```

TDRL plugins injected into REPL: `edit_file(path, content)`, `run_tests()`,
`rollback(reason)`. These physically write/restore files.

### cli.py — `rlm` command
6 subcommands: `init`, `status`, `run`, `benchmark`, `contribute`, `serve`.
Config persisted to `~/.rlm/config.json`. Trajectories to `~/.rlm/traces/`.

```python
# rlm init  → builds SemanticContextGraph, saves to .rlm_graph.json
# rlm run "task" --test-command "pytest"  → calls IntegratedRLM.completion()
# rlm benchmark  → 5 tasks, raw LLM vs RLM Runtime, rich table output
# rlm contribute --enable  → sets contribute_traces=True in config
# rlm serve  → starts uvicorn on api/main.py:app
```

### acc/ — Adaptive Compute Controller
```
acc/__init__.py       exports: AdaptiveComputeController, ComplexityScorer,
                               DepthRecord, EpisodeReport
acc/controller.py     AdaptiveComputeController — NLI grounding gate,
                        overshoot detection, conformal gate. Also contains
                        RLController (DQN-based, needs weights).
acc/complexity.py     ComplexityScorer.score(query, context) → float [0,1]
                        Heuristic: keyword analysis + structural signals +
                        context length. Used to set ACC episode parameters.
acc/conformal.py      Gibbs-Candès ACI implementation
acc/models.py         DepthRecord, EpisodeReport dataclasses
```

`AdaptiveComputeController.should_exit(answer, context, llm_client, confidence,
  iteration, executed_code) → dict{exit, reason, warning, [rollback, peak_iter]}`

### memory/ — Episodic Memory + Semantic Graph
```
memory/system.py    EpisodicMemorySystem — BM25 + dense retrieval,
                      outcome-scored. retrieve(query, top_k) → (memories, conflicts)
memory/graph.py     SemanticContextGraph — NetworkX DiGraph from Python AST.
                      build_from_directory(path), get_blast_radius(node, depth),
                      save_to_disk(path), load_from_disk(path)
memory/base.py      Base interfaces
memory/retrieval.py Dense retrieval utilities
memory_repl.py      MemoryREPL wrapper (used by IntegratedRLM)
```

### engine/ — System2 Planner
```
engine/rlm_engine.py      RLMEngine — orchestrates Decompose→Refine→Synthesize
engine/planner.py         System2Planner — graph + memory + LLM → JSON plan
engine/test_gen.py        TestGenerator — produces verification tests
engine/templates.py       Prompt templates
engine/generic_baseline.py  Vanilla LLM baseline (no recursion)
engine_repl.py            EngineREPL wrapper (used by IntegratedRLM)
```

### utils/ — Shared Utilities
```
utils/llm.py        LLMClient — 9 providers. Auto-selects from env vars.
                     DEFAULT_PROVIDER, DEFAULT_MODEL (auto-detected)
                     LLMClient().completion(prompt_or_messages) → str
                     Providers: openai/anthropic/gemini/mistral/cohere/
                                azure/together/huggingface/ollama

utils/trajectory.py  TrajectoryStep dataclass:
                       (step_number, iteration, action_type, input_state,
                        output_action, outcome, reward, confidence,
                        was_rollback, rollback_reason, timestamp)
                     RLMTrajectory dataclass:
                       (trajectory_id, steps, total_steps, total_rollbacks,
                        final_outcome[bool], verified, final_diff,
                        task_description, model_provider, model_name, ...)
                     TrajectoryCollector: start(), add_step(), finish(success, diff)
                     trajectory_stats(dir) → {total, verified, success_rate}
                     IMPORTANT: final_outcome is a bool (True=success, False=failure)
                                Never pass string "success"/"failure" — bool("failure")==True

utils/upload.py     IMPLEMENTED: upload_trajectory(traj, endpoint) → bool
                                  upload_trajectory_async(traj, endpoint) → None
                                  batch_upload(trajs, endpoint, min_score) → dict
                                  _strip_pii(payload) — regex-based PII redaction
                                  Uses urllib (stdlib), retries with backoff.
                    WIRED: TrajectoryCollector.finish() calls upload_trajectory_async
                           when contribute=True and trajectory.verified=True.
                           Fire-and-forget daemon thread, never blocks the REPL.

utils/prompts.py    DEFAULT_QUERY, next_action_prompt(), build_system_prompt()
utils/tracing.py    TraceStorage — records REPL execution steps
utils/utils.py      convert_context_for_repl(), find_code_blocks(),
                    process_code_execution()
```

### training/ — QLoRA Training Pipeline (complete, Opus-designed)
```
training/__init__.py    exports all public symbols
training/tokens.py      26 control tokens. add_control_tokens(tok, model)
                         mean-inits BOTH embed_tokens AND lm_head.
                         conf_to_token(float) → bucket via CONF_BIN_EDGES
training/render.py      render_trajectory(traj) → RenderedTrajectory
                         Span(text, trainable, role, is_ul)
                         World spans (state, verdict): NOT trainable
                         Decision spans (action, rollback, conf): trainable
                         UL flag: only on final failed action
                         BUG1 FIXED: verdict double-emit (step.outcome truthy)
                         BUG2 FIXED: empty reason blocks on non-reason steps
training/collator.py    encode_trajectory(rt, tok, max_seq_len) → dict
                         Per-span tokenization (no mask drift)
                         Tail-truncation (keeps <final> token)
                         RLMCollator(tok) → batches with tok_weights, ul_mask
training/loss.py        recursion_loss(logits, batch, rb_id, weights) → (loss, metrics)
                         L_total = L_ce + 0.5·L_ul + 0.1·L_rb
                         LossWeights(alpha_ul=0.5, beta_rb=0.1, r_hat=1.0)
training/data.py        load_trajectories(dir), compute_r_hat(trajs)
                         TrajectoryDataset, ReplayMixDataset(4% replay)
training/train.py       train(cfg, eval_fn, probe_fn) — full QLoRA driver
                         build_model_and_tokenizer(cfg) → (model, tok)
                         LoRA: r=32, α=64, dropout=0.05, q/k/v/o + gate/up/down
                         modules_to_save = ["embed_tokens", "lm_head"]
                         Checkpoints on best P3. Forgetting tripwire at >3% drop.
training/eval.py        P1-P4 Recursion Proof protocol
                         ModelRunner (abstract), EvalTask, Rollout dataclasses
                         recursion_proof(runner, tasks) → metrics dict
                         _passes_bar: P3≥0.25, rollback_precision≥0.70,
                                      ECE≤0.10, P1_flip≥0.80
training/run_training.py  LiveModelRunner (drives real model through grammar)
                           verify_fn: apply diff → run tests → bool  [IMPLEMENTED]
                           run_code: tempdir sandbox exec              [IMPLEMENTED]
training/scorer.py      FULLY IMPLEMENTED: TrajectoryScorer.score(traj, corpus) → float
                         filter_corpus(candidates, corpus, min_score=0.4)
                         _score_recursion: 1.0/0.6/0.2 (rollback+win/straight win/fail)
                         _score_difficulty: min(steps, max_steps) / max_steps
                         _score_novelty: 1 - max_cosine_sim vs corpus (TF vectors)
                         _score_efficiency: 1 - steps/max_steps (0 for failures)
                         _score_verification: FAIL→PASS transition count / max_steps
training/synthetic.py   SKELETON EXISTS (NOT implemented yet):
                         generate_synthetic_trajectories(repos, per_repo, out_dir, model)
                         15 repos pre-selected in SYNTHETIC_REPOS list
                         All helper functions stubbed with TODO + detailed specs
                         CLI: python -m training.synthetic --repos 15 --out ~/.rlm/traces/
```

### experiments/ — Benchmarks + RL Training
```
experiments/swe_bench_eval.py   SKELETON EXISTS (NOT implemented yet):
                                  run_eval(limit, model, output, baseline_only)
                                  CLI: python -m experiments.swe_bench_eval --limit 50
                                  Full output JSON schema documented in module docstring
                                  All helper functions stubbed with TODO + detailed specs
experiments/benchmark.py        Benchmarking harness (existing)
experiments/benchmark_suite.py  Task suite definitions (existing)
experiments/summarize_results.py Results aggregation (existing)
experiments/rl/test_driven_env.py  TDRL Gymnasium env:
                                    State=(code,tests,history)
                                    Action=file edit
                                    Reward=test pass rate
experiments/rl/train_test_rl.py    PPO over TDRL env
experiments/rl/offline_env.py      Offline RL environment
experiments/rl/train_rl_controller.py  DQN controller training
experiments/run_*.py               Various sweep/eval scripts
experiments/datasets/              HotpotQA and MATH loaders
experiments/analysis/              Plotting and analysis scripts
```

### api/ — FastAPI + MCP Server
```
api/main.py       FastAPI app — starts with: uvicorn RLM.api.main:app
api/mcp_server.py MCP tool schemas defined (TOOLS list), but NO stdio
                   transport loop — not a working MCP server yet.
                   Tools defined: get_current_context, get_active_file,
                   get_terminal_trace, get_relevant_memories, run_rlm_task
api/agent.py      Agent endpoint (PPO model inference)
api/schemas.py    Pydantic schemas
api/context_os.py context_os object — reads active file, terminal trace
api/sync_broker.py Sync/async bridge for IDE integrations
```

### tests/ — Test Suite (104 passing, 0 failing)
```
tests/test_render.py         ✅ 19 tests — render.py bugs + invariants
tests/test_collator.py       ✅ 18 tests — encode_trajectory + RLMCollator
tests/test_loss.py           ✅ 12 tests — three-term loss with known inputs
tests/test_scorer.py         ✅ 25 tests — all 5 scoring dimensions + combined
tests/test_integrated_repl.py ✅ 16 tests — IntegratedRLM init + completion + TDRL
tests/test_cli.py            ✅ 14 tests — CLI commands with mocked deps
tests/test_graph.py          EXISTS — SemanticContextGraph
tests/test_memory.py         EXISTS — EpisodicMemorySystem
tests/test_mcp_server.py     EXISTS — MCP schema checks
tests/test_api_smoke.py      EXISTS — FastAPI smoke tests
tests/test_planner.py        EXISTS — System2Planner
tests/test_context_os.py     EXISTS — context_os
tests/test_sync.py           EXISTS — sync broker
```

---

## What Is Missing (Needs to Be Built)

### P0 — Zero YC credibility without this
**`experiments/swe_bench_eval.py`** — SKELETON EXISTS. Implement 6 functions:
1. `_load_dataset(split, limit)` — `datasets.load_dataset("princeton-nlp/SWE-bench_Lite")`
2. `_setup_instance(instance, workdir)` — git clone + checkout base_commit + apply test_patch
3. `_instance_test_command(instance)` — pytest command targeting FAIL_TO_PASS tests
4. `_run_baseline(instance, repo_dir, model)` — raw LLM + parse diff + verify
5. `_run_rlm(instance, repo_dir, model)` — IntegratedRLM with TDRL
6. `_verify_patch(instance, patch, workdir)` — fresh clone + apply + run tests
7. `run_eval(...)` — orchestration loop with rich progress bar + JSON output

Full output schema and CLI are already implemented in the skeleton.
Install: `uv pip install datasets gitpython` (not in core deps yet — add to pyproject.toml).

### P1 — Core training infrastructure
**`training/synthetic.py`** — SKELETON EXISTS. Implement 6 functions:
1. `_clone_repo(url, workdir)` — subprocess git clone
2. `_find_bug_fix_commits(repo_dir, repo_url, max_commits)` — git log + heuristic filter
3. `_checkout_pre_fix(repo_dir, commit)` — git checkout parent_sha
4. `_apply_test_side(repo_dir, commit_sha)` — cherry-pick only test file changes
5. `_run_rlm_on_commit(repo_dir, commit, model, out_dir)` — full pipeline
6. `generate_synthetic_trajectories(...)` — orchestration with ThreadPoolExecutor
SYNTHETIC_REPOS (15 repos) and CLI are already defined.

**`utils/trajectory.py`** — one small wiring task:
Wire `upload_trajectory_async` into `TrajectoryCollector.finish()`.
When `contribute_traces=True` stored in config, call it automatically.
Check: does TrajectoryCollector have access to the contribute_traces flag?
If not, add it as a constructor param: `TrajectoryCollector(contribute=False)`.

### P2 — Product completeness
**`api/mcp_server.py`** — ✅ ALREADY FULLY IMPLEMENTED. Has:
- `TOOLS` list (6 tools: get_current_context, get_active_file, get_terminal_trace,
  get_relevant_memories, get_blast_radius, write_agent_memory)
- `call_tool()` → dispatches to context_os
- `handle_request()` → handles initialize / tools/list / tools/call
- `main()` → complete JSON-RPC stdio loop (for line in sys.stdin: ...)
Run as MCP server: `python -m RLM.api.mcp_server`
Claude Code config: `{"command": "python", "args": ["-m", "RLM.api.mcp_server"]}`

**`tests/test_integrated_repl.py`** — ✅ DONE. 16 tests passing.

**`tests/test_cli.py`** — ✅ DONE. 14 tests passing.

### Research Documents (needs literature knowledge — super-model strength)
- `docs/RESEARCH_AUDIT.md` — SWE-agent, CodeAct, OpenDevin, SWE-Llama, Agentless
- `docs/H2_ARCHITECTURE.md` — native RLM architecture design
- `docs/DATA_FLYWHEEL_ANALYSIS.md` — training signal analysis
- `docs/COMPETITIVE_ANALYSIS.md` — honest positioning + moat
- `docs/EXPERIMENTAL_ROADMAP.md` — 6-month experiment plan

---

## Known Design Decisions (Don't Change Without Good Reason)

| Decision | Location | Why |
|----------|----------|-----|
| Per-span tokenization | collator.py | No mask drift vs full-sequence tok |
| Tail truncation (not head) | collator.py | Keeps `<final>` token |
| Mean-init both embed+lm_head | tokens.py | New token rows bootstrap faster |
| 3-term loss weights 0.5/0.1 | loss.py | Tuned for rollback budget |
| LoRA modules_to_save=embed+lm_head | train.py | New token vocab must be saved |
| 4% replay mix | data.py | Forgetting mitigation heuristic |
| Checkpoint on P3 not val_loss | train.py | P3 = feedback_value, our key metric |
| Honest SWE-bench framing | docs/ | Same-base-model ablation only |
| UL on final failed action only | render.py | Not on every failed step |
| final_outcome is bool | trajectory.py | True=success, False=failure. Don't use strings. |
| TF (not TF-IDF) for novelty | scorer.py | Simpler, good enough for corpus filter |
| urllib (not requests) | upload.py | No extra dep — stdlib sufficient |

---

## Environment & Running

```bash
# Install and run
uv pip install -e .
uv run rlm --help
uv run rlm init
uv run rlm run "fix the failing test" --test-command "pytest"
uv run rlm benchmark

# Run tests (74 passing)
uv run pytest tests/test_render.py tests/test_collator.py tests/test_scorer.py tests/test_loss.py -v

# Training (needs GPU)
uv run python -m training.run_training \
    --traj_dir ~/.rlm/traces --out runs/rlm0 \
    --eval_tasks training/eval_tasks.example.json

# SWE-bench eval (skeleton — implement first)
uv run python -m experiments.swe_bench_eval --limit 50 --output results.json

# Synthetic data generation (skeleton — implement first)
uv run python -m training.synthetic --repos 15 --out ~/.rlm/traces/
```

LLM provider auto-detected from env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GENAI_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `AZURE_OPENAI_API_KEY`,
`TOGETHER_API_KEY`, `HF_MODEL_PATH`, `OLLAMA_MODEL` (last fallback).
Current default in `.env`: `OLLAMA_MODEL=ollama/llama3.1`.

---

## Three Horizons (Vision Context)

**H0 (now)**: RLM Runtime — wrapper making any LLM recursive. Product is
`rlm` CLI + IntegratedRLM. Training data collection via `contribute_traces`.

**H1 (6mo)**: RLM-0 — QLoRA fine-tuned Llama 3.1 8B on verified trajectories.
Training pipeline is in `training/`. Needs: trajectory corpus, GPU, scorer ✅ done.

**H2 (18mo)**: Native RLM architecture — recursion baked into weights
structurally. Not designed yet. Needs research work.

**The honest claim**: RLM Runtime + base LLM > same base LLM alone (on
recursive tasks). NOT: RLM-0 8B > frontier models on raw SWE-bench.
The differentiation is structural training signal quality, not raw scale.

---

## Status Summary (as of latest session)

### Completed ✅
- All import paths fixed, package namespace shadow removed, sitecustomize.py
- render.py BUG1 (verdict double-emit) and BUG2 (empty reason blocks) fixed
- test_render.py + test_collator.py: `final_outcome="failure"` → `final_outcome=False`
- ComplexityScorer implemented (acc/complexity.py)
- verify_fn + run_code stubs replaced (training/run_training.py)
- training/scorer.py: all 5 scoring dimensions fully implemented
- utils/upload.py: _strip_pii, upload_trajectory, batch_upload all implemented
- utils/trajectory.py: upload_trajectory_async wired into TrajectoryCollector.finish()
- api/mcp_server.py: complete working MCP stdio server (was already there — confirmed)
- tests: 104 passing, 0 failing
  - test_render.py: 19, test_collator.py: 18, test_scorer.py: 25, test_loss.py: 12
  - test_integrated_repl.py: 16 (new), test_cli.py: 14 (new)
- experiments/swe_bench_eval.py: skeleton with full CLI, output JSON schema, 7 stubs
- training/synthetic.py: skeleton with 15 repos pre-selected, full CLI, 6 stubs
- CODEBASE_FOR_AI.md: this document (kept current)

### For super-model to implement 🎯
1. **[P0] experiments/swe_bench_eval.py** — implement 7 stubbed functions (~300 lines)
   All: _load_dataset, _setup_instance, _instance_test_command, _run_baseline,
        _run_rlm, _verify_patch, run_eval
2. **[P1] training/synthetic.py** — implement 6 stubbed functions (~200 lines)
   All: _clone_repo, _find_bug_fix_commits, _checkout_pre_fix, _apply_test_side,
        _run_rlm_on_commit, generate_synthetic_trajectories
3. **[Research] docs/RESEARCH_AUDIT.md** — needs literature knowledge
4. **[Research] docs/H2_ARCHITECTURE.md** — native architecture design
5. **[Research] docs/DATA_FLYWHEEL_ANALYSIS.md** — training signal analysis
6. **[Research] docs/COMPETITIVE_ANALYSIS.md** — honest moat analysis
7. **[Research] docs/EXPERIMENTAL_ROADMAP.md** — 6-month experiment plan
