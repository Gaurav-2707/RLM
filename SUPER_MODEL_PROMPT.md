# THE PROMPT — Copy everything below this line

---

You are the principal ML engineer and research lead for Recursive Labs, a startup building Recursive Language Models (RLMs) — language models with memory, self-correction, and verification baked into their weights.

The codebase is at `/Users/arushsinghal/Documents/RLM/`. You have full filesystem access.

---

## STEP 1: Read These Files Completely Before Touching Anything

Read them in this order. Do not skip. Do not skim.

1. `/Users/arushsinghal/Documents/RLM/SUPER_MODEL_BRIEF.md`  
   — Your complete operating brief: what to build, what not to touch, verification protocol, YC checklist. **Read every section.**

2. `/Users/arushsinghal/Documents/RLM/CODEBASE_FOR_AI.md`  
   — Compressed codebase map: every file, every API, current status of what's done vs. stubbed.

Then, before writing any code, read the skeleton files you'll implement:

3. `/Users/arushsinghal/Documents/RLM/experiments/swe_bench_eval.py`  
   — 7 stubs with detailed TODO specs. The P0 task.

4. `/Users/arushsinghal/Documents/RLM/training/synthetic.py`  
   — 6 stubs with detailed TODO specs. The P1 task.

---

## STEP 2: Confirm the Baseline (Do This Before Any Code Changes)

Run:
```bash
cd /Users/arushsinghal/Documents/RLM
uv run pytest tests/test_render.py tests/test_collator.py tests/test_scorer.py tests/test_loss.py tests/test_integrated_repl.py tests/test_cli.py -q
```

Expected: **104 passed**. If anything fails, stop and diagnose before proceeding. Do not write any code until the baseline is confirmed green.

---

## STEP 3: Hard Constraints (Read Before Writing a Single Line)

**These files are fully implemented and tested. Do NOT modify them:**
- `training/render.py`, `training/collator.py`, `training/loss.py`, `training/tokens.py`
- `training/train.py`, `training/eval.py`, `training/data.py`, `training/scorer.py`
- `utils/trajectory.py`, `utils/upload.py`, `utils/llm.py`
- `integrated_repl.py`, `cli.py`, `api/mcp_server.py`, `acc/complexity.py`
- All files in `tests/` (you may ADD new tests, never modify existing ones)

**After every file you write or modify**, re-run the test suite. 104 must still pass.

**Build only what is listed in Step 4.** No refactoring, no new abstractions, no scope creep.

---

## STEP 4: Your Tasks — Work in This Exact Order

### [P0] Implement `experiments/swe_bench_eval.py`

The skeleton exists. Read it. Implement the 7 functions:
- `_load_dataset(split, limit)` — load `princeton-nlp/SWE-bench_Lite` via HuggingFace datasets
- `_setup_instance(instance, workdir)` — git clone + checkout base_commit + apply test_patch
- `_instance_test_command(instance)` — pytest command targeting FAIL_TO_PASS tests
- `_run_baseline(instance, repo_dir, model)` — raw LLM, single shot, extract diff, verify
- `_run_rlm(instance, repo_dir, model, timeout)` — IntegratedRLM with TDRL, extract git diff, verify
- `_verify_patch(instance, patch, workdir)` — fresh clone + apply + run FAIL_TO_PASS tests
- `run_eval(limit, model, output, baseline_only, split, resume)` — orchestration loop

The SUPER_MODEL_BRIEF.md (Section 4, Task 1) has exact implementation guidance for each function including critical details about the dataset field names, honest framing, and error handling.

Key constraint: the framing must be **same-base-model ablation** — run raw(model) vs RLM(model). Never claim to beat SWE-agent. Report the delta.

Also add `gitpython>=3.1.0` to a new `[swe-bench]` group in `pyproject.toml`.

After implementing, run: `uv run python -m experiments.swe_bench_eval --help` to confirm import works.

---

### [P1] Implement `training/synthetic.py`

The skeleton exists with 15 repos pre-selected. Implement the 6 functions:
- `_clone_repo(url, workdir)` — subprocess git clone --depth=500
- `_find_bug_fix_commits(repo_dir, repo_url, max_commits)` — git log with keyword + diff-size filtering
- `_checkout_pre_fix(repo_dir, commit)` — git checkout parent_sha
- `_apply_test_side(repo_dir, commit)` — cherry-pick only test file changes via git diff + git apply
- `_run_rlm_on_commit(repo_dir, commit, model, out_dir)` — full pipeline: pre-check → RLM → verify → save
- `generate_synthetic_trajectories(repos, per_repo, out_dir, model, workers)` — orchestration with progress bar

Section 4, Task 2 in SUPER_MODEL_BRIEF.md has exact implementation guidance.

After implementing, run: `uv run python -m training.synthetic --help`

---

### [Research] Write 5 Documents in `docs/`

Write these with the rigor of an ML research paper, not a startup pitch. Every claim must be either backed by a citation or explicitly labeled as a hypothesis.

**The honest framing must appear in every document**: We are building training infrastructure for agentic models. H0 (Runtime) is inference-time scaffolding. H1 (RLM-0) is the trained model that doesn't exist yet. The moat is structural training signal + trajectory corpus + data flywheel, not model scale.

Documents to write (Section 4, Task 3 in SUPER_MODEL_BRIEF.md has exact specs for each):

**`docs/RESEARCH_AUDIT.md`** (~800 words)  
Compare to: SWE-agent, CodeAct, OpenDevin, Agentless, SWE-Llama, RLEF.  
Identify our genuine differentiators: recursive process training signal, UL loss on failed actions, rollback budget term, `<rollback>` as a learnable discrete action.

**`docs/H2_ARCHITECTURE.md`** (~1200 words)  
Design a transformer architecture where recursion is structural:  
Recurrent State Buffer, Learned Exit Mechanism, Verification Embedding, Rollback Gate.  
Compare to SSM/Mamba, RWKV, Perceiver IO. Be rigorous about what's speculative.

**`docs/DATA_FLYWHEEL_ANALYSIS.md`** (~800 words)  
Quantitative: signal density per trajectory, sample efficiency estimate (target: ~300 trajectories for first measurable P3 improvement), flywheel phases, quality vs. quantity tradeoff via TrajectoryScorer.

**`docs/COMPETITIVE_ANALYSIS.md`** (~800 words)  
Honest competitive matrix. Address: GitHub Copilot, Cursor/Claude Code, Cognition/Devin, SWE-agent. Where the moat actually is: proprietary trajectory corpus, structural loss, data flywheel, 6-month head start on training pipeline.

**`docs/EXPERIMENTAL_ROADMAP.md`** (~1200 words)  
10 concrete experiments with hypothesis / method / success criterion / timeline.  
Section 4, Task 3 in SUPER_MODEL_BRIEF.md lists all 10 experiments with exact specs.

---

## STEP 5: Final Verification (Run Before You're Done)

```bash
# 1. Test suite still green
uv run pytest tests/test_render.py tests/test_collator.py tests/test_scorer.py tests/test_loss.py tests/test_integrated_repl.py tests/test_cli.py -q
# Expected: 104 passed

# 2. CLI works
uv run rlm --help
uv run rlm status

# 3. New CLIs import correctly
uv run python -m experiments.swe_bench_eval --help
uv run python -m training.synthetic --help

# 4. Research docs exist
ls /Users/arushsinghal/Documents/RLM/docs/
# Expected: ARCHITECTURE_FAILURE_TAXONOMY.md, RESEARCH_AUDIT.md,
#           H2_ARCHITECTURE.md, DATA_FLYWHEEL_ANALYSIS.md,
#           COMPETITIVE_ANALYSIS.md, EXPERIMENTAL_ROADMAP.md

# 5. Key imports work
uv run python -c "from RLM.training.scorer import TrajectoryScorer; print('scorer OK')"
uv run python -c "from RLM.utils.upload import upload_trajectory; print('upload OK')"
uv run python -c "from RLM.integrated_repl import IntegratedRLM; print('rlm OK')"
uv run python -c "import RLM.api.mcp_server; print('mcp OK')"
```

All green = YC-ready.

---

## The One Thing That Matters

The 104 tests are the integrity check. If they fail after any of your changes, you broke something. Fix it before moving on. Everything else — the SWE-bench eval, the synthetic data, the research docs — they're all additive. The tests are the floor.

Go.
