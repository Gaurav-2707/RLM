# Super-Model Operating Brief — Recursive Labs / RLM

**READ THIS ENTIRE DOCUMENT BEFORE TOUCHING ANY CODE OR FILES.**

This document tells you exactly what to build, what not to touch, how to verify
nothing is broken, and what YC-ready looks like. Every word here is load-bearing.

---

## 0. Who You Are in This Session

You are the principal ML engineer + research lead for Recursive Labs. You have
full filesystem access to the project at `/Users/arushsinghal/Documents/RLM/`.
This is a **one-shot session**. There is no back-and-forth. Do the work completely,
correctly, and in priority order. Leave the repo in a state that a YC partner
could clone, run, and demo in 10 minutes.

---

## 1. The Vision (Read Once, Then Act)

### What RLM Is
A Recursive Language Model has memory, self-correction, and verification baked
into its weights — not orchestrated at inference time. Today's "agents" are LLMs
with scaffolding. RLM-0 will be an LLM that *learned* how to retry, roll back,
and verify from structured training signal.

### Three Horizons
- **H0 (ships now)**: RLM Runtime — a wrapper that makes any LLM recursive.
  The product is the `rlm` CLI + `IntegratedRLM`. Real users, real data collection.
- **H1 (6 months)**: RLM-0 — QLoRA fine-tuned Llama 3.1 8B trained on verified
  recursive trajectories. The training pipeline (`training/`) is built and tested.
  What's missing: the trajectory corpus (which synthetic.py generates) and the
  SWE-bench number (which swe_bench_eval.py produces).
- **H2 (18 months)**: Native RLM architecture — recursion structurally in the
  architecture. Not built yet. You will design it in `docs/H2_ARCHITECTURE.md`.

### The Honest Claim
> RLM Runtime + any base LLM > the same base LLM alone, on recursive coding tasks.

This is the claim. NOT: "RLM-0 8B beats GPT-4 on SWE-bench." That's a different
claim and it's not ours today. The differentiation is structural training signal
quality, not raw model scale. This framing must appear in every research document.

### Why Training Signal Quality Is the Moat
- OpenAI trains on human preference → subjective, expensive, doesn't capture
  failure dynamics
- We train on compiler/test-verified outcomes of the recursive process itself →
  objective, free, captures rollback→recovery behavior that no other dataset has
- The `<rollback>` token, unlikelihood loss on final failed action, and rollback
  budget penalty are why this training signal is structurally different

---

## 2. Current State — What's Built and Working

**Run this first to confirm baseline:**
```bash
cd /Users/arushsinghal/Documents/RLM
uv run pytest tests/test_render.py tests/test_collator.py tests/test_scorer.py \
    tests/test_loss.py tests/test_integrated_repl.py tests/test_cli.py -q
# Expected: 104 passed in ~4s
```

If anything fails, **stop immediately and diagnose before proceeding.**
These 104 tests are your canary. If they break after any of your changes, revert
and fix before moving on.

### What's Implemented and Tested

| Component | File | Status |
|-----------|------|--------|
| Training render | `training/render.py` | ✅ tested (19 tests) |
| Training collator | `training/collator.py` | ✅ tested (18 tests) |
| Training loss | `training/loss.py` | ✅ tested (12 tests) |
| Trajectory scorer | `training/scorer.py` | ✅ tested (25 tests) |
| IntegratedRLM | `integrated_repl.py` | ✅ tested (16 tests) |
| CLI | `cli.py` | ✅ tested (14 tests) |
| Upload pipeline | `utils/upload.py` | ✅ implemented + wired |
| MCP server | `api/mcp_server.py` | ✅ complete stdio loop |
| ACC complexity | `acc/complexity.py` | ✅ implemented |

### What's a Skeleton (Your Job)

| File | Status |
|------|--------|
| `experiments/swe_bench_eval.py` | Skeleton: 7 stubs with detailed TODO specs |
| `training/synthetic.py` | Skeleton: 6 stubs with detailed TODO specs |
| `docs/RESEARCH_AUDIT.md` | Does not exist |
| `docs/H2_ARCHITECTURE.md` | Does not exist |
| `docs/DATA_FLYWHEEL_ANALYSIS.md` | Does not exist |
| `docs/COMPETITIVE_ANALYSIS.md` | Does not exist |
| `docs/EXPERIMENTAL_ROADMAP.md` | Does not exist |

---

## 3. Hard Constraints — Never Violate These

### Files You Must NOT Modify
These are tested and correct. Any change risks breaking the 104-test baseline.

```
training/render.py         # BUG1+BUG2 fixed, 19 tests guard it
training/collator.py       # per-span tokenization, tail-truncation
training/loss.py           # L_ce + 0.5·L_ul + 0.1·L_rb — exact weights
training/tokens.py         # 26 control tokens, mean-init logic
training/train.py          # QLoRA config, LoRA targets, P3 checkpoint logic
training/eval.py           # P1-P4 Recursion Proof protocol
training/data.py           # 4% replay mix, load logic
training/scorer.py         # fully implemented, 25 tests
utils/trajectory.py        # final_outcome is bool — do NOT change this field type
utils/upload.py            # fully implemented
utils/llm.py               # 9-provider LLM client
integrated_repl.py         # hero product — don't refactor
cli.py                     # 6 commands, all tested
api/mcp_server.py          # complete MCP stdio server
acc/complexity.py          # ComplexityScorer implemented
tests/test_*.py            # DO NOT MODIFY existing tests (add new ones only)
CODEBASE_FOR_AI.md         # already updated — leave it
```

### Design Decisions You Must Preserve

| Decision | Why — Don't Change |
|----------|-------------------|
| `final_outcome: bool` in `RLMTrajectory` | `True`=success, `False`=failure. `bool("failure")==True` would silently corrupt UL masks — this bug was already fixed. |
| Per-span tokenization in `collator.py` | No mask drift. The only correct approach for per-role training masks. |
| Tail truncation (not head) | Keeps the `<final>` token. Changing to head truncation loses the training signal. |
| UL only on final failed action | Not on every failed step. This is the core of the unlikelihood term. |
| L_ce + **0.5**·L_ul + **0.1**·L_rb | These weights are calibrated. Don't change them. |
| 4% replay mix | Forgetting mitigation. Don't increase or remove. |
| Checkpoint on **P3**, not val_loss | P3 = feedback_value. This is intentional. |
| Upload uses **urllib** (stdlib) | No extra dep. |
| TF (not TF-IDF) for novelty scoring | Good enough for corpus filter, simpler. |

### Scope Constraints
Build **only** what is listed in Section 4. Do not:
- Add new dependencies without a good reason (and add them to `pyproject.toml` optional deps)
- Refactor working code "while you're in there"
- Add abstractions that aren't needed for the listed tasks
- Change the project structure (no new top-level modules)
- Add type stubs, formatters, or linters (out of scope)

---

## 4. Your Tasks — In Priority Order

Work through these in order. Don't start the next task until the previous one is
done and the test suite is still green.

---

### TASK 1 [P0]: Implement `experiments/swe_bench_eval.py`

**Why P0**: This is the YC number. Without a SWE-bench Lite resolve rate, we have
no credibility. A demo without a number is a toy. A number without a demo is
impressive. We want both.

**The honest framing** (critical — do not deviate):
We run the **same base model** twice: once raw, once wrapped in RLM Runtime.
We report the delta. We are NOT claiming to beat SWE-agent or Devin.
We claim: `RLM(gpt-4o) > raw(gpt-4o)` on SWE-bench Lite.
The output JSON must make this "same-base-model ablation" structure crystal clear.

**Dataset format** (`princeton-nlp/SWE-bench_Lite`):
```python
# Each instance has:
instance["instance_id"]        # e.g. "astropy__astropy-1234"
instance["repo"]               # e.g. "astropy/astropy"
instance["base_commit"]        # git SHA to checkout
instance["problem_statement"]  # the GitHub issue text
instance["hints_text"]         # optional hints
instance["test_patch"]         # patch that installs the failing tests
instance["patch"]              # gold patch (for reference only — never use to "cheat")
instance["FAIL_TO_PASS"]       # list of test nodeids that must flip FAIL→PASS
instance["PASS_TO_PASS"]       # list of test nodeids that must stay passing
instance["environment_setup_commit"]  # SHA for env setup (can skip for simplicity)
```

**Implement these 7 functions** (skeletons with TODO specs already in the file):

1. `_load_dataset(split, limit)` → `List[Dict]`
   ```python
   from datasets import load_dataset
   ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
   instances = [dict(row) for row in ds]
   return instances[:limit] if limit else instances
   ```

2. `_setup_instance(instance, workdir)` → `str` (repo_dir path)
   - `git clone https://github.com/{repo}` into workdir
   - `git checkout {base_commit}`
   - Apply `test_patch` via `patch -p1` (this installs the tests that need to pass)
   - Return repo_dir path
   - Use `subprocess.run(..., check=False)` — don't let git failures crash the evaluator

3. `_instance_test_command(instance)` → `str`
   ```python
   # Target only FAIL_TO_PASS tests for efficiency
   tests = " ".join(instance["FAIL_TO_PASS"][:10])  # cap at 10 test nodeids
   return f"python -m pytest {tests} -x --tb=short -q --timeout=60"
   ```

4. `_run_baseline(instance, repo_dir, model)` → `Dict`
   - Build prompt: issue title + problem_statement + list of changed file names
   - Call `LLMClient(model=model).completion(prompt)` once (no iteration)
   - Extract ```diff ... ``` or ```patch ... ``` block from response
   - Verify with `_verify_patch(instance, patch, fresh_tempdir)`
   - Return `{"resolved": bool, "patch": str, "steps": 1, "error": None}`

5. `_run_rlm(instance, repo_dir, model, timeout)` → `Dict`
   ```python
   from RLM.integrated_repl import IntegratedRLM
   test_cmd = _instance_test_command(instance)
   rlm = IntegratedRLM(
       model=model,
       enable_acc=True,
       enable_engine=True,
       enable_tdrl=True,
       repo_path=repo_dir,
       test_command=test_cmd,
   )
   answer = rlm.completion(context=[], query=instance["problem_statement"])
   patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo_dir, text=True)
   resolved = _verify_patch(instance, patch, tempfile.mkdtemp())
   steps = rlm._trajectory_collector._active.total_steps if ... else 0
   rollbacks = rlm._trajectory_collector._active.total_rollbacks if ... else 0
   return {"resolved": resolved, "patch": patch, "steps": steps, "rollbacks": rollbacks, "error": None}
   ```
   Wrap everything in try/except. Timeout via `signal.alarm(timeout)` or `subprocess.run(timeout=...)`.

6. `_verify_patch(instance, patch, workdir)` → `bool`
   - Clone fresh copy of repo to workdir
   - `git checkout base_commit`
   - Apply test_patch (so the failing tests exist)
   - Apply the candidate patch (`patch -p1`)
   - If patch fails to apply, return False
   - Run `_instance_test_command(instance)` via subprocess
   - Return `result.returncode == 0`
   - **Important**: also check that PASS_TO_PASS tests still pass if you have time

7. `run_eval(limit, model, output, baseline_only, split, resume)` → `Dict`
   - Load dataset
   - For each instance (with `rich.progress.track` progress bar):
     - Create a tempdir per instance, clean up after
     - Run `_setup_instance`
     - Run `_run_baseline`
     - If not baseline_only: run `_run_rlm`
     - Collect result dict
   - Compute summary with: total, baseline_resolved, rlm_resolved,
     baseline_resolve_rate, rlm_resolve_rate, delta, rlm_only_resolved
   - Write JSON to output path
   - Return full results dict

**Important**: Install deps: add `gitpython>=3.1.0` to `[project.optional-dependencies]`
swe-bench group in `pyproject.toml`. `datasets` is already in the `training` group.

---

### TASK 2 [P1]: Implement `training/synthetic.py`

**Why P1**: We need trajectory data before we can train RLM-0. The product flywheel
won't have enough data for months. Synthetic generation from real bug-fixing commits
is how we bootstrap. Target: 500–1000 verified trajectories.

**Implement these 6 functions** (skeletons with TODO specs already in the file):

1. `_clone_repo(url, workdir)` → `str` (repo_dir)
   ```python
   repo_name = url.rstrip("/").split("/")[-1]
   repo_dir = os.path.join(workdir, repo_name)
   subprocess.run(["git", "clone", "--depth=500", url, repo_dir],
                  check=True, capture_output=True)
   return repo_dir
   ```

2. `_find_bug_fix_commits(repo_dir, repo_url, max_commits)` → `List[CommitInfo]`
   ```python
   BUG_KEYWORDS = ["fix", "bug", "error", "issue", "patch", "incorrect",
                   "wrong", "broken", "regression", "resolve", "repair"]
   MIN_DIFF_LINES = 5
   MAX_DIFF_LINES = 400
   
   # Get log: sha, parent_sha, subject
   out = subprocess.check_output(
       ["git", "log", "--format=%H %P %s", "-n", "1000"],
       cwd=repo_dir, text=True
   )
   results = []
   for line in out.strip().splitlines():
       parts = line.split(" ", 2)
       if len(parts) < 3:
           continue
       sha, parent_sha, msg = parts
       if not any(k in msg.lower() for k in BUG_KEYWORDS):
           continue
       # Get changed files
       files = subprocess.check_output(
           ["git", "diff", "--name-only", parent_sha, sha],
           cwd=repo_dir, text=True
       ).strip().splitlines()
       py_files = [f for f in files if f.endswith(".py")]
       test_files = [f for f in py_files
                    if "test" in f.lower() or "spec" in f.lower()]
       if not py_files or not test_files:
           continue
       # Check diff size
       diff = subprocess.check_output(
           ["git", "diff", parent_sha, sha, "--", "*.py"],
           cwd=repo_dir, text=True
       )
       diff_lines = len([l for l in diff.splitlines() if l.startswith(("+", "-"))
                        and not l.startswith(("+++", "---"))])
       if not (MIN_DIFF_LINES <= diff_lines <= MAX_DIFF_LINES):
           continue
       results.append(CommitInfo(
           sha=sha, parent_sha=parent_sha, message=msg,
           files_changed=py_files, test_files=test_files,
           diff_lines=diff_lines, repo_url=repo_url
       ))
       if len(results) >= max_commits:
           break
   return results
   ```

3. `_checkout_pre_fix(repo_dir, commit)` — checkout parent commit
   ```python
   subprocess.run(["git", "checkout", commit.parent_sha],
                  cwd=repo_dir, check=True, capture_output=True)
   ```

4. `_apply_test_side(repo_dir, commit_sha)` → bool
   Cherry-pick only test file changes from the fix commit into the parent state,
   so the failing tests exist before RLM sees the code. Strategy:
   ```python
   # Get the diff for test files only
   diff = subprocess.check_output(
       ["git", "diff", f"{commit.parent_sha}..{commit.sha}",
        "--", "*test*.py", "*spec*.py"],
       cwd=repo_dir, text=True
   )
   if not diff.strip():
       return False  # no test changes to apply
   # Apply just the test side
   proc = subprocess.run(["git", "apply", "--reject", "-"],
                         input=diff, cwd=repo_dir,
                         text=True, capture_output=True)
   return proc.returncode == 0
   ```

5. `_run_rlm_on_commit(repo_dir, commit, model, trajectory_out_dir)` → `SyntheticResult`
   - Call `_checkout_pre_fix(repo_dir, commit)`
   - Call `_apply_test_side(repo_dir, commit)` — if returns False, skip
   - Build issue text: `_extract_issue_text(repo_dir, commit)`
   - Build test command: `_build_test_command(repo_dir, commit)`
   - Pre-check: run test command, verify at least one test fails (confirms the bug is real)
   - Run `IntegratedRLM` with TDRL enabled
   - Extract git diff vs parent_sha
   - Verify with fresh clone + apply patch + run tests
   - If verified, save trajectory JSON via `TrajectoryCollector`
   - Return `SyntheticResult`

6. `generate_synthetic_trajectories(repos, per_repo, out_dir, model, workers)` → `dict`
   - Iterate SYNTHETIC_REPOS[:repos]
   - For each repo: clone to tempdir, find commits, run `_run_rlm_on_commit`
   - If workers > 1: use `concurrent.futures.ThreadPoolExecutor`
     (each worker gets its own cloned repo copy — no shared state)
   - Collect results, print `rich` progress bar
   - Write `synthetic_stats.json` to out_dir
   - Return `{attempted, verified, resolve_rate, by_repo}`

**Error handling**: wrap every per-commit operation in try/except.
One bad repo/commit must never crash the whole run. Log and continue.

---

### TASK 3 [Research]: Write 5 Documents in `docs/`

These are the intellectual foundation for the YC application. Write them with
the rigor and precision of an NeurIPS paper intro, not a startup blog post.
No fluff. Every claim should be either cited or clearly labeled as a hypothesis.

Use the honest framing throughout: we are building training infrastructure
for agentic models, not claiming SOTA on day 1.

---

#### `docs/RESEARCH_AUDIT.md`

**Purpose**: Situate RLM relative to the existing literature. Show we know what
we're talking about and where we're genuinely different.

**Cover these systems** (read the papers, not just the abstracts):
- **SWE-agent** (Yang et al. 2024): Agent Computer Interface (ACI) with bash tools.
  Observation: iterative, but no learning from failures. Each run is stateless.
- **CodeAct** (Wang et al. 2024): LLM generates executable Python actions.
  Observation: no rollback mechanism. No training signal from failures.
- **OpenDevin** (Wang et al. 2024): multi-agent scaffolding with sandbox.
  Observation: inference-time orchestration, not trained behavior.
- **Agentless** (Xia et al. 2024): simple localize→repair→patch without an agent loop.
  Observation: surprisingly competitive. No learning.
- **SWE-Llama** (Jim et al. 2023): fine-tuned on issue→patch pairs.
  Observation: learns "what the patch looks like" but not "how to recover from mistakes."
- **RLEF** (Gehring et al. 2024): RL with execution feedback on code tasks.
  Closest related work. Key difference: our loss uses structured rollback signals,
  not just final binary reward.

**Our genuine differentiation**:
- Training signal from the *recursive process* (rollback→recovery dynamics), not from gold patches
- Unlikelihood loss on final failed action — actively pushes away from dead-end moves
- Rollback budget term — regularizes overuse of the rollback mechanism
- The `<rollback>` control token is a learnable discrete action, not a meta-instruction

**Be honest about limitations**:
- H0 (Runtime) doesn't train — it's inference-time scaffolding like the others
- The trained advantage only comes with RLM-0 (H1)
- We don't yet have the trajectories or the GPU run to prove H1

Structure: Introduction, Related Work table, Key Differentiators, Limitations, Open Questions

---

#### `docs/H2_ARCHITECTURE.md`

**Purpose**: A concrete design proposal for a transformer architecture where
recursion and rollback are structural, not behavioral.

**The question**: How would you build a model where the ability to retry, verify,
and roll back is in the *architecture* rather than in prompting/scaffolding?

**Design the following** (with architectural diagrams in ASCII or description):

1. **Recurrent State Buffer** (RSB): A fixed-length learnable buffer that persists
   across iterations within a task. At each iteration, the model can read from
   and write to the RSB via cross-attention. This gives true statefulness without
   growing context length. Compare to SSM / Mamba recurrent state.

2. **Learned Exit Mechanism**: Instead of prompting the model with "are you done?",
   add a learned binary head that predicts whether to exit or recurse.
   At training time, supervise with `final_outcome` signal.
   At inference time, use conformal calibration (ACC) to set the exit threshold.

3. **Verification Embedding**: After each test run, encode the structured output
   (pass/fail + failing test names) into a special embedding vector injected
   via a cross-attention layer. The model learns to read "what failed" as
   structured signal, not raw text.

4. **Rollback Gate**: A gating mechanism that the model can "open" to reset
   certain activations to their state before the last edit. Analogous to an
   LSTM forget gate but operating on the RSB.

**Be rigorous**: Identify which parts are speculative hypothesis vs. grounded
in existing architecture work. Cite SSM/Mamba, RWKV, Perceiver IO, etc.
as related approaches.

**End with**: Training objectives for H2. How do L_ce, L_ul, and L_rb translate
to the native architecture? What new loss terms might be needed?

---

#### `docs/DATA_FLYWHEEL_ANALYSIS.md`

**Purpose**: Quantitative analysis of how many trajectories we need before
RLM-0 training starts paying off. This is the answer to "when does H1 happen?"

**Analyze these questions**:

1. **Signal density per trajectory**:
   - Average trajectory has ~7 steps, ~1.5 rollbacks (estimate from existing data)
   - Each step with a rollback → 2x training signal vs a step without
   - A single trajectory with 1 rollback at step k generates roughly:
     - k decision spans as positive examples
     - 1 unlikelihood example (the final failed action)
     - Rollback budget signal
   - Compare to SFT: you'd need N gold (issue, patch) pairs to get equivalent
     CE signal from N trajectories. But UL and rollback budget signal don't exist
     in SFT at all.

2. **Empirical sample efficiency estimate**:
   - PEFT fine-tuning on 8B model: literature suggests 100-1000 examples for
     meaningful few-shot specialization
   - Our examples are richer (multiple signals per traj) → estimate ~300 trajectories
     for first measurable P3 improvement on eval tasks
   - Conservative: plan for 1000 verified trajectories before first training run

3. **Flywheel dynamics**:
   Phase 1 (0-1000 trajectories): Synthetic generation from SYNTHETIC_REPOS
   Phase 2 (1000-5000): Product usage via `contribute_traces=True`
   Phase 3 (5000+): Self-improvement loop (RLM-0 generates better trajectories)
   
4. **Quality vs. Quantity tradeoff**:
   - TrajectoryScorer (training/scorer.py) filters on min_score=0.4
   - A rollback+success traj (score ~0.75) is worth ~3x a straight success (score ~0.3)
   - Implication: 300 rollback+success trajectories ≈ 900 straight successes in training value

5. **Data quality risks**:
   - Spurious successes: model edits pass tests by deleting tests
   - Mitigation: PASS_TO_PASS check in swe_bench_eval.py
   - Distribution mismatch: synthetic from SYNTHETIC_REPOS may not match real user tasks
   - Mitigation: TrajectoryScorer novelty dimension incentivizes diversity

---

#### `docs/COMPETITIVE_ANALYSIS.md`

**Purpose**: Honest, rigorous analysis of where RLM has a moat vs. where it doesn't.
YC partners read dozens of these. They reward honesty over hype.

**The honest picture**:

| Dimension | RLM Runtime (H0) | Inference-time agents | RLM-0 (H1, future) |
|-----------|------------------|-----------------------|---------------------|
| Capability today | Runtime delta over base LLM | Best today (SWE-agent etc.) | Not trained yet |
| Moat | None (scaffolding is copyable) | None (scaffolding is copyable) | Structural training signal |
| Data | Accumulating | None | 1000+ trajectories needed |
| Latency | 2-10x base model | 2-10x base model | 1x (baked in) |

**Where the moat actually is**:
1. **Proprietary trajectory corpus**: Verified recursive trajectories are not
   available anywhere. SWE-bench patches exist but they're destination, not process.
2. **Structural loss**: UL on failed actions + rollback budget → only works if you
   have trajectories with the rollback→recovery signal. Can't reproduce without data.
3. **Data flywheel**: Each product user who opts in feeds the next model version.
   Network effects on training data are a real defensibility.
4. **Speed-to-trained-model**: We have the training pipeline today. Others would
   have to build it + collect data. 6-month head start.

**Competitors to address honestly**:
- GitHub Copilot: completion-only, not agentic. Different product.
- Cursor/Claude Code: agentic but no per-model training. Scaffolding.
- Cognition/Devin: full autonomy claim, black box, not open to study.
  We will beat their SWE-bench delta on same-model ablation — that's our claim.
- SWE-agent: OSS, strong baseline. Our comparison target.

**The VC-facing thesis**:
> The race in coding AI is not about which model is biggest today.
> It's about who builds the feedback loop that compounds fastest.
> We are the only team building training infrastructure that captures
> recursive failure dynamics — the hardest part of software debugging.

---

#### `docs/EXPERIMENTAL_ROADMAP.md`

**Purpose**: Concrete, falsifiable 6-month plan. Each experiment has a hypothesis,
a method, and a clear success criterion. YC wants to see you know what you're
measuring.

**Structure each experiment as**:
```
Experiment N: [Name]
Hypothesis: [What we believe will happen]
Method: [Exactly how we'll test it]
Success criterion: [Specific measurable outcome]
Timeline: [Month X]
Resources: [Compute, time, team]
If fails: [What we'll try instead]
```

**Write these experiments** (minimum 8, up to 12):

1. **Baseline SWE-bench measurement** (Month 1)
   H: raw gpt-4o resolves 8-12% of SWE-bench Lite
   M: Run swe_bench_eval.py --baseline-only --limit 100 --model gpt-4o
   S: number in 8-12% range (confirms dataset loading + verification works)

2. **Runtime delta measurement** (Month 1)
   H: IntegratedRLM(gpt-4o) resolves 3-8% more instances than raw gpt-4o
   M: Run swe_bench_eval.py --limit 100 --model gpt-4o (full eval)
   S: delta ≥ 2pp (percentage points), p < 0.05 on instance-level permutation test

3. **Synthetic trajectory generation** (Month 1-2)
   H: We can generate 500+ verified trajectories from SYNTHETIC_REPOS in 2 weeks of API calls
   M: Run synthetic.py --repos 15 --per-repo 70 --model gpt-4o
   S: ≥ 500 verified trajectories with TrajectoryScorer mean score ≥ 0.45

4. **First fine-tuning run** (Month 2-3)
   H: QLoRA fine-tuned 8B on 500 trajectories shows measurable P3 improvement
      over zero-shot 8B on recursion_proof eval tasks
   M: Run training/run_training.py on 500-trajectory corpus, evaluate P3 metric
   S: P3 (final) > P3 (base) by ≥ 0.05 on held-out eval tasks

5. **Scaling law measurement** (Month 3-4)
   H: P3 improvement follows a power law with trajectory count
   M: Train on 100, 300, 500, 1000 trajectories. Plot P3 vs N.
   S: R² > 0.9 on log-log scale (confirms smooth scaling)

6. **Rollback signal ablation** (Month 3)
   H: Removing UL loss (alpha_ul=0) degrades recovery performance
   M: Train two models: full loss vs. CE-only. Compare on rollback-required eval tasks.
   S: Full loss model shows ≥ 10% better recovery rate on rollback-required tasks

7. **Same-model SWE-bench ablation** (Month 4)
   H: RLM-0 (fine-tuned 8B) + Runtime > raw 8B on SWE-bench Lite
   M: Run swe_bench_eval.py with model=RLM-0 vs model=raw-Llama-3.1-8B
   S: delta ≥ 5pp (this is the YC headline number)

8. **Data flywheel validation** (Month 5)
   H: Adding 200 product trajectories (from real user opt-ins) improves P3 vs. synthetic-only
   M: Fine-tune v2 on 500 synthetic + 200 product trajectories. Compare P3 to v1.
   S: P3 improves by ≥ 0.03 (confirms flywheel is real, not theoretical)

9. **Efficiency vs. quality tradeoff** (Month 5-6)
   H: TrajectoryScorer filtering at min_score=0.4 retains >60% of trajectories
      with no degradation in P3 vs unfiltered
   M: Train on filtered vs. unfiltered corpus of same size
   S: Filtered training ≥ unfiltered P3 with fewer trajectories

10. **H2 architecture first prototype** (Month 6)
    H: A simple RSB (recurrent state buffer) with 256-dim state shows reduced
       context growth vs. expanding KV cache on multi-iteration tasks
    M: Implement minimal RSB on top of 1B model. Measure context tokens used per task.
    S: RSB model uses ≤ 60% of KV cache tokens for equivalent performance

---

## 5. Verification Protocol

After every file you write or modify, run:

```bash
uv run pytest tests/test_render.py tests/test_collator.py tests/test_scorer.py \
    tests/test_loss.py tests/test_integrated_repl.py tests/test_cli.py -q
```

**104 tests must pass every time.**

After implementing swe_bench_eval.py, run the smoke test:
```bash
# Dry run without API calls (tests import and CLI)
uv run python -m experiments.swe_bench_eval --help
```

After implementing synthetic.py, run:
```bash
uv run python -m training.synthetic --help
```

After writing research docs, verify they exist:
```bash
ls docs/*.md
```

---

## 6. What YC-Ready Looks Like

When you're done, a YC partner should be able to:

1. **Run the demo**: `uv run rlm run "fix the failing test" --test-command "pytest"` — works
2. **See the number**: `experiments/results/swe_bench_results.json` — resolve rate + delta
3. **Understand the moat**: `docs/COMPETITIVE_ANALYSIS.md` — honest, specific, defensible
4. **Believe the trajectory**: `docs/EXPERIMENTAL_ROADMAP.md` — concrete, falsifiable
5. **Trust the team knows ML**: `docs/RESEARCH_AUDIT.md` + `docs/H2_ARCHITECTURE.md` — rigorous
6. **Understand the flywheel**: `docs/DATA_FLYWHEEL_ANALYSIS.md` — quantitative

The test suite (104 tests) is the proxy for "the codebase is production-quality."
The SWE-bench number is the proxy for "this works."
The research docs are the proxy for "this team knows where they're going."

---

## 7. Final Checklist (Run Before You're Done)

```bash
# 1. All tests green
uv run pytest tests/test_render.py tests/test_collator.py tests/test_scorer.py \
    tests/test_loss.py tests/test_integrated_repl.py tests/test_cli.py -q
# Expected: 104 passed

# 2. CLI works
uv run rlm --help
uv run rlm status

# 3. Skeleton CLIs work (import check)
uv run python -m experiments.swe_bench_eval --help
uv run python -m training.synthetic --help

# 4. Research docs exist
ls -la docs/
# Should show: RESEARCH_AUDIT.md, H2_ARCHITECTURE.md, DATA_FLYWHEEL_ANALYSIS.md,
#              COMPETITIVE_ANALYSIS.md, EXPERIMENTAL_ROADMAP.md

# 5. No new broken imports
uv run python -c "from RLM.training.scorer import TrajectoryScorer; print('OK')"
uv run python -c "from RLM.utils.upload import upload_trajectory; print('OK')"
uv run python -c "from RLM.integrated_repl import IntegratedRLM; print('OK')"
uv run python -c "import RLM.api.mcp_server; print('OK')"
```

If all of these pass and the research docs are written with the rigor described in
Section 4, the repo is YC-ready.
