# Experimental Roadmap — 6 Months, 10 Falsifiable Experiments

**Status:** Execution plan. Each experiment states a hypothesis, an exact method,
a measurable success criterion, a timeline, resources, and a fallback. The
ordering encodes dependencies: early experiments de-risk the premises that later
ones assume.

**Reading the metrics.** P3 = `feedback_value` from the P1–P4 Recursion Proof
protocol (`training/eval.py`). The bar gates are P3 ≥ 0.25, rollback_precision
≥ 0.70, ECE ≤ 0.10, P1_flip ≥ 0.80. "pp" = percentage points. "delta" = RLM
resolve rate − baseline resolve rate on the same base model.

---

## Phase A — Measure the Runtime (Month 1)

### Experiment 1: Baseline SWE-bench measurement
- **Hypothesis:** raw `gpt-4o` resolves 8–12% of SWE-bench Lite under our harness.
- **Method:** `swe_bench_eval.py --baseline-only --limit 100 --model gpt-4o`.
- **Success criterion:** resolve rate in 8–12%. (This is a *harness validation*
  experiment — it confirms dataset loading, patch application, and verification
  work; the published SWE-agent/gpt-4o numbers sit in this band.)
- **Timeline:** Month 1, week 1.
- **Resources:** ~100 OpenAI calls + git clones; <1 day wall-clock.
- **If fails (rate far outside band):** the verifier is wrong, not the model —
  audit `_verify_patch` (patch application, FAIL_TO_PASS selection) before
  trusting any later number.

### Experiment 2: Runtime delta measurement (the H0 headline)
- **Hypothesis:** `IntegratedRLM(gpt-4o)` resolves 3–8 pp more instances than
  raw `gpt-4o`.
- **Method:** `swe_bench_eval.py --limit 100 --model gpt-4o` (both arms,
  same-base-model ablation).
- **Success criterion:** delta ≥ 2 pp with p < 0.05 on an instance-level paired
  permutation test (McNemar on the resolved/not-resolved contingency table).
- **Timeline:** Month 1, weeks 2–3.
- **Resources:** ~100 baseline + ~100 RLM runs (RLM runs are multi-iteration, so
  budget ~10× tokens); 2–3 days.
- **If fails (delta < 2pp):** ablate which RLM component helps — rerun with
  `enable_acc` only, `enable_tdrl` only, etc. If none move the needle, the
  Runtime value proposition is in question and we revisit before collecting data.

---

## Phase B — Build the Corpus and Train (Months 1–3)

### Experiment 3: Synthetic trajectory generation
- **Hypothesis:** we can generate ≥ 500 verified trajectories from
  `SYNTHETIC_REPOS` within ~2 weeks of metered API usage.
- **Method:** `training/synthetic.py --repos 15 --per-repo 70 --model gpt-4o`,
  with per-repo dependency environments provisioned so tests can run.
- **Success criterion:** ≥ 500 trajectories with `TrajectoryScorer` mean
  score ≥ 0.45 (i.e. not dominated by trivial straight-successes).
- **Timeline:** Months 1–2 (overlaps Phase A).
- **Resources:** 15 repo environments; sustained API budget; ~2 weeks.
- **If fails (yield too low):** the bottleneck is environment setup, not logic —
  add per-repo Docker images with pinned deps; widen the commit window
  (`--per-repo`), or add more repos to `SYNTHETIC_REPOS`.

### Experiment 4: First fine-tuning run (the H1 proof-of-life)
- **Hypothesis:** QLoRA-finetuned Llama-3.1-8B on ~500 trajectories shows
  measurable P3 improvement over the zero-shot 8B on held-out recursion-proof
  tasks.
- **Method:** `training/run_training.py` on the Exp.-3 corpus; evaluate P3 via
  `recursion_proof()` on a held-out task set.
- **Success criterion:** P3(final) − P3(base) ≥ 0.05 on held-out tasks, with no
  forgetting-tripwire breach (>3% drop on the general probe).
- **Timeline:** Months 2–3.
- **Resources:** 1× A100-80GB (or equivalent) for ~hours; the implemented QLoRA
  driver.
- **If fails (no P3 gain):** check signal density (are there enough rollback
  trajectories?) before blaming the recipe — may indicate Exp. 3 corpus is too
  easy; regenerate with difficulty filtering.

---

## Phase C — Understand the Signal (Months 3–4)

### Experiment 5: Scaling-law measurement
- **Hypothesis:** P3 improvement grows smoothly (power-law-like) with trajectory
  count N.
- **Method:** train at N ∈ {100, 300, 500, 1000}; plot P3 vs. N on log-log axes.
- **Success criterion:** monotonic increase with R² > 0.9 on the log-log fit
  (confirms predictable returns to data and lets us forecast the H1 budget).
- **Timeline:** Months 3–4.
- **Resources:** 4 training runs; reuses Exp.-3 corpus subsampled.
- **If fails (noisy / flat):** returns to data are unpredictable — pivot strategy
  from "collect more" to "collect better" (lean harder on `TrajectoryScorer`).

### Experiment 6: Rollback-signal ablation (the core scientific claim)
- **Hypothesis:** removing the unlikelihood term (`alpha_ul = 0`) degrades
  recovery performance.
- **Method:** train two models on the same corpus — full loss vs. CE-only —
  evaluate on a curated set of *rollback-required* tasks (bugs where the first
  plausible fix is wrong).
- **Success criterion:** full-loss model shows ≥ 10% higher recovery rate on
  rollback-required tasks than CE-only.
- **Timeline:** Month 3 (parallel with Exp. 5).
- **Resources:** 2 training runs + a hand-curated rollback-required eval set.
- **If fails (no difference):** the structural-signal thesis is wounded — the UL
  term isn't doing work. This is the experiment most worth knowing early; a null
  result redirects the whole H1 bet.

---

## Phase D — Prove the Flywheel (Months 4–6)

### Experiment 7: Same-model SWE-bench ablation (the YC headline number)
- **Hypothesis:** RLM-0 (finetuned 8B) + Runtime > raw 8B on SWE-bench Lite.
- **Method:** `swe_bench_eval.py` with `model=RLM-0` vs. `model=raw-Llama-3.1-8B`.
- **Success criterion:** delta ≥ 5 pp (same-base-model ablation).
- **Timeline:** Month 4.
- **Resources:** RLM-0 from Exp. 4/5 hosted for inference; full Lite eval.
- **If fails (delta < 5pp but Exp. 6 passed):** the recovery signal helps on
  rollback tasks but not on the Lite distribution — report the honest narrower
  claim and investigate distribution mismatch.

### Experiment 8: Data-flywheel validation (the defensibility test)
- **Hypothesis:** adding ~200 *product* trajectories (real user opt-ins) improves
  P3 over a synthetic-only model of the same size.
- **Method:** finetune v2 on 500 synthetic + 200 product; compare P3 to v1
  (synthetic-only, matched count by subsampling).
- **Success criterion:** P3 improves by ≥ 0.03 attributable to the product data.
- **Timeline:** Month 5.
- **Resources:** live product with `contribute_traces` opt-ins; 2 training runs.
- **If fails (product data doesn't help):** the flywheel is theoretical, not real
  — the most important negative result for the business; forces a strategy
  rethink before scaling user acquisition.

### Experiment 9: Efficiency vs. quality tradeoff
- **Hypothesis:** filtering at `min_score = 0.4` retains > 60% of trajectories
  with no P3 degradation vs. unfiltered.
- **Method:** train on filtered vs. unfiltered corpora of matched size; compare
  P3.
- **Success criterion:** filtered P3 ≥ unfiltered P3 while using fewer
  trajectories (confirms the scorer earns its keep).
- **Timeline:** Months 5–6.
- **Resources:** 2 training runs; reuses existing corpus.
- **If fails (filtering hurts):** the scorer is discarding useful data — recheck
  dimension weights, especially the 0.35 recursion weight.

---

## Phase E — Probe the Future (Month 6)

### Experiment 10: H2 architecture first prototype
- **Hypothesis:** a minimal Recurrent State Buffer (256-dim) reduces context
  growth vs. an expanding KV cache on multi-iteration tasks.
- **Method:** implement a minimal RSB on a 1B base model; measure context tokens
  consumed per task at matched task performance.
- **Success criterion:** RSB model uses ≤ 60% of the KV-cache tokens of the
  loop-as-prompt baseline for equivalent resolve rate.
- **Timeline:** Month 6 (exploratory; gated on Exp. 6 passing).
- **Resources:** 1× GPU; research time; small base model.
- **If fails (no token savings):** H2 via RSB is not the right structural bet —
  shelve and revisit after more H1 data; H2 is a research option, not a
  commitment.

---

## Dependency Graph

```
 Exp1 ──▶ Exp2 ───────────────▶ (H0 value established)
   │
   ▼
 Exp3 ──▶ Exp4 ──▶ Exp5
            │  ╲
            ▼   ╲──▶ Exp6  (core scientific claim — highest priority signal)
          Exp7 ◀────┘
            │
            ▼
          Exp8 ──▶ Exp9      (flywheel + efficiency)
                     │
                     ▼
                   Exp10     (H2 probe, gated on Exp6)
```

## Go / No-Go Gates

- **After Exp. 2:** if no Runtime delta, do not invest in data collection.
- **After Exp. 4:** if no P3 gain at 500 trajectories, do not scale the corpus
  until the signal is debugged.
- **After Exp. 6:** if the UL ablation shows no difference, the structural-signal
  thesis fails — pause H1 scaling and reassess.
- **After Exp. 8:** if product data doesn't compound, the flywheel is not real —
  the defining business risk; address before fundraising on it.
