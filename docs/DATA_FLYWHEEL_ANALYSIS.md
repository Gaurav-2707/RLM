# Data Flywheel Analysis — When Does H1 Pay Off?

**Status:** Quantitative analysis with explicit assumptions. Numbers tagged
**[measured]** come from the existing codebase/data; **[estimate]** are
back-of-envelope; **[hypothesis]** are falsifiable predictions.

**The question this answers:** how many verified trajectories do we need before
fine-tuning RLM-0 beats the Runtime-only baseline — i.e., when does H1 happen?

---

## 1. Signal Density Per Trajectory

A trajectory is rendered (see `training/render.py`) into role-tagged spans. Only
*decision* spans (action, rollback, confidence) are trainable; *world* spans
(state, test verdict) are masked. The training signal per trajectory is therefore
proportional to the number of decision tokens, not the total length.

Assumptions **[estimate]** (to be replaced with measured values once the
synthetic corpus exists):

- Average trajectory: ~7 steps, ~1.5 rollbacks.
- A trajectory with a rollback at step `k` contributes:
  - `k` decision spans as positive cross-entropy examples (`L_ce`),
  - exactly 1 unlikelihood example on the final failed action (`L_ul`),
  - 1 rollback-budget signal across the whole sequence (`L_rb`).

**Contrast with SFT on gold patches.** An `(issue → patch)` pair yields one
target sequence: pure `L_ce` on the patch tokens. It contains:

- **zero** `L_ul` signal (there is no failed attempt to push away from), and
- **zero** `L_rb` signal (there is no rollback to budget).

So even at equal trajectory count `N`, our corpus carries two signal channels
that gold-patch SFT structurally cannot. The marginal value of a trajectory is
*not* substitutable by a gold patch.

---

## 2. Sample-Efficiency Estimate

**[established]** PEFT/QLoRA literature (Hu et al., 2021; Dettmers et al., 2023)
shows that for narrow behavioral specialization on an 8B model, meaningful shifts
appear in the **100–1,000 example** range — far below full-finetune scales.

**[hypothesis]** Because each RLM trajectory carries multiple supervised signals
(dense `L_ce` over decisions + targeted `L_ul` + global `L_rb`), the effective
example count per trajectory is >1 relative to single-target SFT. We therefore
predict the **first measurable P3 improvement at ~300 verified trajectories**,
where P3 is the `feedback_value` metric from the P1–P4 Recursion Proof protocol
(`training/eval.py`).

**Planning number:** target **1,000 verified trajectories** before the first
serious fine-tuning run — a 3× margin over the predicted threshold to absorb the
optimism in the estimate.

These are predictions, not results. Exp. 4 and Exp. 5 in
`EXPERIMENTAL_ROADMAP.md` test them directly (the scaling-law experiment fits
P3 vs. N and will tell us the true threshold).

---

## 3. Quality Beats Quantity (and We Can Measure Quality)

`training/scorer.py` (`TrajectoryScorer`, fully implemented + 25 tests) scores
each trajectory in `[0,1]` across five weighted dimensions:

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Recursion (rollback→recovery) | 0.35 | The scarce, high-value signal |
| Difficulty (steps to solve) | 0.25 | Harder problems = richer supervision |
| Novelty (TF cosine vs. corpus) | 0.20 | Diversity guards against mode collapse |
| Verification depth (FAIL→PASS flips) | 0.15 | More verified transitions = more signal |
| Efficiency (terse successes) | 0.05 | Penalizes wandering |

Concrete scores from the implemented scorer:

- A **rollback-then-succeed** trajectory: recursion=1.0 → blended score ≈ **0.75**.
- A **straight success, no rollback**: recursion=0.6 → blended score ≈ **0.45**.
- A **failure**: recursion=0.2, efficiency=0 → blended score ≈ **0.20**.

**[hypothesis]** A rollback+success trajectory is worth ~3× a straight success
in training value (the ratio of their recovery signal). Implication: **~300
rollback+success trajectories ≈ ~900 straight successes** in effective signal.
This is why the synthetic generator and the product both bias toward collecting
recovery trajectories, and why `filter_corpus(min_score=0.4)` discards the
bottom tier rather than training on everything.

---

## 4. Flywheel Dynamics — Three Phases

```
 Phase 1: COLD START            Phase 2: PRODUCT             Phase 3: SELF-IMPROVE
 0 → 1,000 trajectories         1,000 → 5,000                5,000+
 ────────────────────────       ────────────────────         ──────────────────────
 source: synthetic.py mining    source: contribute_traces    source: RLM-0 runs that
 15 OSS repos, bug-fix commits   opt-in from real users       generate new verified
 verified vs. their own tests    on real tasks                trajectories itself
 ↓                               ↓                            ↓
 trains RLM-0 v1                 trains RLM-0 v2              trains RLM-0 v(n+1)
 (proves the signal works)       (in-distribution gains)      (compounding moat)
```

- **Phase 1 → 2 transition** is gated by Exp. 4 (first fine-tune shows P3 gain).
- **Phase 2 → 3 transition** is gated by Exp. 8 (product trajectories add value
  on top of synthetic). This is the experiment that proves the flywheel is real
  and not just a slide.

---

## 5. Data-Quality Risks and Mitigations

| Risk | Mechanism | Mitigation (status) |
|------|-----------|---------------------|
| **Reward hacking** — model "passes" by deleting/weakening tests | Test-verified outcome is gameable | `PASS_TO_PASS` guard in `swe_bench_eval.py`; synthetic verifier replays the *committed* test side, which the model cannot edit away in the clean re-check **[implemented]** |
| **Spurious success** — flaky test passes by luck | Non-determinism | Independent re-verification on a clean checkout (`_verify_candidate` / `_verify_patch`) **[implemented]** |
| **Distribution mismatch** — OSS synthetic ≠ product tasks | Phase-1 corpus skew | `TrajectoryScorer` novelty term rewards diversity; Phase 2 product data corrects distribution **[partial]** |
| **Mode collapse on common bug types** (the 100th off-by-one) | Over-represented easy fixes | Novelty + difficulty weighting in scorer down-weight repeats **[implemented]** |
| **Trajectory length bias** — long wandering runs dominate token budget | Tail truncation could drop signal | Tail-truncation keeps the `<final>` token; efficiency term penalizes length **[implemented]** |

---

## 6. Summary

- Each trajectory carries signal (`L_ul`, `L_rb`) that gold-patch SFT cannot
  provide — so our data is not commoditized by existing SWE datasets.
- Predicted threshold: **~300** verified trajectories for first P3 gain; **plan
  for 1,000**. Falsified or confirmed by Exp. 4–5.
- Quality is measurable and acted on (`TrajectoryScorer`, `min_score=0.4`);
  recovery trajectories are ~3× more valuable.
- The flywheel's reality hinges on Exp. 8 (product data improves on synthetic),
  which is the single most important business-defensibility experiment.

## References
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021.
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* 2023.
