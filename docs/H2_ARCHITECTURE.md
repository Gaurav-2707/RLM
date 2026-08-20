# H2 — A Native Recursive Architecture

**Status:** Design proposal. This is the 18-month horizon. Everything here is
**[hypothesis]** unless tagged otherwise; the purpose is to specify a concrete,
falsifiable architecture, not to claim it works.

**Framing.** H0 (Runtime) makes recursion a *behavior* induced by scaffolding.
H1 (RLM-0) makes recursion a *learned behavior* via structured fine-tuning, but
the mechanism — emit tokens, re-read context, retry — is still implemented in the
prompt/loop. H2 asks: **what if retry, verification, and rollback were structural
properties of the architecture itself?**

---

## 1. The Problem With Loop-as-Prompt

The H0/H1 agent loop has three architectural costs:

1. **Context growth.** Each iteration appends state, action, and verdict to the
   context window. A 10-iteration debugging session can consume tens of
   thousands of tokens, most of it stale. Cost and latency scale with iteration
   count, and attention dilutes over irrelevant history.
2. **No true state.** "Memory" is just earlier tokens in the window. There is no
   compact, learned representation of "what I've tried and what I believe."
3. **Rollback is lossy.** In H0/H1, "rolling back" means emitting a token and
   hoping the model conditions correctly on the now-contradicted history. The
   prior wrong activations are still in the KV cache.

H2 targets all three with four mechanisms.

---

## 2. Mechanism 1 — Recurrent State Buffer (RSB)

A fixed-length learnable buffer `S ∈ ℝ^{m×d}` (e.g. `m=256` slots) that persists
*across iterations within a task* and is read/written via cross-attention.

```
   iteration t                         iteration t+1
 ┌───────────────┐                   ┌───────────────┐
 │  transformer  │   write (gated)   │  transformer  │
 │   layers      │ ───────────────▶  │   layers      │
 │      ▲        │      S_t → S_{t+1}│      ▲        │
 │      │ read   │                   │      │ read   │
 │   ┌──┴──┐     │                   │   ┌──┴──┐     │
 │   │ RSB │ S_t │                   │   │ RSB │ S_{t+1}
 │   └─────┘     │                   │   └─────┘     │
 └───────────────┘                   └───────────────┘
```

- **Read:** designated layers cross-attend from token hidden states (queries) to
  RSB slots (keys/values). The model conditions on compressed task state without
  it occupying the token context.
- **Write:** a gated update `S_{t+1} = (1−g)⊙S_t + g⊙Ŝ` where `g` is a learned
  per-slot gate and `Ŝ` is a proposed update pooled from the iteration's hidden
  states. This is the LSTM/GRU update rule lifted to a slotted buffer.

**Relation to prior work [established]:** This is the recurrent-state idea behind
**SSMs/Mamba** (Gu & Dao, 2023) and **RWKV** (Peng et al., 2023), and the
fixed-size latent bottleneck of **Perceiver IO** (Jaegle et al., 2021). The
novelty **[ours, hypothesis]** is using it as a *task-level* working memory
across discrete agent iterations, not as a token-level sequence mixer.

**Falsifiable claim:** an RSB model uses ≤60% of the KV-cache tokens of an
equivalent loop-as-prompt model for matched performance on multi-iteration tasks
(`EXPERIMENTAL_ROADMAP.md`, Exp. 10).

---

## 3. Mechanism 2 — Learned Exit Mechanism

Instead of prompting "are you done?", add a scalar head `e_t = σ(w·h_t^{CLS})`
predicting whether to halt at iteration `t`.

- **Training:** supervise `e_t` with the trajectory's `final_outcome` and the
  index of the actual terminal step. Trajectories that succeeded at step `k`
  teach `e_k → 1` and `e_{<k} → 0`.
- **Inference:** halt when `e_t` exceeds a threshold calibrated by the existing
  **Adaptive Compute Controller (ACC)** conformal gate (Gibbs–Candès ACI,
  already implemented in `acc/conformal.py`). This reuses our calibration
  machinery — the exit head produces the score, ACC sets the threshold with
  coverage guarantees.

This converts halting from a brittle prompt heuristic into a calibrated learned
decision.

---

## 4. Mechanism 3 — Verification Embedding

After each test run, the world produces structured feedback: pass/fail plus the
set of failing test identifiers. Today (H0/H1) this is serialized to text and
re-tokenized. H2 instead encodes it into a dedicated embedding `v_t ∈ ℝ^d`:

- A small encoder maps `(pass_count, fail_count, hash(failing_test_ids))` → `v_t`.
- `v_t` is injected via a cross-attention layer (the model attends to a short
  sequence of verification embeddings, one per iteration).

**Why [hypothesis]:** structured verdicts are low-entropy and highly repetitive;
forcing them through the text tokenizer wastes context and couples the signal to
surface form. A learned verification channel lets the model read "test_foo still
fails" as a stable vector regardless of how the harness phrases it.

---

## 5. Mechanism 4 — Rollback Gate

A gating mechanism that resets a *subset* of state to a checkpoint when the model
emits a rollback decision.

- Maintain a shallow stack of RSB checkpoints `{S^{(0)}, S^{(1)}, …}` pushed at
  each successful verification.
- On rollback, restore `S_t ← S^{(j)}` for the chosen checkpoint `j`, and apply a
  learned forget gate to the token-level KV entries associated with the
  rolled-back edits.

This is the architectural analog of the H1 `<rollback>` token: instead of *hoping*
the model conditions away from contradicted history, we *structurally remove* the
corresponding state. Analogous to an LSTM forget gate, but operating on the RSB
and a checkpoint stack rather than a single cell state.

**Open risk [hypothesis]:** deciding *which* KV entries belong to a rolled-back
edit is nontrivial; a coarse version (clear everything since checkpoint `j`) is a
safe starting point.

---

## 6. Training Objectives for H2

How the H1 objective `L = L_ce + 0.5·L_ul + 0.1·L_rb` translates:

| H1 term | H2 translation |
|---------|----------------|
| `L_ce` (cross-entropy on decision tokens) | Unchanged — still predict next action token. |
| `L_ul` (unlikelihood on final failed action) | Unchanged in spirit; applies to the action emitted before a failed terminal state. |
| `L_rb` (rollback budget) | Re-targets the **rollback gate** activation rather than a token probability: penalize expected gate-open mass above `r̂`. |

New terms H2 likely needs **[hypothesis]**:

1. **Exit-calibration loss** `L_exit`: BCE between `e_t` and the supervised halt
   label, plus a calibration penalty (e.g. ECE-style) so ACC's conformal gate has
   well-behaved scores.
2. **State-consistency loss** `L_sc`: encourage the RSB to be predictive of
   future verdicts — an auxiliary head from `S_t` predicting whether the next
   action will pass. This forces the buffer to actually encode task progress.
3. **Checkpoint-reconstruction loss** (optional): after a rollback, the restored
   state should match the checkpoint it claims to restore (a regularizer on the
   gate).

---

## 7. What Is Speculative vs. Grounded

- **Grounded [established]:** recurrent fixed-size state (SSM/Mamba/RWKV),
  latent bottlenecks (Perceiver IO), gated state updates (LSTM/GRU), conformal
  calibration for thresholds (our ACC).
- **Speculative [hypothesis]:** using these as *task-level* agent memory; the
  rollback gate over a KV checkpoint stack; the verification embedding channel;
  that any of this beats a well-tuned loop-as-prompt at fixed compute.

H2 is a research bet, not a roadmap commitment. The gating decision to pursue it
is Exp. 10 succeeding *and* H1 (Exp. 6) proving the recovery signal matters at
all.

---

## References

- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
  2023.
- Peng et al. *RWKV: Reinventing RNNs for the Transformer Era.* 2023.
- Jaegle et al. *Perceiver IO: A General Architecture for Structured Inputs &
  Outputs.* 2021.
- Gibbs & Candès. *Adaptive Conformal Inference Under Distribution Shift.* 2021.
- Welleck et al. *Neural Text Generation with Unlikelihood Training.* 2019.
