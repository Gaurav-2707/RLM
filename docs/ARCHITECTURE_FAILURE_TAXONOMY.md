# Architecture Failure Taxonomy
### Why current LLMs structurally cannot solve agentic software engineering, and why recursive training is the fix

**Recursive Labs — Technical Whitepaper v1.0**
*Audience: ML researchers, technical investors. Written to withstand adversarial review.*

> **Epistemic stance.** This document separates three claim types: **[E] established** (uncontested in the literature), **[D] derived** (follows from [E] under stated assumptions), and **[T] thesis** (our bet — defensible, not yet proven at scale). We do not hide the [T]s; the argument is stronger for marking them. The central [T] is falsifiable by the bar in §5.

---

## 1. The Fundamental Problem

### 1.1 What agentic software engineering mechanistically requires

A software task — "fix this failing test," "add this feature," "resolve this issue" — is not a text-generation problem. It is a **closed-loop control problem over a discrete, verifiable state space**. Decomposed mechanistically, success requires five distinct capabilities:

1. **State tracking.** The agent must maintain a faithful model of the *current* repository state — which files exist, what the last edit did, what the test runner just reported — and update it after every action. This is a *maintained* belief, not a *recalled* one (§FM-2).
2. **Multi-step planning under dependency.** Edits are interdependent: changing a function signature obligates updating every call site. The agent must order actions so preconditions hold, i.e. plan over a dependency DAG, not a linear script.
3. **Self-correction.** Actions fail. The agent must detect failure, localize cause, **revert** to a known-good state, and try a different action — a retry/rollback loop, not a forward-only pass.
4. **Calibrated confidence.** To decide *when to stop, when to test, when to ask*, the agent needs `P(this action is correct)` to track reality. A miscalibrated estimate makes the control policy diverge (§FM-3).
5. **Verified outcomes.** "Done" is defined by an *external oracle* — the compiler and the test suite — not by the agent's own judgment. The reward is objective and binary-ish, available for free, and not subject to the agent's persuasion.

### 1.2 Why this is categorically different from next-token prediction

A base LLM is trained to model `P(x_t | x_{<t})` over a text corpus. This objective is:

- **Open-loop.** Generation never observes the consequences of its own output. There is no `observe → act → observe` cycle inside the forward pass.
- **Unverified.** The training signal is *likelihood under a human corpus*, not *correctness under an executor*. Plausible-looking text is rewarded identically whether the code runs or not.
- **Markov-in-context only.** The model's "state" is the KV cache — a function of the literal token prefix. It has no separate, writable, persistent store that survives context eviction (§FM-2).
- **Single-trajectory.** One forward pass produces one path. There is no native mechanism for "that branch failed, restore and branch elsewhere." Backtracking, if it happens at all, must be *simulated in tokens* — and is therefore subject to the same error accumulation it is meant to correct.

**[D]** The five requirements of §1.1 are control-theoretic primitives (state, feedback, recovery, calibration, external reward). The next-token objective optimizes none of them. The mismatch is not a quality gap that scale closes; it is an **objective mismatch**. A model can be superhuman at `P(x_t|x_{<t})` and still lack a closed loop, because the loop is not in the objective. This is the thesis the rest of the taxonomy makes precise.

---

## 2. Failure Mode Taxonomy

Each failure mode is presented as: **root cause → why prompting can't fix it → why RLHF can't fix it → what RLM training does instead.**

A note on the structure of every argument below: prompting changes the *input distribution*, RLHF changes *which outputs of the existing capability set are preferred*, and neither adds a *capability the architecture+objective never trained*. RLM training changes the **objective and the data-generating process** so the capability is in the weights.

---

### FM-1 — Single-Pass Generation Collapse

**Claim.** Greedy/sampled single-pass decoding fails systematically once a task requires more than a few interdependent edits, and the failure is governed by a multiplicative error law.

**The mathematics. [D]** Model a task as `n` interdependent decision points (edits, each of which must be individually correct *and* consistent with the others). Let `p` be the probability the model gets one point right *conditioned on all prior points being right*. Under single-pass generation with no verification or recovery, the joint success probability is

```
P(success) = ∏_{i=1}^{n} p_i  ≤  p_max^n
```

For even a strong `p = 0.95`, the success curve is brutal:

| n (interdependent edits) | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| P(success) at p=0.95 | 0.95 | 0.86 | 0.77 | 0.60 | 0.36 |
| P(success) at p=0.90 | 0.90 | 0.73 | 0.59 | 0.35 | 0.12 |

This is **exponential decay in sequence length**. Worse, real tasks have `p_i` *coupled*: an early wrong edit corrupts the context the model conditions on for later edits, so the *effective* `p_i` degrades as `i` grows — the curve is steeper than the geometric bound. This is the formal content of "error accumulation": single-pass generation has **no mechanism to arrest the cascade**, because each token is conditioned on the model's own (possibly corrupted) prior output with no external correction.

**Why >3 edits is the practical wall.** Empirically `p` per *semantically-loaded* edit on novel code sits well below 0.95. At `p=0.85`, `P(success)` crosses 0.5 around `n=4`. Most non-trivial SWE-bench instances require touching multiple interdependent locations — squarely in the collapse regime.

**Why prompting can't fix it.** Prompting raises `p_i` by some constant factor (better instructions, few-shot exemplars, CoT). But `∏ p_i` is exponential in `n`; a constant multiplier on each `p_i` is overwhelmed by length. You cannot prompt your way out of an exponential with a constant. **[D]**

**Why RLHF can't fix it.** RLHF optimizes for human-preferred *single responses*. It has no notion of a multi-step trajectory with intermediate verification; it cannot reward "recovered on step 7 after failing on step 4" because that trajectory isn't in its data model. It can make each pass *look* better (raising perceived `p`) while leaving the open-loop structure — and thus the `∏` — intact.

**What RLM training does instead.** Replace the single `∏` with a **closed loop that resets the product**. If after each action the agent verifies against the test oracle and rolls back on failure, the relevant quantity is no longer `∏ p_i` but the probability of *eventually* succeeding within a step budget `B`:

```
P(success in ≤ B attempts per point) = ∏_i [ 1 − (1 − p_i)^{k_i} ],   Σ k_i ≤ B
```

Each verified-and-recovered point converts `p_i` into `1 − (1−p_i)^{k_i}`, which → 1 as attempts grow. **Verification + rollback is the only operation that turns an exponential-decay process into a bounded-error one.** RLM-0 is trained on trajectories that *exhibit this loop* (state → action → test → rollback → retry), so the weights internalize "act, check, recover" as the default decoding behavior — not an external scaffold wrapped around an open-loop model.

---

### FM-2 — No Persistent State

**Claim.** The context window is not a state store, and enlarging it does not solve the state problem for long tasks. The relevant distinction is **"has seen" vs. "maintains."**

**Root cause. [E/D]** The only "memory" a transformer has across a task is (a) the token sequence in context and (b) the KV cache derived from it. Both are:

- **Append-mostly and read-only-ish.** The model cannot *edit* a prior belief in place; it can only append new tokens that *contradict* old ones and hope attention prefers the new. There is no `state[x] = y` write primitive.
- **Subject to eviction.** Long tasks overflow any finite window; older state is dropped or summarized lossily.
- **Positionally degraded.** "Lost in the middle" effects **[E]** mean retrieval reliability is non-uniform across the window; a fact 80k tokens back is not equally available as one 2k back.
- **Undifferentiated.** Critical state (the current diff, the last error) sits in the same undifferentiated token stream as noise (verbose logs), with no typed, addressable structure.

**"Has seen" vs. "maintains."** A model *has seen* a fact if it appeared in context. A model *maintains* a fact if it can be relied on to read the *current, correct* value on demand and update it after a change. Context gives you the former. Agentic control needs the latter. **[D]** A long context makes more tokens *visible*; it does not give the model a *consistent, writable, queryable state object*. These are different data structures — a log vs. a database — and no amount of log length yields a database's guarantees (point lookup, in-place update, consistency).

**Why prompting can't fix it.** "Keep track of the state in a scratchpad" pushes state *into the token stream* — the exact substrate that is append-only, evictable, and positionally degraded. The scratchpad is itself subject to FM-1 corruption. You are storing your database in the log you said was unreliable.

**Why RLHF can't fix it.** RLHF reweights outputs; it does not add a memory subsystem or change what survives context eviction. It cannot reward "maintained the invariant across 40 steps" because it operates on single responses.

**What RLM training does instead.** Two layers. (1) *Product layer:* RLM Runtime provides **episodic memory** — a typed, addressable, persistent store outside the context window (retrieve-on-demand, write-after-action). (2) *Weights layer:* RLM-0 is trained on trajectories where the `input_state` is an *explicit, refreshed* state snapshot at each step and the model's actions are conditioned on it. The model learns the *behavior* of "consult current state, act, write back" rather than relying on context persistence. The state lives in a store with database semantics; the model learns to use it. **[T]** that this behavior transfers — falsified/confirmed by the maintained-invariant metric in §5.

---

### FM-3 — Miscalibrated Confidence (the overconfidence trap)

**Claim.** RLHF systematically *destroys* calibration on objectively verifiable tasks, and miscalibration makes self-correction structurally impossible.

**Root cause. [E]** Base LLMs are often *reasonably* calibrated post-pretraining (next-token probabilities track frequencies). **RLHF degrades calibration** — this is documented (e.g. GPT-4 technical report: the post-RLHF model is markedly *less* calibrated than the base model). The mechanism **[D]**: preference optimization rewards *confident, decisive, helpful-sounding* answers. Hedging is dispreferred by raters. The policy is pushed toward sharp, high-confidence outputs *regardless of correctness*, because the reward model scores *style and assertiveness*, not *verified accuracy*. Confidence and correctness are decoupled by the training signal.

**Why this kills self-correction. [D]** Self-correction is a *decision* gated on an internal error signal: "I am probably wrong → revert/retry." That decision requires `P(correct)` to be **low when the agent is in fact wrong**. An overconfident model reports `P(correct) ≈ 0.95` *even on its errors*, so the revert/retry trigger never fires. The control loop has a broken sensor. You cannot build a thermostat on a thermometer that always reads 72°. This is why "please double-check your work" fails: the model *does* check, concludes (confidently) that it's right, and proceeds. The prompt cannot inject a calibrated signal the weights don't produce.

**Why prompting can't fix it.** Verbalized confidence elicited by a prompt inherits the same miscalibrated distribution; asking "how sure are you?" samples the same overconfident policy. Calibration is a property of the *output distribution*, set by training — a prompt reweights inputs, not the distribution's reliability.

**Why RLHF can't fix it.** RLHF is the *cause*. More RLHF, RLAIF, or Constitutional AI all use a *preference* signal (human or model) that rewards assertive helpfulness. Same signal, same overconfidence. **[D]** You cannot fix a calibration failure induced by preference-on-style with more preference-on-style.

**What RLM training does instead.** Train confidence against **ground truth**. In our trajectory schema, each step carries a `confidence` value *and* an objectively verified `outcome` (test pass/fail). RLM-0 is trained so that emitted confidence (discretized into buckets) is supervised by *whether the action actually passed tests*. Calibration becomes a **measured, optimized quantity** (Expected Calibration Error against test outcomes), not a stylistic by-product. The error sensor is built against the executor, the one judge that cannot be charmed. **[D]** that this is the *only* signal that can produce calibration, because calibration is by definition agreement between stated confidence and *real* outcomes, and only the executor provides real outcomes for free at scale.

---

### FM-4 — Hallucinatory Self-Evaluation

**Claim.** A model cannot reliably verify its own output using the same weights that produced it, and chain-of-thought "checking" fails for a mechanistic reason.

**Root cause. [D]** Self-evaluation asks the model to compute `is_correct(y)` where `y` was sampled from the *same* distribution `p_θ`. If the model assigned high probability to a wrong `y`, that is *because* its internal model judges `y` plausible. Asked to evaluate `y`, it consults the **same internal model** and finds `y`... plausible. The error and its detection are **correlated through shared weights**: the failure mode of generation is the failure mode of evaluation. A model's blind spots in producing code are *exactly* its blind spots in reviewing it. There is no independent estimator. **[D]**

**Why CoT "checking" fails mechanistically.** Chain-of-thought verification ("let me trace through this function...") is *more generation from the same policy*, conditioned on the thing it's evaluating. It is subject to: (a) the same knowledge gaps (if the model doesn't know the API contract, it mis-traces it the same way it mis-wrote it); (b) **self-consistency pressure** — having just emitted `y`, the conditional `P(y is correct | y, "let me check")` is biased upward because the context now contains `y` asserted as the model's answer; the model is primed to ratify, not refute. CoT checking measures *internal coherence*, not *external correctness*. Coherent-and-wrong is the dangerous quadrant, and CoT cannot exit it. **[E]** self-consistency and confirmation effects in LLM self-critique are observed empirically.

**Why prompting can't fix it.** Any prompt-elicited self-check is sampled from `p_θ` conditioned on `y`. You cannot prompt a distribution into becoming independent of itself.

**Why RLHF can't fix it.** RLHF tunes `p_θ`; the self-evaluation still runs on `p_θ`. The correlation between generation errors and evaluation errors is structural, not preference-tunable.

**What RLM training does instead.** Replace self-evaluation with **external verification**, and train the model to *defer to it*. The compiler/test suite is an estimator **statistically independent** of `p_θ`: it judges `y` by execution, not by plausibility under the model. RLM trajectories make the *test result* the pivot of the loop — the model's job is not to *judge* correctness but to *act, submit to the oracle, and respond to its verdict*. The architectural fix is to stop asking the model to be its own judge. **[D]** Independence of the verifier from the generator is the whole game; only an executor provides it.

*(This is also why our P3 "feedback-ablation" metric (§5) is decisive: if removing the real verdict doesn't hurt the model, it was self-evaluating; if it does, the model genuinely uses the independent signal.)*

---

### FM-5 — Reward Hacking via Fluency

**Claim.** RLHF models preferentially emit fluent-wrong over awkward-right, and this directly destroys agentic reliability.

**Root cause. [D]** The RLHF reward model is trained on *human preference judgments*. Humans, especially under rating-time constraints, are **systematically biased toward fluency, confidence, and surface coherence** — they cannot execute the code, so they reward what *reads* correct. The policy therefore maximizes `E[reward_model(y)]`, which is `E[looks-correct-to-a-human(y)]`, *not* `E[is-correct(y)]`. When these diverge — fluent-wrong vs. awkward-right — the gradient points at **fluent-wrong**. This is textbook reward hacking / proxy-objective failure: optimizing the proxy (perceived quality) at the expense of the target (actual correctness). **[E]** documented as sycophancy and stylistic reward hacking.

**Why this is fatal for agents specifically. [D]** In a multi-step loop, a fluent-wrong action is *worse than an obviously-broken one*, because:

1. It passes the model's own (correlated) self-check (FM-4) — it *looks* right.
2. It inflates confidence (FM-3) — it *feels* right.
3. So no rollback fires, and the error is committed to state (FM-2) and compounded by subsequent steps (FM-1).

Fluency is precisely the property that *evades every internal error signal*. An agent built on a fluency-maximizing policy is optimized to make the **hardest-to-detect** mistakes. The failure modes are not independent; FM-5 is the amplifier that makes FM-1–FM-4 lethal in composition.

**Why prompting can't fix it.** Prompting cannot change which output the *reward-shaped weights* prefer. "Be correct, not just fluent" is a request to a policy whose maximum is fluency.

**Why RLHF can't fix it.** It is the source. Scaling the reward model scales the *proxy*, not the target. RLAIF/Constitutional AI swap human raters for a model rater trained to *imitate* the same preference distribution — same proxy, possibly sharper. **[D]**

**What RLM training does instead.** Make the reward the **target itself**: test-pass is `+1`, test-fail is `−0.5`, with `final_outcome` gated on real verification. There is *no proxy*. Fluent-wrong code fails the tests and is penalized identically to ugly-wrong code; awkward-right code passes and is rewarded. The fluency channel is severed from the reward channel. We additionally apply **unlikelihood** on verified-failing actions, actively pushing probability mass *away* from outputs the executor rejected. The model cannot hack a reward that is computed by running the code. **[D]**

---

### 2.x How the failure modes compose

These are not five bugs; they are one structural deficiency viewed from five angles. The composition is the point:

```
FM-2 (no maintained state) ─┐
                            ├─► errors enter and persist
FM-1 (open-loop ∏ p_i) ─────┘
                            
FM-5 (fluent-wrong preferred) ─► errors are maximally camouflaged
FM-3 (overconfidence) ─────────► error sensor reads "fine"
FM-4 (self-eval correlated) ───► second opinion is the same opinion
                            
        ⇒ no internal signal can fire a correction
        ⇒ the cascade of FM-1 runs unchecked to failure
```

Removing any one is insufficient (fix calibration but keep open-loop generation and you still get FM-1 collapse). The fix must be **simultaneous and architectural**: a closed loop, an external verifier, calibrated-against-truth confidence, persistent state. That is the definition of an RLM.

---

## 3. Why Existing Solutions Don't Fix This

The unifying error of every current approach: **they treat the loop as something you build *around* the model, leaving the model open-loop, self-judging, overconfident, and fluency-hacked.** Orchestration cannot install a capability the weights lack; it can only call the weights more times.

**Devin / SWE-agent / OpenHands (agent orchestration).** What they *do*: a scaffold (planner, tool-use harness, retry logic, a test runner) that wraps a frozen base LLM in an external loop. This is real and useful — it adds the *outer* loop the base model lacks. What they *cannot fix*: every call inside the loop still hits a model with FM-1–FM-5. The orchestrator can re-prompt after a failure, but the model that produced the fluent-wrong, overconfident, self-ratified action is *the same model on the retry*. The scaffold's rollback is not *internalized*; the model does not *learn* to recurse, so it repeats correlated mistakes and burns the step budget. **[D]** Orchestration multiplies attempts; it does not raise per-attempt independence or calibration. (Our own product *is* such an orchestrator — RLM Runtime — but we treat it as the **data-collection flywheel**, not the end state. The orchestrator's job is to generate the verified trajectories that train the loop *into the weights*. §4.)

**Cursor / Copilot (assistive completion).** Correctly framed, these solve a *different, easier* problem: **human-in-the-loop micro-edits** where the developer *is* the closed loop — the human provides state tracking, error detection, rollback, and verification. They are excellent at it and we make no claim against them. But the human is doing FM-1–FM-5 mitigation. Remove the human (full autonomy) and the deficiencies reappear in full. They don't solve agentic autonomy; they *augment* a human who already has the loop.

**Better prompting / chain-of-thought.** Attacks the *symptom*. CoT raises per-step `p` modestly and makes reasoning legible, but (a) it cannot beat the exponential of FM-1 with a constant factor, and (b) CoT *self-checking* is FM-4 by construction — generation from the same policy, primed to ratify. Prompting changes the input distribution; the pathologies are properties of the output distribution and training objective.

**More parameters / "GPT-5" / scale.** Scale raises `p_i` and broadens knowledge — genuinely helpful for FM-1's *base rate*. But FM-3, FM-4, FM-5 are **not capability deficits; they are objective-induced biases.** A bigger model trained with the *same* next-token-then-RLHF pipeline is *more* fluent (worse FM-5 camouflage), often *more* confidently wrong (FM-3), and its self-evaluation is *still* correlated with its generation (FM-4 — same weights). Scale moves the base rate; it does not change the *sign* of the objective mismatch. **[D]** You cannot scale your way from an open-loop objective to a closed-loop capability.

**Constitutional AI / RLAIF.** Same training-signal problem, one level removed. The "constitution" or AI feedback is still a **preference/plausibility signal**, now emitted by a model that learned human preferences. It rewards principled-*sounding* outputs judged by plausibility, not execution. It cannot produce calibration-against-truth (FM-3) or sever fluency from reward (FM-5) because **no executor is in the loop**. The ground truth is still a judgment, not a test. **[D]**

**The through-line.** Every approach above is missing the same ingredient: a **training signal grounded in objective execution** that installs the loop **in the weights**. That is the gap RLM training fills.

---

## 4. The RLM Thesis

### 4.1 Objectively verified execution trajectories are the correct training signal

For agentic software tasks, ground truth **exists and is cheap**: the compiler and test suite are an oracle that returns a verified, near-binary correctness signal at the cost of a CI run. This is categorically better than human preference:

| Property | RLHF signal (preference) | RLM signal (execution) |
|---|---|---|
| Ground truth? | No — a proxy for correctness | **Yes — correctness itself** |
| Hackable by fluency? | Yes (FM-5) | **No — code is run** |
| Calibration target? | Style/assertiveness | **Actual pass/fail (FM-3 fixable)** |
| Independent of generator? | No (rater bias, or RLAIF = same model) | **Yes (executor ⫫ p_θ) (FM-4 fixable)** |
| Cost to label | High (humans) | **~free (CI), and self-generating** |
| Rewards the loop? | No (single response) | **Yes (full trajectory: act→test→rollback→retry)** |

**[D]** For any task with an automatable verifier, an execution-grounded signal *dominates* a preference signal on every axis that FM-3/4/5 depend on. Software engineering is the canonical such domain — which is why it is the right wedge.

### 4.2 Recursive behavior must be in weights, not orchestration — the CUDA analogy

You do not achieve GPU parallelism by writing a Python `for` loop that *calls* the GPU once per element. Parallelism has to be expressed in the execution substrate (CUDA kernels), not simulated in the host language above it. **The host-language loop is orchestration; the kernel is architecture.**

Agentic recursion is identical. Wrapping a frozen LLM in a Python control loop (Devin, SWE-agent) is the host-language `for` loop: it *invokes* a non-recursive primitive repeatedly and pays full overhead per call (re-deriving state from scratch, repeating correlated errors, no learned recovery). Native recursion — retry, rollback, calibrated confidence, state maintenance *as default decoding behavior* — must live **in the weights**, the way parallelism lives in the kernel. **[T]** This is the company's core bet: the loop belongs in the substrate, and a model with the loop in-weights will dominate any orchestration-over-frozen-model stack on reliability per unit compute, for the same reason kernels dominate host-loops.

### 4.3 What RLM-0 proves *before* the native architecture

RLM-0 is QLoRA-on-Llama-3.1-8B — deliberately *not* a new architecture. Its purpose is a **falsifiable existence proof**: that recursive behavior (rollback-on-failure, calibrated confidence, feedback-driven recovery) can be **internalized into weights via execution-grounded SFT**, such that the model exhibits the loop *without an external scaffold instructing it to*. If an 8B LoRA can demonstrably move the §5 metrics — especially **feedback-value** (the model's recovery collapses when the real verdict is hidden) and **calibration-against-tests** — then the thesis (§4.2) is validated *in principle*, and the native architecture is an optimization of a proven idea, not a leap of faith. RLM-0 de-risks the architecture by proving the *signal* and the *behavior* first. **[T→testable]**

### 4.4 The flywheel

```
   RLM Runtime (product) ── runs real SWE tasks for real users
              │
              ▼
   Every run, test-verified, becomes a trajectory  ──►  (state, action, test, rollback, reward)
              │                                              objectively labeled, ~free
              ▼
   Trajectories train RLM-0 (and successors)
              │
              ▼
   Better model ──► better Runtime ──► more usage ──► more trajectories ──┐
              ▲                                                            │
              └────────────────────────────────────────────────────────  ┘
```

**Why this beats any human-labeling pipeline. [D]** (1) *Correctness:* labels are execution-verified, not human-guessed — no FM-5 leakage into the training set. (2) *Cost & scale:* labels are produced as a *byproduct of usage*, not a separate annotation spend; marginal cost ≈ a CI run. (3) *Distribution:* the data is *exactly* the distribution of real tasks users bring, not a curated benchmark — it self-corrects toward where the model is actually used. (4) *Hard-negative richness:* failed-and-recovered trajectories (the rollback data that *no static dataset contains*) are generated naturally, and these are the single most valuable examples for teaching recursion. A human pipeline can match *none* of these four simultaneously. The flywheel's data is both cheaper and *higher quality* than anything money can label — the rare case where the economical signal is also the superior one, because the executor is both free and correct.

---

## 5. The Quantitative Bar

We commit to falsifiable thresholds. RLM-0 has internalized recursive behavior **iff** it clears all of the following on a **held-out task distribution** (repos/issues absent from training — generalization, not recall). These map directly to the P1–P4 "recursion proof" protocol in `training/eval.py`.

### 5.1 Recursion-proof metrics (the load-bearing claims)

| # | Metric | Threshold | What it rules out |
|---|---|---|---|
| **M1** | **Feedback-value** = RecoveryRate(with verdict) − RecoveryRate(verdict ablated) | **≥ +0.25** | Self-evaluation illusion (FM-4). If hiding the *real* test result barely changes behavior, the model never used the independent signal. **This is the single most decisive number.** |
| **M2** | **Recovery Rate** — of tasks whose *first* action fails, fraction driven to verified PASS | **≥ 0.50** | Open-loop collapse (FM-1). Measures the closed loop directly. |
| **M3** | **Rollback precision** (rolled-back states that were genuinely broken / all rollbacks) | **≥ 0.70** | "Always roll back" degeneracy. Proves rollback is *discriminative*, not a tic. |
| **M4** | **Verdict-flip response** — on a forced FAIL after a good action, P(model reacts: retry/rollback) | **≥ 0.80** | Confirms the model is *causally* conditioning on the verdict token, not on content. |
| **M5** | **Calibration (ECE vs. test outcomes)** | **≤ 0.10** | Overconfidence trap (FM-3). Confidence must track *real* pass rates. |
| **M6** | **Steps-to-success** (median, novel tasks) | **≤ data median** | Memorization. A recurser solves novel tasks *at least as efficiently*; a memorizer regurgitates and is slower off-distribution. |

**Distinguishing learned recursion from luck/memorization** is the entire purpose of M1, M4, and M6. M1/M4 are *causal interventions* on the feedback channel (a memorizer keyed on surface content is invariant to them; a recurser is not). M6 is an off-distribution efficiency test (memorization degrades, capability transfers). Passing accuracy alone proves nothing; passing M1+M4+M6 *together* is hard to fake by memorization. **[D]**

### 5.2 Comparison to SWE-bench SOTA

SWE-bench Verified is reported as a single number: **% of issues resolved (test-verified)**. As of this writing, top agent scaffolds report roughly the **65–75%** range on Verified, achieved by **strong frontier models inside heavy orchestration** (the §3 "host-loop" pattern).

Our claim is **not** "RLM-0 (an 8B LoRA) beats a frontier model's absolute resolve-rate." That would be dishonest — 8B will not out-resolve a top frontier model in raw capability. Our claim is sharper and more defensible:

- **[T] Efficiency-of-recursion claim.** On the *recovery subset* (tasks requiring ≥1 failed-then-corrected step), RLM-0 achieves a **higher Recovery Rate per unit inference compute** than the *same base model (Llama-3.1-8B) wrapped in an equivalent external scaffold.** I.e. holding the base model fixed, putting the loop *in the weights* beats putting it in the *orchestration* — the CUDA-vs-host-loop claim, measured. This is an **apples-to-apples ablation** (same weights ± LoRA, same harness) and is the honest way to prove §4.2.
- **[T] Signal claim.** RLM-0's **feedback-value (M1)** and **calibration (M5)** exceed those of *any* RLHF-tuned model of comparable size on the same protocol — because those quantities are *defined by* the execution signal RLHF lacks (FM-3/FM-4). We expect a *large* gap here, not a marginal one.

**The proof obligation we accept:** publish M1–M6 on held-out tasks, plus the same-base-model scaffold ablation, with the eval harness open (`training/eval.py`). If RLM-0 fails M1 (feedback-value < 0.25), the thesis is wrong and we will say so. We are putting a falsifiable number on the central claim — which is more than any "scale fixes it" position does.

---

## Appendix: Assumptions and honest limits

- **A1.** The verifier (test suite) is a *sound* correctness oracle for the task. Where tests are weak/missing, the RLM signal degrades to the quality of the available verifier — a real limit, mitigated by test-generation (`engine/test_gen.py`) but not eliminated. We do not claim to fix tasks with no checkable success criterion.
- **A2.** The geometric model of FM-1 (`∏ p_i`) is a *lower-bound abstraction*; real `p_i` are coupled and the true curve is task-dependent. The *direction* (exponential collapse without a loop) is robust; the exact crossover `n` is illustrative.
- **A3.** FM-3's "RLHF degrades calibration" is established for current frontier models **[E]**; we assume it generalizes to the RLHF-tuned models we compare against.
- **A4.** §4.2/§4.3/§5.2 claims marked **[T]** are bets validated *only* by clearing §5. We present them as the strongest *defensible* case, explicitly falsifiable — not as settled results.
- **A5.** "Categorically cannot" in the title is shorthand for "cannot *under the standard next-token-then-preference objective*, at any scale." It is an objective-mismatch claim, not a claim that transformers are forever incapable — *with the right objective (ours), they can.*
