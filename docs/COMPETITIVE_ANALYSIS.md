# Competitive Analysis — Where the Moat Is (and Isn't)

**Status:** Honest positioning. The goal is to be the analysis a skeptical YC
partner would write *for* us, not the one a founder writes *at* them. Where we
have no moat, we say so.

---

## 1. The Honest Picture

| Dimension | RLM Runtime (H0, ships now) | Inference-time agents (SWE-agent, OpenHands, Devin) | RLM-0 (H1, not built) |
|-----------|----------------------------|------------------------------------------------------|------------------------|
| Capability **today** | Runtime delta over base LLM (unproven magnitude) | **Best available today** | Does not exist yet |
| Moat | **None** — scaffolding is copyable | **None** — scaffolding is copyable | Structural training signal + corpus |
| Data position | **Accumulating** verified trajectories | Generally none retained | Trained on the corpus H0 collects |
| Latency | 2–10× base model | 2–10× base model | ~1× (recovery learned, fewer retries) |
| Defensibility source | The data exhaust, not the wrapper | Brand / distribution | Data flywheel + 6-month pipeline lead |

The uncomfortable truth in row two: **at the H0 layer we have no technical
moat.** Any competent team can build an agent loop with rollback. We state this
plainly because the moat is not H0 — it is the *data H0 produces* and the
*pipeline that turns it into H1*.

---

## 2. Where the Moat Actually Is

1. **Proprietary trajectory corpus.** Verified *recursive* trajectories —
   including the failed-then-recovered paths — do not exist in any public
   dataset. SWE-bench gives you destinations (gold patches); it does not give you
   the journey. Our corpus is the journey, verified by execution. Reproducing it
   requires building the Runtime *and* running it at scale to collect data —
   which is the lead time below.

2. **Structural training signal.** The unlikelihood term on failed actions and
   the rollback-budget term (`L = L_ce + 0.5·L_ul + 0.1·L_rb`) only function if
   you possess trajectories containing the rollback→recovery signal. A competitor
   with gold patches but no trajectories *cannot construct these loss terms*. The
   moat is the data–loss coupling, not either piece alone.

3. **Data flywheel with network effects.** Every opted-in product user
   (`contribute_traces=True`) feeds the next model version. More users → more
   verified trajectories → better RLM-0 → better product → more users. This is a
   data network effect, the most durable kind of software defensibility.

4. **Speed-to-trained-model.** The training pipeline exists *today* — rendering,
   collation, three-term loss, QLoRA driver, P1–P4 evaluation, scorer — all
   implemented and tested (104 passing tests). A competitor starting now must
   build all of it *and then* collect data. Conservatively a **6-month** head
   start on the part that compounds.

---

## 3. Competitors, Addressed Honestly

- **GitHub Copilot** — Completion-first, not an autonomous repair agent.
  Different product, different UX. Not a direct competitor to RLM's
  fix-and-verify loop, though it owns the IDE surface we'd want to reach.

- **Cursor / Claude Code** — Excellent agentic coding harnesses. But they are
  *scaffolding over frontier models*; they do not train a model on their own
  recursive traces (and have little incentive to, given their model partnerships).
  We are not trying to out-harness them — we are trying to own the *training
  signal* their category generates and discards.

- **Cognition / Devin** — The strongest autonomy claim, but a black box. Not open
  to study, so our comparison is necessarily indirect. Our falsifiable claim is
  narrow and honest: **on a same-base-model ablation, RLM(model) > raw(model)**
  on SWE-bench Lite. We do not claim to beat Devin's absolute number.

- **SWE-agent (OSS)** — Our primary *measurement* baseline. Strong, reproducible,
  open. We will report our delta against the same base model it uses, so the
  comparison is apples-to-apples rather than confounded by model choice.

---

## 4. The Bear Case (what would kill us)

We list this because pretending it doesn't exist fools no one:

- **Agentless-style results generalize.** If a fixed pipeline keeps matching
  agent loops, the "recursion is valuable" premise weakens, and our recovery
  signal may not justify the complexity.
- **Frontier models internalize recovery on their own.** If GPT-5/Claude-next
  already backtrack well from scale alone, the marginal value of explicitly
  training the behavior shrinks.
- **Synthetic data doesn't transfer.** If OSS-mined trajectories don't help on
  product tasks (Exp. 8 fails), Phase 1 → 2 stalls and the flywheel never spins.

Each is a real risk; each maps to a specific experiment in the roadmap. We would
rather know early.

---

## 5. The VC-Facing Thesis

> The race in coding AI is not about which model is biggest today. It is about who
> builds the feedback loop that compounds fastest. We are the only team building
> training infrastructure that captures **recursive failure dynamics** — the
> rollback-and-recover behavior that is the hardest, scarcest part of real
> software debugging, and the part every existing system throws away.

The product (H0) is the data-collection apparatus. The moat (H1) is what the data
becomes. We are explicit that the moat is prospective, and the roadmap exists to
de-risk it experiment by experiment.
