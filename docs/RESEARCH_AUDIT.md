# Research Audit — RLM in the Landscape of Agentic Code Repair

**Status:** Living document. Claims are labeled **[established]** (supported by
published results), **[ours]** (our design decision), or **[hypothesis]**
(believed but not yet demonstrated).

**Framing (read first).** Recursive Labs builds *training infrastructure for
agentic models*. We are not claiming state-of-the-art on SWE-bench today. Our H0
product (RLM Runtime) is inference-time scaffolding, in the same family as the
systems below. Our differentiated bet is H1 (RLM-0): a model fine-tuned on the
*recursive process itself* — the rollback→recovery dynamics that scaffolding
discards. This document situates that bet against the literature.

---

## 1. Introduction

The dominant paradigm in LLM-based software engineering is the **agent loop**: an
LLM is given tools (shell, file editor, search), observes execution feedback, and
iterates until it submits a patch. SWE-bench (Jimenez et al., 2023) made this
measurable by scoring FAIL→PASS resolution on real GitHub issues.

Two observations motivate RLM:

1. **The agent loop is stateless across failures.** When a run fails, the next
   run starts cold. The information contained in *how* the model failed — which
   edit broke which test, when it should have backtracked — is thrown away.
2. **Training signal is almost always the destination, never the journey.**
   Systems that fine-tune (e.g. SWE-Llama) learn from `(issue → gold patch)`
   pairs. They learn what a correct patch *looks like*, not how to *recover* from
   a wrong one.

RLM's thesis **[ours]**: the recovery trajectory is the most valuable and most
scarce training signal for agentic competence, and no existing dataset captures
it.

---

## 2. Related Work

| System | Mechanism | Learns from failure? | Rollback? | Relation to RLM |
|--------|-----------|----------------------|-----------|-----------------|
| **SWE-agent** (Yang et al., 2024) | Agent–Computer Interface (ACI): custom bash/edit tools tuned for LM ergonomics | No — each episode stateless | No explicit primitive | Our baseline + comparison target |
| **CodeAct** (Wang et al., 2024) | LM emits executable Python actions; interpreter feedback in-context | No | No | Action space inspiration; we add structured control tokens |
| **OpenDevin / OpenHands** (Wang et al., 2024) | Multi-agent scaffolding + sandbox | No (orchestration, not weights) | Via re-planning, not a primitive | Product-level comparator |
| **Agentless** (Xia et al., 2024) | Fixed localize→repair→validate pipeline, no agent loop | No | N/A | Shows scaffolding ≠ necessity; pressures our "why agent?" answer |
| **SWE-Llama** (Jimenez et al., 2023) | SFT on (issue → gold patch) | Only final patches | No | Closest *training* comparator; learns destination not journey |
| **RLEF** (Gehring et al., 2024) | RL with execution feedback on code | Yes — but scalar terminal reward | Implicit in policy | **Closest related work**; see §3 |

### The RLEF comparison (most important)

RLEF (Reinforcement Learning from Execution Feedback) is the nearest neighbor: it
optimizes a policy against execution outcomes rather than human preference. The
distinction **[ours]**:

- RLEF compresses an episode into a **scalar terminal reward**. The credit
  assignment back to individual decisions is left to the RL algorithm's value
  estimator.
- RLM renders the episode into a **structured token sequence** with explicit
  roles: world-supplied spans (state, test verdicts — not trainable) vs.
  model-decision spans (actions, rollbacks, confidence — trainable). The learning
  signal is *localized to the decision tokens*, and a specific unlikelihood term
  targets the final failed action. We are doing supervised structure-imposition
  on the trajectory, not pure RL.

This makes RLM's signal denser and more stable than a scalar reward (no
high-variance policy-gradient estimation) while still being grounded in
objective execution outcomes.

---

## 3. Our Genuine Differentiators

All **[ours]**, and all contingent on H1 (none are claimed for the H0 Runtime):

1. **Training signal from the recursive process.** We render verified
   trajectories — including the rollbacks — into the training sequence. The
   corpus is `(state, decision, verdict)*` chains, not `(issue, patch)` pairs.

2. **Unlikelihood loss on the final failed action.** Following Welleck et al.
   (2019) on unlikelihood training, we apply a UL term **only** to the last
   action of a *failed* trajectory. The model is pushed away from the specific
   dead-end move, not from failing in general. (Applying UL to every failed step
   would punish productive exploration — a deliberate design choice.)

3. **Rollback as a learnable discrete action.** `<rollback>` is one of 26 control
   tokens in the vocabulary, mean-initialized into both the embedding and the LM
   head. The model learns *when* to backtrack as a first-class prediction, not
   as a meta-instruction injected by a harness.

4. **Rollback budget regularization.** An auxiliary loss term (`L_rb`,
   weight 0.1) penalizes expected rollback mass above a calibrated budget `r̂`,
   preventing the degenerate policy of rolling back constantly.

The combined objective is `L = L_ce + 0.5·L_ul + 0.1·L_rb`. The two auxiliary
terms have **no analog in SFT-on-gold-patches** and only a high-variance scalar
analog in RL-from-reward.

---

## 4. Limitations (stated plainly)

- **[established for us]** The H0 RLM Runtime does not train anything. It is
  inference-time scaffolding and is, by itself, as copyable as SWE-agent. Its only
  defensible output is the *trajectory data* it collects.
- **[hypothesis]** The structural training signal yields a measurable capability
  gain. This is unproven until the first RLM-0 fine-tuning run (see
  `EXPERIMENTAL_ROADMAP.md`, Exp. 4 and 6).
- **[hypothesis]** Synthetic trajectories from open-source bug-fix commits are
  in-distribution enough to transfer to product tasks. We mitigate distribution
  mismatch via the novelty term in `TrajectoryScorer`, but it is not yet
  validated.
- **Agentless (Xia et al.)** is a standing rebuke: if a fixed pipeline matches
  agent loops on SWE-bench Lite, our "the loop is valuable" premise needs the
  recovery-signal evidence to hold up. We treat this as the experiment to win,
  not an assumption.

---

## 5. Open Questions

1. **How much recovery signal is enough?** What fraction of trajectories must
   contain a genuine rollback→recovery before `L_ul`/`L_rb` change behavior?
   (Quantified as a hypothesis in `DATA_FLYWHEEL_ANALYSIS.md`.)
2. **Does the rollback token transfer across repos?** Or does the model learn
   repo-specific backtracking heuristics that don't generalize?
3. **Can H2 internalize the loop?** If recovery becomes structural (see
   `H2_ARCHITECTURE.md`), do we still need explicit rollback tokens, or do they
   become a training-time scaffold we can remove at inference?
4. **Is verified-outcome signal subject to reward hacking?** A model can "pass"
   by deleting tests. Our `PASS_TO_PASS` guard in evaluation addresses the
   measurement; whether the training signal itself is gameable is open.

---

## References (as cited; verify exact venues/years before external publication)

- Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub
  Issues?* 2023.
- Yang et al. *SWE-agent: Agent–Computer Interfaces Enable Automated Software
  Engineering.* 2024.
- Wang et al. *Executable Code Actions Elicit Better LLM Agents (CodeAct).* 2024.
- Wang et al. *OpenDevin/OpenHands: An Open Platform for AI Software Developers.*
  2024.
- Xia et al. *Agentless: Demystifying LLM-based Software Engineering Agents.*
  2024.
- Gehring et al. *RLEF: Reinforcement Learning from Execution Feedback.* 2024.
- Welleck et al. *Neural Text Generation with Unlikelihood Training.* 2019.
