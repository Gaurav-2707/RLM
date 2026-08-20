# Website Brief — Recursive Labs
### For Codex. Cursor-style. Ship something that stops the scroll.

---

## 0. The Vibe

Reference sites: **cursor.sh**, **linear.app**, **vercel.com**, **resend.com**.

Dark. Fast. Dense with signal. Every pixel earns its place. Animations serve
the message — they don't decorate it. Someone who lands here should feel like
they've found the serious team working on the serious version of this problem.

**Stack:** Next.js 14 + Tailwind + Framer Motion. Deploy on Vercel.

**Palette:**
```
bg-primary:    #050505   (near black)
bg-surface:    #0f0f0f   (card bg)
bg-elevated:   #1a1a1a   (hover states)
text-primary:  #f5f5f5
text-muted:    #6b7280
accent-blue:   #3b82f6
accent-glow:   #60a5fa   (for glows/gradients)
accent-purple: #8b5cf6   (secondary gradient stop)
border:        #ffffff10 (barely-there borders)
```

**Typography:**
- Headlines: `Geist` or `Inter`, 700–900 weight
- Body: `Inter`, 400–500
- Code/numbers: `JetBrains Mono` or `Geist Mono`

**Motion principles (Framer Motion):**
- Fade-up on scroll entry: `y: 30 → 0, opacity: 0 → 1, duration: 0.6`
- Stagger children: `0.1s` delay between siblings
- Hover on cards: `scale: 1.02, duration: 0.2`
- No bounce. `ease: [0.25, 0.1, 0.25, 1]` everywhere.
- Hero elements: animate in on mount, not on scroll

---

## 1. Navigation

**Sticky. Blur backdrop. Height 60px.**

```
[RL]  Recursive Labs          Problem  Solution  Research  Team     [Get Early Access →]
```

- Logo: `RL` monogram in accent blue, then "Recursive Labs" in white
- Nav links: fade to white on hover, muted by default
- CTA button: `bg-blue-600 hover:bg-blue-500`, subtle glow on hover
  (`box-shadow: 0 0 20px rgba(59,130,246,0.4)`)
- On mobile: hamburger → full-screen overlay menu
- Background: `backdrop-blur-md bg-black/60` after 20px scroll

---

## 2. Hero Section

**Full viewport height. This is the money shot.**

### Background
- Subtle radial gradient: `from purple-900/20 via transparent to transparent`,
  positioned top-center
- Animated floating orbs (Framer Motion, very slow, 20–30s loops):
  - One large (~600px) blurred blue orb, top-left, `opacity: 0.08`
  - One medium (~400px) blurred purple orb, bottom-right, `opacity: 0.06`
- Fine dot grid overlay (`bg-dot-pattern`), `opacity: 0.3`

### Content (centered, max-w-4xl)

**Eyebrow label** (animate in first, small, accent blue, monospace):
```
Context OS for AI Coding Agents
```

**Main headline** (animate in second, ~72px desktop / ~44px mobile, 900 weight):
```
Your AI forgot
the bug it just saw.
```

**Gradient treatment:** "forgot" gets a gradient text:
`bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent`

**Sub-headline** (animate in third, ~20px, text-muted, max-w-2xl, line-height 1.7):
```
Claude compacted your context. The exact error — gone. The patch that
almost worked — gone. So the model tries the same broken approach again.

RLM is the context layer that eliminates this entirely.
Semantic retrieval instead of compression. Every failure remembered.
12× fewer tokens. Zero information lost.
```

**CTA row** (animate in fourth):
```
[  Get Early Access  →  ]        [ Read the Research  ↗ ]
   (blue, glowing)                  (ghost, muted)
```

**Stat chips row** (animate in last, monospace, small, horizontal):
```
[ 12×  token compression ]  ·  [ 0  info lost ]  ·  [ ~0ms  retrieval ]  ·  [ 122  passing tests ]
```
Style: `border border-white/10 bg-white/5 rounded-full px-4 py-1.5 text-sm`

### Hero Visual

Below the copy, a **floating terminal card** (Framer Motion: enter from below,
slight `rotateX(5deg)` for depth, subtle continuous `y` float animation ±6px
on a 4s loop):

```
┌─────────────────────────────────────────────────────────────────┐
│  ● ● ●   rlm run "fix the KeyError in auth.py"                  │
│                                                                  │
│  Context OS active                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  tokens_served              2,847    ← 41,200 avoided    │   │
│  │  compression_ratio          14.5×                        │   │
│  │  compaction_calls_avoided   5                            │   │
│  │  failure_warnings           1 ─────────────────────────┐│   │
│  └─────────────────────────────────────────────────────────┘│   │
│    "prior approach to auth.py failed — avoid length checks"  │   │
│  ───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  [■■■■■■■■■■░░] Step 3/5  rollback → retry with hmac.digest    │
│                                                                  │
│  ✓ Tests pass. Trajectory saved.                                 │
└─────────────────────────────────────────────────────────────────┘
```

Style the terminal card:
- `bg-[#0f0f0f] border border-white/10 rounded-2xl`
- Top bar with three dots (red/yellow/green)
- Key values (`14.5×`, `2,847`, `✓`) in accent blue
- `rollback` in amber/orange
- `failure_warnings` in soft red
- Typewriter animation ONLY on the last line (`✓ Tests pass.`) — everything
  else is already "rendered"

---

## 3. Social Proof Bar

**Thin bar, full width, below hero. Subtle.**

```
────────────────────────────────────────────────────────────────
  The only training infrastructure that captures recursive
  failure dynamics — the part every existing system throws away.
────────────────────────────────────────────────────────────────
```

Centered italic quote, text-muted. No attribution. Just the line.

---

## 4. The Problem Section

**Scroll-triggered. Dark section with subtle top border.**

### Section label (small, uppercase, letter-spaced, accent blue):
```
THE PROBLEM
```

### Title (~52px, bold):
```
Compaction is silently
corrupting your AI's memory.
```

**"corrupting"** gets the gradient treatment.

### Three cards (grid, scroll-stagger animation, glass morphism):

**Card style:**
```css
background: rgba(255,255,255,0.03);
border: 1px solid rgba(255,255,255,0.08);
border-radius: 16px;
backdrop-filter: blur(10px);
```

**Card 1 — The Trigger**
Icon: a clock or context window icon (Lucide)
Title: `It happens every 8,000 tokens`
Body:
> Claude Code compacts. GPT slides the window. Cursor truncates.
> The threshold hits mid-task, mid-thought, mid-debugging session.
> The model's next response starts from a lie.

**Card 2 — The Loss**
Icon: broken file icon
Title: `The wrong things get summarised`
Body:
> Line numbers. Stack traces. The file state before the last edit.
> The patch that failed and why. These are not summarisable.
> A summary of "there was an error" is useless. The error is the signal.

**Card 3 — The Cost**
Icon: token/coin icon
Title: `You pay for the damage twice`
Body:
> First: the compaction call itself — tokens, latency, money.
> Then: the model re-explores paths it already failed.
> A 50-turn session: **$4.69** naive. **$0.37** with RLM.

**Below cards — animated comparison block:**

Two columns side by side. Numbers count up on scroll entry (Framer Motion
`useSpring` or `useCountUp`).

```
Without RLM                    With RLM Context OS
───────────────────            ───────────────────
37,500 tokens / call     →     3,000 tokens / call
$4.69 / session          →     $0.37 / session
5–15s compaction delay   →     ~0ms retrieval
Prior failures: lost     →     Prior failures: surfaced
```

---

## 5. Solution Section

**Alternating layout. Light on the left, demo on the right.**

### Label: `THE SOLUTION`

### Title:
```
Semantic retrieval.
Not compression.
```

**Body (~18px, max-w-xl):**
> RLM maintains a local-first semantic graph of everything that happened —
> every edit, every test result, every rollback, every failure — and retrieves
> only what's relevant to the current moment.
>
> The history is never deleted. It's indexed. When your AI needs context,
> it gets the 3k-token relevant slice — not the 40k-token dump.

**Right side: animated MCP response card**

A code block that "types in" on scroll entry:

```json
// get_current_context  →  2,847 tokens served
{
  "compression_ratio": 14.5,
  "compaction_calls_avoided": 5,
  "retrieved_memories": [{
    "action": "edit auth.py — token length check",
    "outcome": "broke db.py integration test",
    "outcome_score": -1.0
  }],
  "failure_warnings": [
    "⚠ Similar approach failed — avoid length-based token checks"
  ],
  "blast_radius": {
    "auth.py": { "depth": 0 },
    "db.py":   { "depth": 1, "type": "dependent" }
  }
}
```

Key values highlighted in blue. Warning in amber. Typewriter effect on scroll.

**Three integration pills below (horizontal, icon + label):**
```
[MCP icon] Works with Claude Code
[Terminal] Works in any terminal agent
[Cursor icon] Works with Cursor
```

---

## 6. Feature Bento Grid

**Bento grid layout. 12-column grid. Mix of 1/2/3-column-wide cards.**

This is the Cursor/Vercel move — a grid of cards that shows depth without
requiring the user to read everything.

**Card A (wide, 2/3):** "Recursive Recovery Loop"
> Edit → test → fail → rollback → memory write → retry.
> Deterministic file snapshots. Every attempt recorded.
> Verified trajectories, not just final patches.

Include a mini animated flow diagram (SVG, simple nodes and arrows,
draw-on-scroll):
```
[Edit] → [Test] → [Fail] → [Rollback] → [Graph] → [Retry] → [Pass ✓]
                              ↑__________record failure___________|
```

**Card B (1/3):** "Zero Data Leaves Your Machine"
> Local-first. MCP server runs on stdio.
> Graph stored in `.rlm_graph.json`.
> Memory in `.rlm_memory.json`.
> No cloud dependency.

**Card C (1/3):** "Works With Any LLM"
> GPT-4o. Claude. Gemini. Mistral.
> Cohere. Local Ollama. Azure.
> One client. Auto-detect from `.env`.

**Card D (2/3):** "Training Data Others Throw Away"
> Every session where RLM avoids compaction also produces a richer
> verified trajectory — the complete rollback→recovery path. This is
> the training corpus for RLM-0: a model that has *learned* to recover,
> not just been prompted to.
>
> Include small "signal density" visual — three bars:
> `Gold patch SFT: [██░░░░] L_ce only`
> `RL from reward:  [████░░] L_ce + sparse reward`
> `RLM training:    [██████] L_ce + L_ul + L_rb`

**Card E (1/3):** "122 Passing Tests"
> Full training pipeline. Render → collate → loss → score → eval.
> All tested. All passing. Not scaffolding.

---

## 7. How It Works — Three Horizons

**Full-width section. Bold number treatment.**

### Label: `THE ROADMAP`

### Title:
```
Three horizons.
One compound bet.
```

**Three large numbered blocks (vertical stack, left-aligned numbers):**

```
01                              Context OS + Runtime
────────────────────────────────────────────────────
Ships now. Any LLM wrapped in a recursive execution
loop with semantic context retrieval. The scaffold
is copyable. The data it produces is not.

02                              RLM-0 (in training)
────────────────────────────────────────────────────
Fine-tuned 8B model trained on recovery trajectories.
Three-term loss: L_ce + 0.5·L_ul + 0.1·L_rb.
A model that has learned to backtrack, not just
been prompted to. Smaller, faster, better on recovery.

03                              Native Architecture
────────────────────────────────────────────────────
Recurrent State Buffer. Learned exit mechanism.
Rollback as a structural property, not a token.
The 18-month research bet. Gated on H1 proving
the signal matters.
```

**Small honest callout (styled as a terminal comment, monospace, muted):**
```
// H0 moat = the data exhaust, not the scaffold
// H1 moat = structural training signal + 6-month corpus lead
// H2 moat = architecture-level recovery (research)
```

---

## 8. Research Section

**Dark card grid. 3-column on desktop, 1 on mobile.**

### Label: `RESEARCH`

### Title:
```
We publish what we know.
We label what we don't.
```

**Subtitle (muted):**
> [E] established  ·  [D] derived  ·  [T] thesis — our bet.

**Six cards — each with gradient top border on hover:**

Hover state: `border-t-2 border-gradient(blue→purple)`, card lifts `y: -4px`

1. **Architecture Failure Taxonomy** — 279 lines
   *Why LLMs structurally cannot solve agentic software engineering*
   `[Read →]`

2. **Research Audit** — Honest comparison to SWE-agent, RLEF, Agentless
   *Where we sit in the literature*
   `[Read →]`

3. **Competitive Analysis** — H0 moat = None, stated plainly
   *The flywheel is the bet*
   `[Read →]`

4. **Data Flywheel Analysis** — When does H1 pay off?
   *~300 verified trajectories for first signal*
   `[Read →]`

5. **H2 Architecture** — Recurrent State Buffer design
   *Rollback as structure, not token*
   `[Read →]`

6. **Experimental Roadmap** — 10 falsifiable experiments
   *Go/no-go gates. We'll know if we're wrong.*
   `[Read →]`

---

## 9. Team Section

### Label: `TEAM`

### Title:
```
Three people.
All the right backgrounds.
```

**Three cards (horizontal row, minimal):**

Each card: photo circle (placeholder gradient circle if no photo), name,
role, 3 bullet points, subtle hover lift.

**Card 1**
Role: `ML Research`
- IIT graduate, published ML research
- Research internship at frontier lab
- Leads: training pipeline, loss formulation, P1–P4 eval harness

**Card 2**
Role: `Enterprise GTM`
- Senior Director, Publicis Sapient
- Scaled AI programs across Fortune 500 enterprises
- Leads: enterprise pilots, partnerships, go-to-market

**Card 3**
Role: `Production ML`
- Principal ML Scientist, Oracle
- Shipped ML at enterprise scale under production constraints
- Leads: infrastructure, deployment, systems architecture

**Below team — one line, centered, muted:**
> We ship fast, test everything, and don't overclaim what we haven't proved.

---

## 10. CTA Section

**Full-width. Gradient background. The close.**

Background: `bg-gradient-to-br from-blue-950/60 via-purple-950/40 to-black`
Large glowing orb center-left behind content.

### Title (~56px, bold):
```
Stop losing context.
Start collecting signal.
```

### Sub:
```
Early access is open. No cloud. No data leaves your machine.
MCP install takes 5 minutes.
```

**Email input + button (inline on desktop, stacked on mobile):**
```
[ your@company.com                    ] [ Get Early Access → ]
```

Input style: `bg-white/5 border border-white/10 rounded-xl`
Button: blue, glowing, `shadow: 0 0 30px rgba(59,130,246,0.5)`

**Three trust chips below (small, horizontal):**
```
✓ Local-first, no cloud     ✓ Works with Claude Code + Cursor     ✓ Free during pilot
```

---

## 11. Footer

**Minimal. Two columns.**

```
Recursive Labs                    Problem · Solution · Research · Team
team@recursivelabs.ai
[GitHub ↗]  [Twitter/X ↗]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
© 2026 Recursive Labs.  Built honestly.
```

---

## Animation Spec (Framer Motion — exact)

```js
// Standard scroll-reveal (use this for all sections)
const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.25, 0.1, 0.25, 1] } }
}

// Stagger container
const stagger = {
  visible: { transition: { staggerChildren: 0.1 } }
}

// Hero float (terminal card)
const float = {
  animate: {
    y: [0, -8, 0],
    transition: { duration: 4, repeat: Infinity, ease: "easeInOut" }
  }
}

// Gradient orbs (hero background)
const orb = {
  animate: {
    scale: [1, 1.1, 1],
    opacity: [0.06, 0.1, 0.06],
    transition: { duration: 8, repeat: Infinity, ease: "easeInOut" }
  }
}

// Card hover
const cardHover = {
  scale: 1.02,
  transition: { duration: 0.2 }
}

// Typewriter (hero terminal, last line only)
// Use framer-motion's layout + stagger on individual characters
// Or use a library like `react-type-animation`
```

---

## Animations NOT to build

- Particle systems (too heavy, too 2019)
- Scroll-jacking (kills UX)
- 3D transforms on text
- Loading screens
- Cursor trails
- Anything that plays sound

---

## Copy Rules

**Use:** bold claims about the problem and the approach  
`"Your AI is running blind. We fixed the memory."`  
`"The context layer every AI tool is missing."`  
`"We capture the signal everyone else throws away."`

**Don't use:** specific unverified performance metrics as headlines  
~~"86% task success rate"~~ ← not real, don't use  
~~"State of the art on SWE-bench"~~ ← not yet  

Real numbers you CAN use (all verified from the codebase):
- `12×` token compression
- `$4.69 → $0.37` per 50-call session
- `122` passing tests
- `14.5×` compression ratio (from live MCP response)
- `5` compaction calls avoided (from live MCP response)

---

## Pages

Single page with anchor nav. That's it. No `/blog`. No `/pricing`.
`/` — everything above  
`/research/[doc]` — optional, renders the markdown docs from GitHub

---

## SEO

```html
<title>Recursive Labs — Context OS for AI Coding Agents</title>
<meta name="description" content="Eliminate context compaction.
12× fewer tokens, zero information lost, works with Claude Code,
Cursor, and any MCP-compatible agent. Local-first.">
<meta property="og:image" content="/og-hero.png">
<!-- og:image should be the terminal card from the hero, dark bg -->
```

---

## Mobile Checklist

- [ ] Hero headline breaks to 2 lines max at 375px
- [ ] Terminal card scrolls horizontally (don't shrink the font)
- [ ] Bento grid collapses to single column
- [ ] Comparison table: two stacked blocks, not side-by-side
- [ ] CTA: email input + button stacked vertically
- [ ] Nav: hamburger → full-screen overlay
- [ ] Orbs: reduce opacity by half on mobile (performance)
