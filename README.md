# RLM — Recursive Language Model

A Python framework that enables LLMs to handle **arbitrarily large contexts** by giving the model a Python REPL as an interactive tool. Instead of stuffing a million-line document into a context window, the model writes code to navigate, slice, and query the data iteratively — and can spawn nested sub-LLM calls from within that code.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Adaptive Compute Controller (ACC)](#adaptive-compute-controller-acc)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Configuration](#configuration)

---

## Overview

**The core problem:** LLMs have a fixed context window. RLM solves this by turning the LLM into an agent that:

1. Receives a large context (e.g. 1M lines of text)
2. Writes Python code in a sandboxed REPL to search and analyse it
3. Calls nested sub-LLMs mid-execution via `llm_query()`
4. Iterates until it produces a `FINAL(answer)` or `FINAL_VAR(variable)`

**Needle-in-a-haystack benchmark (`main.py`):** hides a secret 7-digit number inside 1 000 000 lines of random text and asks the model to find it.

---

## Architecture

```
Your Query + Giant Context
         │
         ▼
   ┌─────────────┐
   │  RLM_REPL   │  ← Root LLM (Gemini / OpenAI)
   └──────┬──────┘
          │ writes Python
          ▼
   ┌─────────────┐
   │   REPLEnv   │  ← Sandboxed exec (notebook-style)
   └──────┬──────┘
          │ llm_query() hook
          ▼
   ┌─────────────┐
   │   Sub_RLM   │  ← Nested LLM call from within code
   └─────────────┘
          │ FINAL(answer) / FINAL_VAR(var)
          ▼
       Result
```

### Key components

| Component | Location | Purpose |
|---|---|---|
| `RLM_REPL` | `RLM/rlm_repl.py` | Orchestrates root LLM + REPL agentic loop |
| `REPLEnv` | `RLM/repl.py` | Sandboxed Python REPL with `llm_query()` / `FINAL_VAR()` |
| `RLMEngine` | `RLM/engine/rlm_engine.py` | Structured 3-step pipeline: Decompose → Refine → Synthesise |
| `EpisodicMemorySystem` | `RLM/memory/system.py` | BM25 retrieval with recency decay |
| **`ACC`** | **`RLM/acc/`** | **Adaptive Compute Controller (see below)** |

---

## Adaptive Compute Controller (ACC)

The **ACC** sits between the environment and the RLM, dynamically selecting the reasoning depth at each decision step based on a measured **complexity score**.

### Three-tier mapping rule

| Complexity Score | Depth | Reasoning Mode |
|---|---|---|
| `score < 0.35` | `d = 1` | Shallow — fast, cheap |
| `0.35 ≤ score ≤ 0.70` | `d = 2` | Medium |
| `score > 0.70` | `d = 3` | Deep — thorough, expensive |

### Budget enforcement

A configurable `max_api_calls` cap ensures the total API calls per episode never exceed the budget. If a requested depth would exceed the remaining budget, the ACC **gracefully falls back** to a shallower depth. When the budget is fully exhausted it returns `0` as a stop signal.

### Episode recording & post-hoc analysis

Every depth decision is stored as a `DepthRecord`. After an episode, `end_episode()` returns an `EpisodeReport` containing:

- Full depth distribution (counts per tier)
- Whether the distribution is genuinely non-uniform
- Average complexity across all steps
- Pearson correlation between complexity scores and depths selected

### Usage

```python
from RLM.acc import AdaptiveComputeController, ComplexityScorer

scorer = ComplexityScorer()
acc    = AdaptiveComputeController(max_api_calls=30)

acc.new_episode()

for step_query, step_context in episode_data:
    score = scorer.score(step_query, context=step_context)
    depth = acc.select_depth(complexity_score=score)

    if depth == 0:
        break  # budget exhausted

    # Pass `depth` to your RLMEngine or RLM_REPL as max_iterations, etc.
    print(f"Step score={score:.3f} → depth={depth}")

report = acc.end_episode()
print(report.summary())
```

See `example_acc.py` for a complete runnable demo.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Gemini
export GENAI_API_KEY="your-key-here"

# — or OpenAI —
export OPENAI_API_KEY="your-key-here"
```

### 3. Run the needle-in-a-haystack demo

```bash
python main.py
```

### 4. Run the ACC demo (no API key needed)

```bash
python example_acc.py
```

---

## Project Structure

```
Capstone/
├── main.py                    # Needle-in-a-haystack benchmark entry point
├── example_acc.py             # ACC standalone demo
│
├── RLM/
│   ├── rlm.py                 # Abstract RLM base class
│   ├── rlm_repl.py            # RLM_REPL: root LLM + REPL agentic loop
│   ├── repl.py                # REPLEnv sandboxed execution environment
│   ├── models.py              # Pydantic models (AgentState, MemoryEntry, …)
│   │
│   ├── acc/                   # ◀ Adaptive Compute Controller
│   │   ├── __init__.py
│   │   ├── controller.py      #   AdaptiveComputeController (core logic)
│   │   ├── complexity.py      #   ComplexityScorer (heuristic scoring)
│   │   └── models.py          #   DepthRecord, EpisodeReport
│   │
│   ├── engine/
│   │   ├── rlm_engine.py      # 3-step reasoning engine (Decompose/Refine/Synthesise)
│   │   └── templates.py       # Prompt templates for engine steps
│   │
│   ├── memory/
│   │   ├── base.py            # MemoryEntry dataclass
│   │   ├── system.py          # EpisodicMemorySystem (BM25 + recency decay)
│   │   └── retrieval.py       # BM25Retriever, JaccardRetriever
│   │
│   ├── utils/
│   │   ├── llm.py             # LLMClient (Gemini / OpenAI unified wrapper)
│   │   ├── prompts.py         # System & action prompt templates
│   │   └── utils.py           # Code-block parsing, REPL execution helpers
│   │
│   └── logger/
│       ├── root_logger.py     # ColorfulLogger
│       └── repl_logger.py     # REPLEnvLogger
│
└── tests/
    └── test_acc.py            # pytest suite for the ACC
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output covers:

- `ComplexityScorer` — range bounds, ordering, context signal, custom weights
- `AdaptiveComputeController` — depth mapping boundaries, budget clamping, exhaustion
- `EpisodeReport` — distribution stats, correlation, summary keys

---

## Configuration

### `ComplexityScorer`

| Parameter | Default | Description |
|---|---|---|
| `length_cap` | `50` | Token count at which the length sub-score saturates |
| `context_weight` | `0.15` | Weight of context-length signal in the final score |
| `weights` | `(0.35, 0.30, 0.35)` | Blend of (entropy, length, keywords) sub-scores |

### `AdaptiveComputeController`

| Parameter | Default | Description |
|---|---|---|
| `max_api_calls` | `None` | Episode budget cap; `None` = unlimited |
| `depth_costs` | `{1:1, 2:2, 3:3}` | API call cost per depth tier |

---

## Supported Models

| Provider | Model string prefix | API key env var |
|---|---|---|
| Google Gemini | `gemini-*` | `GENAI_API_KEY` |
| OpenAI | anything else | `OPENAI_API_KEY` |