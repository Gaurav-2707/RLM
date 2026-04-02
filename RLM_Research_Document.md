# RLM (Recursive Language Model) Framework
**Comprehensive Architecture and Technical Overview**

---

## 1. Abstract & Core Philosophy
The **Recursive Language Model (RLM)** framework is designed to explicitly bypass the fixed token-window constraints of standard Large Language Models (LLMs). Instead of attempting to increase context length limits linearly—which runs into quadratic scaling complexities—RLM encapsulates the LLM within a programmatic orchestration loop. 

To search efficiently through exceptionally large datasets (e.g., millions of lines of unstructured text), the framework arms the LLM with a sandboxed **Python Read-Eval-Print Loop (REPL)**. The model dynamically generates Python scripts to parse, filter, compute, and navigate the dataset iteratively. The framework further permits the generated code to autonomously spawn recursive "sub-LLM" instances (`llm_query()`) to resolve mid-execution semantic queries on localized chunks of data.

## 2. System Architecture
At a high level, the system features a multi-layered design prioritizing autonomous iteration, memory persistency, and compute efficiency. 

### 2.1 The REPL Agent Loop (`RLM/rlm_repl.py`)
This serves as the root orchestrator. 
1. The orchestrator embeds the large context into a generated temporary file.
2. It constructs a system prompt for the "Root LLM" (defaulting to `gemini-2.5-flash`), indicating its role to write Python code to investigate the context.
3. The Root LLM responds with Python code blocks.
4. The framework extracts, parses, and executes these code blocks via the `REPLEnv`, sending the `stdout`, `stderr`, and any terminal expression evaluations back to the LLM.
5. The loop repeats until the LLM returns a final answer via standard formatting or the globally injected `FINAL_VAR(variable_name)` function.

### 2.2 Sandboxed Execution Environment (`RLM/repl.py`)
The `REPLEnv` represents the tightly controlled sandbox for programmatic iterations.
- **Safety**: Modifies the Python globals dictionary to block unsafe functions (`exec`, `eval`, `input`) while providing essential built-ins and standard data structures.
- **State Persistency**: Retains local variables (`self.locals`) across execution steps, making it function similarly to a Jupyter Notebook environment.
- **Hooks**: Injects custom functions such as `llm_query(prompt)` directly into the Python environment, allowing the Root LLM to delegate semantic evaluation of subset variables to a fresh, stateless LLM client mid-execution.

---

## 3. Cognitive Subsystems

### 3.1 Adaptive Compute Controller (ACC) (`RLM/acc/`)
LLM iterations are resource-intensive. The **ACC** subsystem mathematically bounds the amount of computation spent based on query complexity.
*   **Complexity Scorer (`complexity.py`)**: Before generating an iteration, it assigns a `[0, 1]` normalized complexity score based on:
    1.  **Lexical Entropy**: Shannon entropy of the token distribution (vocabulary richness).
    2.  **Length Metric**: Scales logically with token counts (saturating at 50 tokens).
    3.  **Keyword Density**: Weighted presence of "deep" analytical words (e.g., *synthesize, prove*) versus "shallow" terms (e.g., *who, list*).
*   **Budgeting Controller (`controller.py`)**: Maps the complexity score to reasoning **"Depths"**:
    *   *Depth 1 (Shallow)*: Score `< 0.35`
    *   *Depth 2 (Medium)*: Score `0.35 - 0.70`
    *   *Depth 3 (Deep)*: Score `> 0.70`
    It vigorously tracks expenditures against an `max_api_calls` budget limit per episode, gracefully clamping down reasoning depth if the framework nears its absolute computing limit.

### 3.2 Structured Reasoning Engine (`RLM/engine/rlm_engine.py`)
For semantic evaluations that do not warrant REPL code, the framework implements a structured three-step psychological workflow:
1.  **Decomposition**: Breaks the core query into atomic sub-problems.
2.  **Refinement**: Takes the summary of the decomposition, discards noise, and refines the core logic.
3.  **Synthesis**: Formulates the absolute final argument based strictly on the refined sub-problems.

### 3.3 Episodic Memory System (`RLM/memory/system.py`)
To prevent the framework from repeating failed logic paths or forgetting discoveries while traversing millions of lines, the `EpisodicMemorySystem` actively tracks `[State -> Action -> Outcome]` triads.
*   **Retrieval**: Combines standard BM25 sparse vector retrieval with a mathematical recency decay function: `score = α(BM25) + β(Outcome_Score * e^{-λΔt})`.
*   **Conflict Detection**: Uses Jaccard similarity across past stored states to deduce if a proposed action has historically produced conflicting outcomes (e.g., an identical previous thought pattern resulted in failure).
*   **Pruning**: Imposes a strict capacity logic, aggressively pruning memories that possess the lowest computed recency-outcome score combination.

---

## 4. Master Orchestration (`RLM/agent_controller.py`)
The overarching `AgentController` stitches the execution modules together. At every operational tick, the agent:
1.  Searches its Episodic Memory using its current context representation.
2.  Provides the historical context, memories, and task description to the Gemini APIs via robust JSON structured prompting.
3.  Parses the subsequent reasoning output (encompassing *rationale, proposed action, expected outcome*). Falls back to granular regex extraction if the LLM output deviates from strict JSON schema.
4.  Dispatches the action to the REPL/engine, receives an outcome score, commits it to memory, and steps forward.

## 5. Summary
The RLM structure proposes a highly advanced agentic loop. By removing memory capacity from inside the LLM Transformer context block, and instead delegating it to Python runtime variables, BM25 indices, and external file states, the architecture achieves a virtually unbounded effective context processing window optimized by autonomous Adaptive Computational limits.
