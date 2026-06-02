# Recursive Language Model (RLM) & Adaptive Compute Controller

This repository contains the implementation and experimental suite for the **Recursive Language Model (RLM)** framework. It is designed to study the **Reasoning Overshoot** problem: the phenomenon where scaling test-time reasoning steps can degrade agent performance due to overthinking, answer overwriting, and context noise accumulation.

To solve this, the framework formalizes reasoning as an **optimal stopping problem** and implements an **Adaptive Compute Controller (ACC)** to dynamically halt reasoning at the compute-optimal point.

---

## 🚀 Features

- **Recursive Python REPL Environment**: A sandbox environment ([repl.py](file:///Users/arushsinghal/Documents/RLM/repl.py)) where LLMs can search context, run Python code, and query sub-LMs.
- **Optimal Stopping Theory Model**: Closed-form approximations and Bellman backward induction equations to find $N_{\text{opt}}$.
- **Adaptive early-exit controller**:
  - Grounding via strict Natural Language Inference (NLI) check.
  - Rollback mechanism to return the peak confidence answer when overthinking occurs.
  - DQN Reinforcement Learning-based controller trained offline on reasoning traces.
- **Episodic Memory System**: Retrieves relevant past trajectories and exposes conflict warnings to prevent repeat failures.
- **Evaluation Sweeps**: Tools for fixed-budget sweeps, scaling law analyses, ReAct baseline sweeps, and calibration.

---

## 📁 Repository Structure

```
├── acc/                    # Adaptive Compute Controller (grounding, conformal calibration, RL/heuristic controllers)
├── baselines/              # Reference agent implementations (ReAct, Best-of-N)
├── engine/                 # Deep Reasoning Engine (Decompose -> Refine -> Synthesize)
├── experiments/            # Benchmark sweeps on GSM8K, HotpotQA, and MATH
├── logger/                 # Colorful and structured logging utilities
├── memory/                 # Episodic Memory System (storage, retrieval, conflict checking)
├── theory/                 # Mathematical framework for optimal stopping
├── utils/                  # LLM API client wrappers, prompts, tracing, and helpers
├── pyproject.toml          # Project metadata and dependency definitions
└── uv.lock                 # Lockfile for reproducible environment setup
```

---

## 🛠️ Setup Instructions

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package and environment management.

### 1. Install `uv`
If you don't have `uv` installed:
```bash
curl -LsSf https://astral-sh/uv/install.sh | sh
```

### 2. Install Dependencies & Setup Virtual Environment
Run the following command in the project root to create the virtual environment and install all dependencies:
```bash
uv sync
```

### 3. Set API Keys
Depending on the model provider you use, set your API keys:
```bash
export OPENAI_API_KEY="your-openai-api-key"
# or
export GENAI_API_KEY="your-gemini-api-key"
```

---

## 🏃 Running Experiments & Demos

> **Note:** Because the package imports use the namespace `RLM`, always execute commands from the parent directory of the repository, or set the `PYTHONPATH` to include the project root:
> ```bash
> export PYTHONPATH=$PYTHONPATH:$(pwd)
> ```

### 1. Run the Theoretical Optimal Stopping Demo
To visualize the theoretical expected quality curves and $N_{\text{opt}}$ calculation under sub-Gaussian noise assumption:
```bash
uv run python theory/optimal_stopping.py
```

### 2. Run Scaling Law Sweeps
To run a fixed-budget sweep across different reasoning budgets to find the empirical $N_{\text{opt}}$:
```bash
uv run python -m RLM.experiments.run_scaling_law_sweep --model gpt-4o-mini --num_samples 100 --dataset gsm8k
```

### 3. Run the Adaptive Controller
To run and evaluate the Adaptive Compute Controller:
```bash
uv run python -m RLM.experiments.run_adaptive_controller --model gpt-4o-mini --dataset gsm8k
```
