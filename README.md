# RLM — Recursive Language Model (ACC API Integration)

This repository holds the **Recursive Language Model (RLM)** framework. Recently, the **Adaptive Compute Controller (ACC)** has been heavily integrated into the core framework and exposed safely through a new **FastAPI** service wrapper!

## What We've Accomplished Today

1. **REPL Integration (`RLM/acc_repl.py`)**: 
   - A newly established `AdaptiveRLM` wrapper seamlessly connects the ACC's heuristic complexity scorer directly to the RLM engine. This intelligently dictates how many iterations the RLM executes based on problem complexity.
   - **Shallow Tasks** operate securely inside a constrained execution loop (max 5 iterations).
   - **Deep Tasks** unlock aggressive compute (max 20 iterations).

2. **REST API Interface (`api/main.py`)**:
   - Spun up a FastAPI service instance that perfectly encapsulates episode budgets.
   - **POST `/api/v1/session`**: Starts tracked episodes under a defined constraint `max_api_calls`.
   - **POST `/api/v1/score`**: Allows uncoupled standalone query complexity analysis.
   - **POST `/api/v1/query`**: Safely launches the `AdaptiveRLM` against the active model architecture (e.g. `gemini-2.5-pro`) safely bound by the session constraints! 

## Quick Start (API)

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Add a `.env` file to the root of your project:
```
GENAI_API_KEY=your_gemini_key_here
```

### 3. Launch the Server
Boot the newly created FastAPI layer locally.
```bash
uvicorn api.main:app --reload
```

### 4. Play with endpoints
Create a session and start tracking dynamic complexity limits:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/session' \
  -H 'accept: application/json' \
  -d '{"max_api_calls": 30}'
```