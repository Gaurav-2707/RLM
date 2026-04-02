# Adaptive Compute Controller (ACC)

The **Adaptive Compute Controller (ACC)** is an intelligent module designed for the Recursive Language Model (RLM) framework. It dynamically manages the depth of LLM reasoning based on the measured complexity of the problem, ensuring efficient resource utilization without sacrificing precision on complex tasks.

## 🚀 Features

- **Complexity Scorer**: Analyzes incoming queries and context to produce a normalized complexity score `[0, 1]` based on lexical entropy, context length, and presence of deterministic vs. exploratory keywords.
- **Dynamic Reasoning Depth**: Uses a three-tier heuristic to map the complexity score to a reasoning depth tier (Shallow `1`, Medium `2`, Deep `3`).
- **Budget Enforcement**: Prevents excessive LLM chaining by tracking total API calls made across an episode.
- **REST API Enabled**: Exposes the logic bounds seamlessly through a FastAPI wrapper, making it extremely easy to hook up to UIs, web services, and metrics trackers.
- **REPL Integration**: Wraps standard `RLM_REPL` interaction limiters so iterations map organically to reasoning depths `Depth 1 = 5 iters`, `Depth 2 = 10 iters`, `Depth 3 = 20 iters`.

## 📁 File Structure
- `complexity.py`: Heuristics logic scoring token arrays and semantics.
- `controller.py`: Core routing mechanism managing tier assignment and budget caps.
- `models.py`: Dataclass models representing states like `DepthRecord` and `EpisodeReport`.
- `../acc_repl.py` (Integration): Wrapper integrating the logic cleanly within `RLM_REPL`.

## Usage Demo

```python
from RLM.acc import AdaptiveComputeController, ComplexityScorer

acc = AdaptiveComputeController(max_api_calls=25)
scorer = ComplexityScorer()

acc.new_episode()
score = scorer.score("Explain Newton's laws.", context="")
depth = acc.select_depth(score) # Dynamically picks depth tier!

print(f"Assigned Depth: {depth}")
```
