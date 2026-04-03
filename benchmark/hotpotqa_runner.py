"""
HotpotQA benchmark runner.

Loads the HotpotQA distractor validation set, runs RLM on a subset,
and returns per-example EM + F1 results plus aggregate stats.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable

from benchmark.metrics import score as compute_score


def _format_context(example: dict) -> str:
    """Safe context formatter handling HuggingFace datasets dict format."""
    try:
        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]
        paragraphs = []
        for title, sentences in zip(titles, sentences_list):
            para = f"**{title}**\n" + " ".join(sentences)
            paragraphs.append(para)
        return "\n\n".join(paragraphs)
    except Exception:
        return str(example.get("context", ""))


def load_hotpotqa(num_examples: int = 50, question_type: Optional[str] = None) -> List[dict]:
    """
    Load HotpotQA validation set from HuggingFace datasets.
    
    Parameters
    ----------
    num_examples : int
        How many examples to load (from the start of validation set).
    question_type : str, optional
        Filter by 'bridge' or 'comparison'. None loads all types.
    
    Returns
    -------
    List[dict]
        List of example dicts with keys: id, question, context_str, answer, type.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    print(f"Loading HotpotQA validation set ({num_examples} examples)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    examples = []
    for ex in dataset:
        if question_type and ex["type"] != question_type:
            continue
        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "context_str": _format_context(ex),
            "answer": ex["answer"],
            "type": ex["type"],
        })
        if len(examples) >= num_examples:
            break

    print(f"Loaded {len(examples)} examples.")
    return examples


def run_benchmark(
    examples: List[dict],
    rlm_factory: Callable,
    mode: str = "baseline",
    on_result: Optional[Callable[[dict], None]] = None,
) -> Dict[str, Any]:
    """
    Run RLM on a list of HotpotQA examples.
    
    Parameters
    ----------
    examples : list
        List of example dicts from load_hotpotqa().
    rlm_factory : callable
        Called with no args, returns a fresh RLM instance. Called once
        per run (memory persists across examples if enable_memory=True).
    mode : str
        Label for this run ('baseline' or 'enhanced').
    on_result : callable, optional
        Called with each per-example result dict as it completes.
        Use for streaming progress to a dashboard.
    
    Returns
    -------
    dict
        {
            "mode": str,
            "results": [per-example result dicts],
            "aggregate": {"em": float, "f1": float, "avg_time": float},
        }
    """
    rlm = rlm_factory()
    results = []

    for i, ex in enumerate(examples):
        print(f"[{mode}] {i+1}/{len(examples)}: {ex['question'][:60]}...")
        t0 = time.time()

        try:
            predicted = rlm.completion(
                context=ex["context_str"],
                query=ex["question"],
            )
        except Exception as e:
            predicted = f"ERROR: {e}"

        elapsed = time.time() - t0
        metrics = compute_score(predicted, ex["answer"])

        result = {
            "id": ex["id"],
            "question": ex["question"],
            "gold": ex["answer"],
            "predicted": predicted,
            "type": ex["type"],
            "em": metrics["em"],
            "f1": round(metrics["f1"], 4),
            "time_s": round(elapsed, 2),
            "mode": mode,
            "index": i,
        }
        results.append(result)

        if on_result:
            on_result(result)

    # Aggregate
    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    times = [r["time_s"] for r in results]

    aggregate = {
        "em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
        "f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "avg_time": round(sum(times) / len(times), 2) if times else 0.0,
        "total": len(results),
        "correct_em": sum(em_scores),
    }

    return {"mode": mode, "results": results, "aggregate": aggregate}


def save_results(results: dict, path: str):
    """Save benchmark results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")


def load_results(path: str) -> dict:
    """Load previously saved benchmark results."""
    with open(path) as f:
        return json.load(f)
