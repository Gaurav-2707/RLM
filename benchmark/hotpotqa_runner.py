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

_JUDGE_CLIENT = None

def get_judge_client():
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from RLM.utils.llm import LLMClient
        # Use llama3 (widely available in Ollama)
        _JUDGE_CLIENT = LLMClient(model="ollama/llama3")
    return _JUDGE_CLIENT

def llm_judge(question: str, predicted: str, ground_truth: str) -> int:
    """Uses a 70B LLM to grade if the predicted answer is semantically correct on a 1-5 scale."""
    # Fast-path for exact string matching
    if predicted.strip().lower() == ground_truth.strip().lower():
        return 5
        
    # Pre-filter for messy output (still used as a penalty, but not an immediate 0)
    code_leakage = ["```", "llm_query(", "FINAL(", "FINAL_VAR(", "final_answer", "answer ="]
    leakage_penalty = False
    for leak in code_leakage:
        if leak in predicted:
            leakage_penalty = True
            break

    prompt = f"""You are an EXPERT evaluator grading a reasoning agent's performance on the HotpotQA dataset.
Evaluate the 'Agent's Predicted Answer' against the 'Ground Truth' for the given 'Question'.

Question: {question}
Ground Truth: {ground_truth}
Agent's Predicted Answer: {predicted}

GRADES (1 to 5):
1: Completely wrong, irrelevant, or hallucinates unrelated facts.
2: Significant inaccuracies or missed the core point entirely.
3: Contains the correct fact, but buried in reasoning junk, or has minor partially correct info.
4: Factually correct and clear, but perhaps includes slight extra fluff or minor formatting issues.
5: Perfect, concise, and factually identical to the core meaning of the ground truth.

Return ONLY the integer digit (1, 2, 3, 4, or 5). No explanation."""
    try:
        response = get_judge_client().completion(prompt, max_tokens=10).strip()
        # Strictly extract the first digit found
        for char in response:
            if char in ("1", "2", "3", "4", "5"):
                score = int(char)
                return max(1, score - 1) if leakage_penalty and score > 2 else score
        return 1
    except Exception as e:
        print(f"  [Judge Error] Failed to grade: {e}")
        return 1


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
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

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
    trace_dir: Optional[str] = None,
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
    trace_dir : str, optional
        Directory to save structured research traces (.json).
    
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
        
        # We still compute classic F1, but we override EM with the 70B Judge score
        metrics = compute_score(predicted, ex["answer"])
        judge_score = llm_judge(ex["question"], predicted, ex["answer"])
        
        result = {
            "id": ex["id"],
            "question": ex["question"],
            "gold": ex["answer"],
            "predicted": predicted,
            "type": ex["type"],
            "semantic_score": judge_score,
            "em": 1 if judge_score >= 4 else 0, # Treat 4 and 5 as EM-equivalent for legacy tracking
            "em_strict": metrics["em"],
            "f1": round(metrics["f1"], 4),
            "time_s": round(elapsed, 2),
            "mode": mode,
            "index": i,
        }

        # Save trace if tracer is available and trace_dir is set
        if trace_dir and hasattr(rlm, "tracer"):
            trace_path = os.path.join(trace_dir, f"{ex['id']}_score{result['semantic_score']}_f1{result['f1']}.json")
            # Attach final metrics to trace for cross-analysis
            rlm.tracer.set_metadata({
                "semantic_score": result["semantic_score"],
                "em": result["em"],
                "f1": result["f1"]
            })
            rlm.tracer.save(trace_path)

        results.append(result)

        if on_result:
            on_result(result)

    # Aggregate
    scores = [r["semantic_score"] for r in results]
    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    times = [r["time_s"] for r in results]

    aggregate = {
        "avg_semantic_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
        "f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "avg_time": round(sum(times) / len(times), 2) if times else 0.0,
        "total": len(results),
        "total_em": sum(em_scores),
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
