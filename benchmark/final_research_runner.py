import os
import sys
import json
import time
import re
import argparse
from typing import List, Dict, Any, Optional, Callable

# Robust root path discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RLM.integrated_repl import IntegratedRLM
from RLM.utils.llm import LLMClient
from benchmark.metrics import score as compute_score
from benchmark.metrics import trace_consistency_judge

# --- LLM Judge Config (8B for local stability, 70B for high-precision papers) ---
JUDGE_MODEL = "ollama/llama3.1:70b" 
_JUDGE_CLIENT = None

def get_judge_client():
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = LLMClient(model=JUDGE_MODEL)
    return _JUDGE_CLIENT

def llm_judge(question: str, predicted: str, ground_truth: str) -> int:
    """Uses LLM to grade if the predicted answer is semantically correct on a 1-5 scale."""
    if predicted.strip().lower() == ground_truth.strip().lower():
        return 5
        
    prompt = f"""Evaluate the 'Agent's Predicted Answer' against the 'Ground Truth' for the given 'Question'.
Question: {question}
Ground Truth: {ground_truth}
Agent's Predicted Answer: {predicted}

GRADES (1 to 5):
1: Completely wrong.
2: Significant inaccuracies.
3: Contains correct fact, but buried in junk.
4: Factually correct, minor formatting issues.
5: Perfect and concise.

Return ONLY the integer digit."""
    try:
        response = get_judge_client().completion(prompt, max_tokens=10).strip()
        for char in response:
            if char in ("1", "2", "3", "4", "5"):
                return int(char)
        return 1
    except Exception:
        return 1

# --- Data Loaders ---

def load_hotpotqa(num_examples: int = 50) -> List[dict]:
    from datasets import load_dataset
    print(f"Loading HotpotQA validation set ({num_examples} examples)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    def _format_context(ex):
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        return "\n\n".join([f"**{t}**\n" + " ".join(s) for t, s in zip(titles, sentences)])

    examples = []
    for ex in dataset:
        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "context_str": _format_context(ex),
            "answer": ex["answer"],
            "type": ex["type"],
        })
        if len(examples) >= num_examples:
            break
    return examples

def load_logiqa(num_examples: int = 50) -> List[dict]:
    local_path = "benchmark/data/logiqa_test.jsonl"
    dataset = []
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
    else:
        from datasets import load_dataset
        dataset = load_dataset("tasksource/logiqa-2.0-nli", split="test")

    examples = []
    for ex in dataset:
        context = ex.get("premise", "")
        hypothesis = ex.get("hypothesis", "")
        answer = ex.get("label", "")
        examples.append({
            "id": f"logiqa_{len(examples)}",
            "question": f"{context}\n\n{hypothesis}",
            "context_str": context,
            "answer": str(answer),
            "type": "logical_reasoning"
        })
        if len(examples) >= num_examples:
            break
    return examples

# --- Runner Engine ---

def run_benchmark(
    examples: List[dict],
    rlm_factory: Callable,
    mode: str = "baseline",
    trace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    rlm = rlm_factory()
    results = []

    for i, ex in enumerate(examples):
        print(f"[{mode}] {i+1}/{len(examples)}: {ex['question'][:60]}...")
        t0 = time.time()
        try:
            predicted = rlm.completion(context=ex["context_str"], query=ex["question"])
        except Exception as e:
            predicted = f"ERROR: {e}"
        elapsed = time.time() - t0
        
        metrics = compute_score(predicted, ex["answer"])
        judge_score = llm_judge(ex["question"], predicted, ex["answer"])
        
        if hasattr(rlm, "update_last_memory"):
            rlm.update_last_memory(judge_score)
        
        result = {
            "id": ex["id"],
            "question": ex["question"],
            "gold": ex["answer"],
            "predicted": predicted,
            "semantic_score": judge_score,
            "em": 1 if judge_score >= 4 else 0,
            "f1": round(metrics["f1"], 4),
            "time_s": round(elapsed, 2),
        }

        if trace_dir and hasattr(rlm, "tracer"):
            trace_path = os.path.join(trace_dir, f"{ex['id']}_score{result['semantic_score']}.json")
            rlm.tracer.set_metadata({"semantic_score": result["semantic_score"], "f1": result["f1"]})
            rlm.tracer.save(trace_path)

        results.append(result)

    scores = [r["semantic_score"] for r in results]
    aggregate = {
        "avg_semantic_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "em": round(sum([r["em"] for r in results]) / len(results), 4) if results else 0.0,
        "f1": round(sum([r["f1"] for r in results]) / len(results), 4) if results else 0.0,
        "total": len(results),
    }
    return {"results": results, "aggregate": aggregate}

class VanillaBaseline:
    def __init__(self, model: str):
        self.llm = LLMClient(model=model)
    def completion(self, query: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (short):"
        return self.llm.completion(prompt).strip()

def run_research_matrix(samples: int = 100):
    configs = [
        {"name": "L3.1-8B-Baseline", "mode": "vanilla", "model": "ollama/llama3.1:8b"},
        {"name": "L3.1-8B-RLM", "mode": "rlm", "model": "ollama/llama3.1:8b", "acc": False, "mem": False, "eng": False},
        {"name": "L3.1-8B-RLM-ACC", "mode": "rlm", "model": "ollama/llama3.1:8b", "acc": True, "mem": False, "eng": False},
        {"name": "L3.1-8B-RLM-ACC-Memory", "mode": "rlm", "model": "ollama/llama3.1:8b", "acc": True, "mem": True, "eng": True},
        {"name": "L3.1-70B-Baseline", "mode": "vanilla", "model": "ollama/llama3.1:70b"}
    ]
    datasets = {"hotpotqa": load_hotpotqa, "logiqa": load_logiqa}
    final_report = {}

    for d_name, loader in datasets.items():
        examples = loader(samples)
        final_report[d_name] = {}
        for cfg in configs:
            print(f"\n>>> Running {cfg['name']} on {d_name} <<<")
            def rlm_factory():
                if cfg["mode"] == "vanilla": return VanillaBaseline(model=cfg["model"])
                return IntegratedRLM(model=cfg["model"], enable_acc=cfg["acc"], enable_memory=cfg["mem"], enable_engine=cfg["eng"], memory_path=f"mem_{d_name}_{cfg['name']}.json")
            
            output_dir = f"benchmark/results/research/{d_name}/{cfg['name']}"
            results = run_benchmark(examples=examples, rlm_factory=rlm_factory, mode=cfg["name"], trace_dir=os.path.join(output_dir, "traces"))
            final_report[d_name][cfg["name"]] = results["aggregate"]
            
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "summary.json"), "w") as f:
                json.dump(results, f, indent=2)

    with open("benchmark/results/research/final_summary.json", "w") as f:
        json.dump(final_report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    run_research_matrix(samples=args.samples)
