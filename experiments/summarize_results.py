import argparse
import csv
import json
import os
from collections import defaultdict


REQUIRED_METADATA = {"method", "method_display", "task_id", "success", "source", "model"}
REQUIRED_METRICS = {"steps", "context_tokens", "failure_loops", "runtime_s"}


def _iter_traces(input_dir: str):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json") and file != "manifest.json":
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    yield path, json.load(f)


def _validate_trace(path: str, trace: dict):
    metadata = trace.get("metadata", {})
    metrics = trace.get("metrics", {})
    missing_meta = REQUIRED_METADATA - set(metadata)
    missing_metrics = REQUIRED_METRICS - set(metrics)
    if missing_meta or missing_metrics:
        raise ValueError(
            f"{path} missing metadata={sorted(missing_meta)} metrics={sorted(missing_metrics)}"
        )


def summarize(input_dir: str):
    grouped = defaultdict(list)
    for path, trace in _iter_traces(input_dir):
        _validate_trace(path, trace)
        grouped[trace["metadata"]["method"]].append(trace)

    if not grouped:
        raise ValueError(f"No trace JSON files found under {input_dir}")

    rows = []
    for method, traces in grouped.items():
        display = traces[0]["metadata"]["method_display"]
        total = len(traces)
        successes = sum(1 for trace in traces if trace["metadata"]["success"])
        metrics = [trace["metrics"] for trace in traces]
        rows.append({
            "Method": display,
            "Accuracy": f"{round((successes / total) * 100)}%",
            "Avg Steps": round(sum(m["steps"] for m in metrics) / total),
            "Avg Context Tokens": round(sum(m["context_tokens"] for m in metrics) / total),
            "Failure Loops": round(sum(m["failure_loops"] for m in metrics) / total),
            "Runtime (s)": round(sum(m["runtime_s"] for m in metrics) / total),
        })

    preferred_order = [
        "Direct LLM (GPT-4o / Zero-shot)",
        "Direct Small LLM (GPT-4o-mini / Zero-shot)",
        "Direct LLM (Frontier / Zero-shot)",
        "Direct Small LLM (Configured / Zero-shot)",
        "ReAct-style Agent (LangChain)",
        "ReAct-style Agent (Configured)",
        "Best-of-N (N=20)",
        "Best-of-N (N=5)",
        "RLM-1 (No Episodic Memory)",
        "RLM-1 (Full Neuro-Symbolic System)",
        "RLM-1 (Full System)",
    ]
    order = {name: idx for idx, name in enumerate(preferred_order)}
    rows.sort(key=lambda row: order.get(row["Method"], 999))
    return rows


def write_csv(rows, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Method",
            "Accuracy",
            "Avg Steps",
            "Avg Context Tokens",
            "Failure Loops",
            "Runtime (s)",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize trace-backed benchmark results.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input trace directory.")
    parser.add_argument("--out", default="ablation_results.csv", help="Output CSV path.")
    args = parser.parse_args()
    rows = summarize(args.input_dir)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} benchmark rows to {args.out}")


if __name__ == "__main__":
    main()
