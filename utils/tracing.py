import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

class TraceStorage:
    """
    Structured storage for RLM reasoning traces.
    Captures metadata from all sub-systems (ACC, Memory, REPL, Engine)
    for downstream NLP research tasks.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.metadata = {}
        self.query = ""
        self.predicted_answer = ""
        self.acc_data = {}
        self.memory_data = {}
        self.repl_history = []
        self.start_time = datetime.now()

    def set_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

    def set_query(self, query: str):
        self.query = query

    def set_predicted_answer(self, answer: str):
        self.predicted_answer = answer

    def set_acc_data(self, data: Dict[str, Any]):
        self.acc_data = data

    def set_memory_data(self, data: Dict[str, Any]):
        self.memory_data = data

    def add_repl_step(self, iteration: int, response: str, 
                      code: Optional[str] = None, 
                      stdout: Optional[str] = None, 
                      stderr: Optional[str] = None, 
                      engine_history: Optional[List] = None,
                      context_length: int = 0,
                      snapshot_answer: Optional[str] = None,
                      confidence: Optional[float] = None):
        """Append a single iteration of the REPL loop to the trace."""
        self.repl_history.append({
            "iteration": iteration,
            "response": response,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "engine_history": engine_history,
            "context_length": context_length,
            "snapshot_answer": snapshot_answer,
            "confidence": confidence
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert accumulated trace to a serializable dictionary."""
        return {
            "metadata": {
                **self.metadata,
                "duration_s": (datetime.now() - self.start_time).total_seconds(),
                "timestamp": self.start_time.isoformat()
            },
            "query": self.query,
            "predicted_answer": self.predicted_answer,
            "acc_data": self.acc_data,
            "memory_data": self.memory_data,
            "repl_history": self.repl_history
        }

    def calculate_research_metrics(self) -> Dict[str, float]:
        """
        Calculate research-specific metrics:
        - ERR (Error Recovery Rate): Successful steps after a failed step.
        - CAR (Compute-to-Accuracy Ratio): Accuracy / Total tokens/calls.
        """
        # ERR calculation
        errors = 0
        recoveries = 0
        for i in range(len(self.repl_history) - 1):
            if self.repl_history[i].get("stderr"):
                errors += 1
                # If next step has no stderr, it's a recovery
                if not self.repl_history[i+1].get("stderr"):
                    recoveries += 1
        
        err = recoveries / errors if errors > 0 else 1.0

        # CAR calculation (simplified: 1/num_steps if correct)
        semantic_score = self.metadata.get("semantic_score", 0)
        is_correct = 1.0 if semantic_score >= 4 else 0.0
        num_steps = len(self.repl_history)
        car = is_correct / max(1, num_steps)

        # Syntax errors and Exceptions count
        syntax_errors = sum(1 for step in self.repl_history if step.get("stderr") and ("SyntaxError" in step["stderr"] or "Exception" in step["stderr"]))

        return {
            "error_recovery_rate": round(err, 4),
            "compute_to_accuracy_ratio": round(car, 4),
            "total_steps": num_steps,
            "errors_encountered": errors,
            "syntax_errors": syntax_errors
        }

    def save(self, filepath: str):
        """Save the trace to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Inject research metrics before saving
        data = self.to_dict()
        data["research_metrics"] = self.calculate_research_metrics()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
