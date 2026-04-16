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
                      engine_history: Optional[List] = None):
        """Append a single iteration of the REPL loop to the trace."""
        self.repl_history.append({
            "iteration": iteration,
            "response": response,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "engine_history": engine_history
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

    def save(self, filepath: str):
        """Save the trace to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
