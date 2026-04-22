"""
Memory REPL adapter.

Wraps EpisodicMemorySystem and exposes:
  - retrieve_as_context(query, top_k) → formatted string for system prompt injection
  - store(query, reasoning, action, outcome, outcome_score) → store new experience
  - get_repl_function()              → memory_retrieve() callable for REPLEnv globals
  - memory_count()                   → number of stored memories
  - reset()                          → clear all memories
"""

from typing import List
from RLM.memory.system import EpisodicMemorySystem
from RLM.memory.base import MemoryEntry


class MemoryREPL:
    """
    Adapter that connects EpisodicMemorySystem to the REPL environment.

    Parameters
    ----------
    capacity : int
        Maximum number of memories to retain. Oldest/lowest-scoring pruned first.
    decay_rate : float
        Lambda for exponential recency decay (smaller = slower decay).
    """

    def __init__(self, capacity: int = 200, decay_rate: float = 0.0001):
        self.capacity = capacity
        self.system = EpisodicMemorySystem(
            capacity=capacity,
            decay_rate=decay_rate,
        )

    # ------------------------------------------------------------------
    # RLM_REPL outer-loop interface
    # ------------------------------------------------------------------

    def retrieve_as_context(self, query: str, top_k: int = 3) -> tuple[str, List[str]]:
        """
        Retrieve top_k memories relevant to *query* and return a (context_str, conflicts).
        """
        results, conflicts = self.system.retrieve(current_state=query, top_k=top_k)
        if not results:
            return "", []

        lines = ["=== Relevant Past Experience ==="]
        for i, (entry, score) in enumerate(results, 1):
            lines.append(f"[{i}] State: {entry.state}")
            if entry.reasoning:
                lines.append(f"    Reasoning: {entry.reasoning}")
            lines.append(
                f"    Outcome: {entry.outcome} (score: {entry.outcome_score:.2f})"
            )
        return "\n".join(lines), conflicts

    def store(
        self,
        query: str,
        reasoning: str,
        action: str,
        outcome: str,
        outcome_score: float = 0.5,
    ) -> List[str]:
        """
        Store a new episodic memory.

        Parameters
        ----------
        query : str
            The question / state description.
        reasoning : str
            Summary of the reasoning path taken.
        action : str
            The primary action the RLM took (e.g. code snippet summary).
        outcome : str
            Textual description of the outcome.
        outcome_score : float
            Quality of the outcome in [-1.0, 1.0]. Default 0.5 (neutral-positive).

        Returns
        -------
        List[str]
            List of conflict warning strings (empty if no conflicts detected).
        """
        entry = MemoryEntry(
            state=query,
            reasoning=reasoning,
            action=action,
            outcome=outcome,
            outcome_score=outcome_score,
        )
        return self.system.add_memory(entry)

    def get_repl_function(self) -> callable:
        r"""
        Return a ``memory_retrieve(query, top_k=3)`` callable suitable for
        injection into ``REPLEnv.globals``.

        The model can call ``memory_retrieve("my sub-question")`` inside a
        ``\`\`\`repl`` block to get relevant past experiences as a string.
        """
        def memory_retrieve(query: str, top_k: int = 3) -> str:
            """Retrieve relevant past experiences from memory. Returns a formatted string."""
            context, _ = self.retrieve_as_context(query=query, top_k=top_k)
            return context

        return memory_retrieve

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def memory_count(self) -> int:
        """Return the number of memories currently stored."""
        return len(self.system.memories)

    def reset(self):
        """Clear all stored memories."""
        self.system.memories = []
        self.system.retriever = type(self.system.retriever)()

    def warmup_memory(self, examples: List[dict]):
        """
        Pre-populate memory with 'Gold' examples.
        Each example should be a dict with: query, reasoning, action, outcome, score.
        """
        for ex in examples:
            self.store(
                query=ex.get("query", ""),
                reasoning=ex.get("reasoning", ""),
                action=ex.get("action", "gold_standard"),
                outcome=ex.get("outcome", ""),
                outcome_score=ex.get("score", 1.0)
            )
    def save(self, filepath: str):
        """Save memories to disk."""
        self.system.save(filepath)

    def load(self, filepath: str):
        """Load memories from disk."""
        self.system.load(filepath)
