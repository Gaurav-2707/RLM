from dataclasses import asdict
from .base import MemoryEntry
from .retrieval import DenseRetriever
import math
import time
from typing import List, Tuple, Optional

class EpisodicMemorySystem:
    """
    Episodic Memory System for Recursive Language Models.
    Features BM25 retrieval, recency decay, and buffer management.
    """
    def __init__(
        self, 
        capacity: int = 100, 
        alpha: float = 0.6,    # relevance weight
        beta: float = 0.4,     # recency*outcome weight
        decay_rate: float = 0.0001, # lambda for recency
        conflict_thresh: float = 0.8 # Jaccard thresh for conflict
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.decay_rate = decay_rate
        self.conflict_thresh = conflict_thresh
        self.memories: List[MemoryEntry] = []
        try:
            self.retriever = DenseRetriever()
        except Exception:
            from .retrieval import BM25Retriever
            self.retriever = BM25Retriever()
        self.global_step_count = 0

    def add_memory(self, entry: MemoryEntry) -> List[str]:
        """Adds a memory and checks for conflicts."""
        self.global_step_count += 1
        entry.timestamp = self.global_step_count
        conflicts = self._detect_conflicts(entry)
        self.memories.append(entry)
        
        if len(self.memories) > self.capacity:
            self._prune()
            
        return conflicts

    def _detect_conflicts(self, new_entry: MemoryEntry) -> List[str]:
        """Detects if states are similar but outcomes are opposite."""
        conflicts = []
        from .retrieval import JaccardRetriever
        jaccard = JaccardRetriever()
        
        for i, m in enumerate(self.memories):
            sim = jaccard.score(new_entry.state, m.state)
            if sim > self.conflict_thresh:
                # Check for opposite outcomes (e.g. positive vs negative score)
                if (new_entry.outcome_score * m.outcome_score) < 0:
                    conflicts.append(f"Conflict with memory {i}: Similar state, opposite outcome.")
        return conflicts

    def _prune(self):
        """Removes low-outcome, aged memories."""
        def prune_score(m: MemoryEntry):
            recency = math.exp(-self.decay_rate * (self.global_step_count - m.timestamp))
            # Lower score means more likely to be pruned
            # We want to keep high outcome and high recency
            return 0.7 * m.outcome_score + 0.3 * recency

        self.memories.sort(key=prune_score)
        # Remove the lowest scoring one
        self.memories = self.memories[1:]

    def retrieve(self, current_state: str, top_k: int = 5) -> Tuple[List[Tuple[MemoryEntry, float]], List[str]]:
        """
        Retrieves top_k memories and a list of conflict warnings.
        returns: (scored_memories, conflict_warnings)
        """
        if not self.memories:
            return [], []

        # Update BM25 index with current memories
        states = [m.state for m in self.memories]
        self.retriever.fit(states)
        
        scored_memories = []
        conflicts = []

        for i, m in enumerate(self.memories):
            relevance = self.retriever.score(current_state, i)
            recency = math.exp(-self.decay_rate * (self.global_step_count - m.timestamp))
            
            # Formulate: score = α * relevance + β * (recency * outcome_score)
            total_score = (self.alpha * relevance) + (self.beta * (recency * m.outcome_score))
            
            if m.outcome_score > 0:
                scored_memories.append((m, total_score))

             # Logic for "Failure Pattern" detection (Conflict Awareness for Paper)
            # If similarity is high but outcome was bad, flag as internal conflict 
            if relevance > self.conflict_thresh and m.outcome_score < 0:
                path = self.get_path(m)
                path_str = " -> ".join(f"[{p.action}]" for p in path)
                conflicts.append(
                    f"Warning: A previous similar approach for '{m.state[:50]}...' "
                    f"failed (score: {m.outcome_score}). Path: {path_str}. Reason: {m.outcome}."
                )

        # Sort by total_score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[:top_k], list(set(conflicts))

    def get_path(self, entry: MemoryEntry) -> List[MemoryEntry]:
        """Traverses backwards from the entry to reconstruct the path of actions."""
        path = []
        visited = set()
        
        # Helper to find entry by ID
        id_to_mem = {m.entry_id: m for m in self.memories if m.entry_id}
        
        curr = entry
        while curr and curr.entry_id not in visited:
            visited.add(curr.entry_id)
            path.append(curr)
            # Take the first parent for linear sequence representation (if multiple)
            if curr.parent_ids and curr.parent_ids[0] in id_to_mem:
                curr = id_to_mem[curr.parent_ids[0]]
            else:
                curr = None
                
        # Return path starting from the oldest parent
        path.reverse()
        return path

    def save(self, filepath: str):
        """Persist memories to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in self.memories], f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """Load memories from a JSON file."""
        import json
        import os
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.memories = [MemoryEntry(**d) for d in data]
        
        if self.memories:
            self.global_step_count = int(max(m.timestamp for m in self.memories))
        else:
            self.global_step_count = 0
            
        # Prune if over capacity after loading
        if len(self.memories) > self.capacity:
            self._prune()
