from RLM.memory.base import MemoryEntry
from RLM.memory.system import EpisodicMemorySystem


def test_memory_retrieval_emits_failure_warning():
    memory = EpisodicMemorySystem(capacity=10, conflict_thresh=0.1)
    memory.add_memory(MemoryEntry(
        state="auth boundary bug password length eight",
        reasoning="Changed password check in the wrong direction",
        action='{"file": "auth.py", "search": "> 8", "replace": "< 8"}',
        outcome="tests_failed",
        outcome_score=-1.0,
    ))

    _, warnings = memory.retrieve("auth boundary bug password length eight", top_k=3)

    assert warnings
    assert "failed" in warnings[0]
