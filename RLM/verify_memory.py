import time
from memory import EpisodicMemorySystem, MemoryEntry

def test_memory_system():
    # 1. Initialize System
    mem_sys = EpisodicMemorySystem(capacity=10, alpha=0.8, beta=0.2)
    print("--- Initialized EpisodicMemorySystem (Capacity: 10) ---")

    # 2. Add some memories
    states = [
        "User asked about weather in London",
        "User wants to book a flight to Paris",
        "System failed to find weather data",
        "User is angry about the delay",
        "System successfully booked the flight",
    ]
    
    for i, s in enumerate(states):
        entry = MemoryEntry(
            state=s,
            reasoning=f"Reasoning for step {i}",
            action=f"Action {i}",
            outcome="Success" if i % 2 == 0 else "Failure",
            outcome_score=0.9 if i % 2 == 0 else -0.8
        )
        conflicts = mem_sys.add_memory(entry)
        if conflicts:
            print(f"Conflicts detected: {conflicts}")
    
    print(f"Total memories stored: {len(mem_sys.memories)}")

    # 3. Test Retrieval
    query = "weather in London"
    print(f"\n--- Retrieving for query: '{query}' ---")
    results = mem_sys.retrieve(query, top_k=2)
    for mem, score in results:
        print(f"Score: {score:.4f} | State: {mem.state} | Outcome Score: {mem.outcome_score}")

    # 4. Test Conflict Detection
    print("\n--- Testing Conflict Detection ---")
    conflict_entry = MemoryEntry(
        state="User asked about weather in London", # Same as index 0
        reasoning="Testing conflict",
        action="Fail",
        outcome="Failure",
        outcome_score=-0.9 # Opposite to index 0 which was 0.9
    )
    conflicts = mem_sys.add_memory(conflict_entry)
    for c in conflicts:
        print(f"ALERT: {c}")

    # 5. Test Pruning (Capacity is 10)
    print("\n--- Testing Pruning (Exceeding capacity 10) ---")
    for i in range(10):
        entry = MemoryEntry(
            state=f"Spam state {i}",
            reasoning="none",
            action="none",
            outcome="Neutral",
            outcome_score=0.1
        )
        mem_sys.add_memory(entry)
    
    print(f"Total memories after overflow: {len(mem_sys.memories)} (Expected: 10)")

if __name__ == "__main__":
    test_memory_system()
