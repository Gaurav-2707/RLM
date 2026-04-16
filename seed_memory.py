import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from RLM.integrated_repl import IntegratedRLM

GOLD_EXAMPLES = [
    {
        "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "reasoning": "Identify nationalities: Scott Derrickson (American), Ed Wood (American). Compare.",
        "outcome": "Yes, both were American.",
        "score": 1.0
    },
    {
        "query": "Which band, Letters to Cleo or Screaming Trees, had more members?",
        "reasoning": "Count members: Letters to Cleo (5 members), Screaming Trees (4 core members). Compare counts.",
        "outcome": "Letters to Cleo had more members.",
        "score": 1.0
    },
    {
        "query": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "reasoning": "Find locations: Laleli Mosque (Laleli), Esma Sultan Mansion (Ortakoy). Compare neighborhoods.",
        "outcome": "No, they are in different neighborhoods.",
        "score": 1.0
    }
]

def seed_rlm_memory(rlm: IntegratedRLM):
    if not rlm.enable_memory or not rlm._memory_adapter:
        print("Memory not enabled for this RLM instance.")
        return
    
    print(f"Seeding memory with {len(GOLD_EXAMPLES)} gold examples...")
    rlm._memory_adapter.warmup_memory(GOLD_EXAMPLES)
    print("Memory seeding complete.")

if __name__ == "__main__":
    # Example usage
    rlm = IntegratedRLM(enable_memory=True)
    seed_rlm_memory(rlm)
