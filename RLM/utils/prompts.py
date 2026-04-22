"""
Prompt templates for the RLM REPL Client (NLA-REPL).
"""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# ─── System Prompt ──────────────────────────────────────────────────────────
REPL_SYSTEM_PROMPT = """You are a precise question-answering agent with a Python REPL.

TOOLS:
- `search_context("query")` → snippets
- `read_full_context()` → full string
- `llm_query("prompt", context)` → synthesis
- `print(val)` → scratchpad

RULES:
1. FORMULATE HYPOTHESIS: Use the [Global Context Preview] to form a preliminary answer. 
2. VERIFY: Use ```repl``` to confirm facts. DO NOT guess if the preview is truncated or ambiguous.
3. LOGICAL REASONING: For logic puzzles, ALWAYS use this scratchpad format in ```repl```:
   print("PREMISES: <list all facts>")
   print("CONSTRAINTS: <list limits>")
   print("INFERENCE: <deduction steps>")
4. STICK TO FACTS: If the context doesn't contain the answer, say "Insufficient information".

FINAL ANSWER FORMAT:
- "FINAL ANSWER: <answer>"
- Keep it 1-3 words (e.g., "entailment", "Yes", "John Doe").
- This MUST be the last line.
"""

def build_system_prompt() -> list[Dict[str, str]]:
    return [{"role": "system", "content": REPL_SYSTEM_PROMPT}]


USER_PROMPT = """Query: "{query}"

Analyze facts vs hypothesis. If unverified, use ```repl```. When certain, output FINAL ANSWER: <answer>."""


def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {
            "role": "user",
            "content": (
                "Provide a final answer NOW. "
                "Output: FINAL ANSWER: <concise answer>\n"
                "Max 5 words. No code."
            )
        }
    if iteration == 0:
        return {
            "role": "user",
            "content": (
                f"Query: \"{query}\"\n\n"
                "1. Form a hypothesis from the [Global Context Preview].\n"
                "2. If the preview is enough, give FINAL ANSWER.\n"
                "3. If facts are missing, use ```repl``` to search/read."
            )
        }
    else:
        return {"role": "user", "content": USER_PROMPT.format(query=query)}