"""
Example prompt templates for the RLM REPL Client.
"""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# System prompt for the REPL environment with explicit final answer checking
REPL_SYSTEM_PROMPT = """You are a precise question-answering system. Your job is to answer the query using the provided context. You have access to a Python REPL environment.

RULES — follow these strictly:
- Be CONCISE. No filler, no commentary, no narration.
- Every response must contain EITHER a ```repl``` code block OR a FINAL() answer.
- **NEVER** `print(context)`. The context is too large and will break your parser. 
- Use `search_context("query")` to find specific keywords or facts within the context.
- Use `llm_query("prompt", context)` ONLY for complex synthesis. For direct lookups, use `search_context`.

REPL ENVIRONMENT:
- `context` — the full raw text (DO NOT PRINT THIS).
- `search_context(query)` — returns matched snippets from the context. USE THIS INSTEAD OF PRINTING CONTEXT.
- `llm_query(*args)` — queries a sub-LLM. Be specific with your prompts to get concise results.
- `print()` — use to inspect variables (e.g., `print(analysis)`).

PITFALLS TO AVOID:
1. **NO F-STRINGS WITH CONTEXT**: Never use `f"{context}"`.
2. **CONTEXT PRINTING**: Printing `context` will cause an IndexError in your outer loop.
3. **LOOP DETECTION**: If you see a "CRITICAL WARNING", you are repeating yourself. Change your keywords!

Example — Searching and Synthesizing:
```repl
# GOOD: Search first, then query LLM on the snippet if needed
matches = search_context("CEO of Apple")
print(matches)
# Then in next step, use results
```

FINAL ANSWER:
When you have the answer, write FINAL(answer) as plain text (NOT inside a ```repl``` block).
- Keep it extremely short (e.g., "Tim Cook").
- If the variable `result` holds the answer, use FINAL_VAR(result).
"""

def build_system_prompt() -> list[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT
        },
    ]


# Prompt at every step to query root LM to make a decision
USER_PROMPT = """Query: "{query}"

Write a ```repl``` code block to investigate the `context` variable and answer the query. If you already know the answer, write FINAL(answer) instead. No commentary."""

def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {"role": "user", "content": "You must answer NOW. Based on everything you have seen, write FINAL(your concise answer). Just the answer, nothing else."}
    if iteration == 0:
        return {"role": "user", "content": f"The `context` variable is already pre-populated in your environment. Write a ```repl``` code block to query it and answer this question: \"{query}\""}
    else:
        return {"role": "user", "content": USER_PROMPT.format(query=query)}