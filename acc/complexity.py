"""
ComplexityScorer — lightweight heuristic to estimate task complexity
from a query string and optional context.

Returns a float in [0, 1] used by AccController.new_episode() to set:
  - the overshoot threshold  (simpler tasks: 0.15, harder: 0.25)
  - the compute budget cap   (simpler tasks: 0.3,  harder: 0.6)

Design philosophy: intentionally simple heuristics. The ACC doesn't need
a perfect complexity estimate — it just needs a coarse signal that
separates "trivial lookup" from "multi-step reasoning" from "deep analysis".
The conformal calibration corrects for any miscalibration at runtime.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword lists — each hit adds to the raw complexity score
# ---------------------------------------------------------------------------

_HIGH_COMPLEXITY_KEYWORDS = [
    # multi-step reasoning
    "why", "explain", "analyse", "analyze", "compare", "contrast",
    "evaluate", "design", "architect", "trade-off", "tradeoff",
    # code tasks that need understanding
    "refactor", "debug", "optimize", "optimise", "implement", "rewrite",
    "fix", "bug", "error", "fails", "broken", "issue",
    # depth signals
    "step by step", "walk me through", "in detail", "comprehensively",
    "all the", "every", "entire",
]

_MEDIUM_COMPLEXITY_KEYWORDS = [
    "how", "what does", "summarise", "summarize", "list", "show me",
    "find", "locate", "where", "which files", "which function",
]

_LOW_COMPLEXITY_KEYWORDS = [
    "what is", "who is", "when", "version", "name of", "tell me",
    "show", "print", "display",
]


class ComplexityScorer:
    """
    Estimates task complexity from query text and optional context.

    score(query, context=None) -> float in [0, 1]
      ~0.0  trivial lookup / factual question
      ~0.3  moderate: single-step code task or short reasoning
      ~0.6  complex: multi-file, multi-step, or deep analysis
      ~1.0  very complex: architectural reasoning or full codebase tasks
    """

    def __init__(
        self,
        context_length_weight: float = 0.2,
        keyword_weight: float = 0.5,
        structure_weight: float = 0.3,
    ):
        self._ctx_w = context_length_weight
        self._kw_w = keyword_weight
        self._struct_w = structure_weight

    def score(self, query: str, context: Optional[str] = None) -> float:
        """
        Return a complexity score in [0, 1].

        Parameters
        ----------
        query   : The task description / user question.
        context : Optional code/file context injected alongside the query.
        """
        q = (query or "").lower()

        # ── 1. Keyword score (0–1) ──────────────────────────────────────────
        kw_score = 0.0
        for kw in _HIGH_COMPLEXITY_KEYWORDS:
            if kw in q:
                kw_score = min(1.0, kw_score + 0.15)
        for kw in _MEDIUM_COMPLEXITY_KEYWORDS:
            if kw in q:
                kw_score = min(1.0, kw_score + 0.08)
        for kw in _LOW_COMPLEXITY_KEYWORDS:
            if kw in q:
                kw_score = max(0.0, kw_score - 0.05)

        # ── 2. Structural signals (0–1) ─────────────────────────────────────
        struct_score = 0.0

        # Long queries tend to be more complex
        word_count = len(q.split())
        if word_count > 50:
            struct_score += 0.4
        elif word_count > 20:
            struct_score += 0.2
        elif word_count > 8:
            struct_score += 0.1

        # Multi-sentence = more sub-tasks
        sentence_count = len(re.split(r'[.!?]+', q.strip()))
        if sentence_count >= 4:
            struct_score += 0.3
        elif sentence_count >= 2:
            struct_score += 0.15

        # Contains code snippets or file paths
        if re.search(r'```|`[^`]+`|\.py\b|\.js\b|\.ts\b|def |class |import ', q):
            struct_score += 0.2

        struct_score = min(1.0, struct_score)

        # ── 3. Context length score (0–1) ───────────────────────────────────
        ctx_score = 0.0
        if context:
            ctx_len = len(context)
            if ctx_len > 10_000:
                ctx_score = 1.0
            elif ctx_len > 4_000:
                ctx_score = 0.7
            elif ctx_len > 1_000:
                ctx_score = 0.4
            elif ctx_len > 200:
                ctx_score = 0.2

        # ── 4. Weighted combination ─────────────────────────────────────────
        raw = (
            self._kw_w * kw_score
            + self._struct_w * struct_score
            + self._ctx_w * ctx_score
        )

        # Clamp to [0.05, 0.95] — never fully trivial, never fully maxed
        return max(0.05, min(0.95, raw))
