"""
Complexity Scorer for the Adaptive Compute Controller (ACC).

Computes a normalised complexity score in [0, 1] for a given query
and/or context.  The score is a weighted combination of three
heuristic signals:

    1. Lexical entropy   – vocabulary richness of the query.
    2. Query length      – longer queries tend to be more complex.
    3. Keyword density   – presence of reasoning-intensive terms.

All three sub-scores are individually normalised to [0, 1] before
being blended, so the final score is always in [0, 1].
"""

from __future__ import annotations

import math
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword lists calibrated for LLM reasoning tasks
# ---------------------------------------------------------------------------

_SHALLOW_KEYWORDS: set[str] = {
    "what", "who", "when", "where", "list", "name", "define", "identify",
    "simple", "basic", "quick",
}

_DEEP_KEYWORDS: set[str] = {
    "why", "how", "explain", "analyse", "analyze", "compare", "contrast",
    "evaluate", "synthesize", "reason", "infer", "deduce", "prove",
    "implications", "consequences", "relationship", "mechanism", "strategy",
    "plan", "design", "critique", "justify", "argue",
}


class ComplexityScorer:
    """
    Estimates reasoning complexity of a query (and optionally its context).

    Parameters
    ----------
    length_cap : int
        Query token count above which the length sub-score saturates to 1.0.
        Default: 50 tokens.
    context_weight : float
        Weight given to context-length signal when context is provided.
        Must be in [0, 1].  Default: 0.15.
    weights : tuple[float, float, float]
        (w_entropy, w_length, w_keywords) controlling how the three
        sub-scores are blended.  They need not sum to 1; they will be
        normalised internally.
    """

    def __init__(
        self,
        length_cap: int = 50,
        context_weight: float = 0.15,
        weights: tuple[float, float, float] = (0.35, 0.30, 0.35),
    ) -> None:
        if not (0.0 <= context_weight <= 1.0):
            raise ValueError("context_weight must be in [0, 1].")
        if len(weights) != 3 or any(w < 0 for w in weights):
            raise ValueError("weights must be a 3-tuple of non-negative floats.")

        self.length_cap = length_cap
        self.context_weight = context_weight

        total = sum(weights)
        self.w_entropy, self.w_length, self.w_keywords = (
            w / total for w in weights
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def score(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> float:
        """
        Returns a complexity score in [0, 1].

        Parameters
        ----------
        query   : The user / agent query string.
        context : Optional context string.  When provided, context length
                  modestly boosts the score.
        """
        query = query.strip()
        if not query:
            return 0.0

        tokens = self._tokenize(query)

        s_entropy  = self._lexical_entropy(tokens)
        s_length   = self._length_score(tokens)
        s_keywords = self._keyword_score(tokens)

        base_score = (
            self.w_entropy  * s_entropy
            + self.w_length   * s_length
            + self.w_keywords * s_keywords
        )

        # Blend in context signal when present
        if context:
            ctx_signal = self._context_signal(context)
            base_score = (
                (1.0 - self.context_weight) * base_score
                + self.context_weight * ctx_signal
            )

        return round(min(max(base_score, 0.0), 1.0), 6)

    # ------------------------------------------------------------------ #
    # Sub-score helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _lexical_entropy(self, tokens: list[str]) -> float:
        """Shannon entropy of the token distribution, normalised to [0, 1]."""
        if len(tokens) < 2:
            return 0.0

        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        n = len(tokens)
        entropy = -sum(
            (c / n) * math.log2(c / n) for c in freq.values()
        )

        # Maximum possible entropy for n tokens is log2(n)
        max_entropy = math.log2(n) if n > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _length_score(self, tokens: list[str]) -> float:
        """Maps token count onto [0, 1] with saturation at length_cap."""
        return min(len(tokens) / self.length_cap, 1.0)

    @staticmethod
    def _keyword_score(tokens: list[str]) -> float:
        """
        Returns a score in [0, 1] based on keyword density.

        Deep keywords push the score up; shallow keywords pull it down.
        """
        token_set = set(tokens)
        deep_hits    = len(token_set & _DEEP_KEYWORDS)
        shallow_hits = len(token_set & _SHALLOW_KEYWORDS)

        # Net signal in [-1, 1]
        net = deep_hits - 0.5 * shallow_hits
        cap = max(len(_DEEP_KEYWORDS), 1)
        normalised = net / cap

        # Shift to [0, 1]
        return min(max((normalised + 1.0) / 2.0, 0.0), 1.0)

    def _context_signal(self, context: str) -> float:
        """
        Estimates context complexity from raw character length.
        Saturates at 1 million characters (typical large-context threshold).
        """
        _SATURATION = 1_000_000
        return min(len(context) / _SATURATION, 1.0)
