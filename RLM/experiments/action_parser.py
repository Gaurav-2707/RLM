"""
Action Parser for GridWorld experiments.

Extracts a cardinal direction action from free-form LLM output.
Designed for robustness: handles verbose reasoning, markdown,
JSON wrapping, and degrades gracefully with a random fallback.

Valid actions: ["up", "down", "left", "right"]
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

logger = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({"up", "down", "left", "right"})

# Ordered from most specific to least specific
_PATTERNS = [
    # Pattern 1: Explicit action tags (e.g. "ACTION: right", "Action: down")
    re.compile(r"(?:action|move|direction)\s*[:=]\s*(up|down|left|right)", re.I),
    # Pattern 2: Quoted action (e.g. '"right"', "'left'")
    re.compile(r"""['"](\s*(?:up|down|left|right)\s*)['"]""", re.I),
    # Pattern 3: "I choose/pick/select <action>"
    re.compile(r"(?:choose|pick|select|go|move)\s+(up|down|left|right)", re.I),
    # Pattern 4: Bare keyword at start of line
    re.compile(r"^\s*(up|down|left|right)\s*$", re.I | re.M),
    # Pattern 5: Last resort — any occurrence in the text
    re.compile(r"\b(up|down|left|right)\b", re.I),
]


def parse_action(
    llm_output: str,
    valid_actions: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
) -> tuple[str, bool]:
    """
    Parse an action from LLM output.

    Parameters
    ----------
    llm_output : str
        Raw text output from the LLM.
    valid_actions : list[str] or None
        Subset of {"up","down","left","right"} that are currently valid
        in the environment.  If provided and the parsed action is not
        in this list, falls back to a random valid action.
    rng : random.Random or None
        RNG instance for deterministic fallback.  If None, uses the
        module-level random.

    Returns
    -------
    tuple[str, bool]
        (action, was_parsed).  was_parsed is False if a random fallback
        was used.
    """
    if rng is None:
        rng = random.Random()

    fallback_pool = list(valid_actions) if valid_actions else list(VALID_ACTIONS)

    if not llm_output or not llm_output.strip():
        logger.warning("Empty LLM output — falling back to random action.")
        return rng.choice(fallback_pool), False

    # Try each pattern in priority order
    for pattern in _PATTERNS:
        match = pattern.search(llm_output)
        if match:
            action = match.group(1).strip().lower()
            if action in VALID_ACTIONS:
                # Check if it's a valid move in the current state
                if valid_actions is not None and action not in valid_actions:
                    logger.debug(
                        "Parsed action '%s' is not valid (valid=%s) — "
                        "falling back to random.",
                        action, valid_actions,
                    )
                    return rng.choice(fallback_pool), False
                return action, True

    # No pattern matched
    logger.warning(
        "Could not parse action from LLM output (len=%d). "
        "First 200 chars: %s",
        len(llm_output), llm_output[:200],
    )
    return rng.choice(fallback_pool), False
