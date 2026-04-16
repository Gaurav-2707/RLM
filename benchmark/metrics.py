"""
HotpotQA benchmark metrics.
Official normalization + EM + F1 matching (mirrors hotpot_evaluate_v1.py).
"""

import re
import string
from collections import Counter
from typing import Tuple


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation/extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list:
    if not s:
        return []
    return normalize_answer(s).split()


def exact_match(prediction: str, ground_truth: str) -> int:
    """1 if predictions match after normalization, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Token-level F1, precision, recall."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if not pred_tokens or not gold_tokens:
        # Edge case: if either is empty, F1 is 1 only if both are empty
        exact = int(pred_tokens == gold_tokens)
        return exact, exact, exact

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def score(prediction: str, ground_truth: str) -> dict:
    """Return a dict with em, f1, precision, recall."""
    em = exact_match(prediction, ground_truth)
    f1, prec, rec = f1_score(prediction, ground_truth)
    return {"em": em, "f1": f1, "precision": prec, "recall": rec}
