"""
HotpotQA benchmark metrics.
Official normalization + EM + F1 matching (mirrors hotpot_evaluate_v1.py).
"""

import re
import string
from collections import Counter
from typing import Tuple, List


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

    # Add a strip of trailing punctuation and normalized spacing
    s = lower(s)
    s = remove_punc(s)
    s = remove_articles(s)
    s = white_space_fix(s)
    return s.strip()


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


def trace_consistency_judge(question: str, trace_history: List[dict], final_answer: str) -> int:
    """
    Uses a 70B LLM to judge if the agent's code and reasoning steps 
    directly and correctly lead to the final answer.
    Returns a score 1-5.
    """
    from RLM.utils.llm import LLMClient
    judge = LLMClient(model="ollama/llama3.1:70b")
    
    # Format the trace for the judge
    history_text = ""
    for step in trace_history:
        history_text += f"\n--- Iteration {step['iteration']} ---\n"
        history_text += f"Reasoning/Code: {step['response']}\n"
        history_text += f"Output: {step['stdout'] or 'None'}\n"
        if step.get('stderr'):
            history_text += f"Error: {step['stderr']}\n"

    prompt = f"""You are a Logical Auditor. Your task is to verify if an agent's reasoning process is CONSISTENT with its final answer.

Question: {question}
Reasoning History:
{history_text}

Agent's Final Answer: {final_answer}

GRADES (1 to 5):
1: Contradictory - Logic clearly points elsewhere, yet the agent guessed this answer.
2: Disconnected - The reasoning has nothing to do with the question or answer.
3: Partially Support - The logic finds some relevant info but skips steps or has minor logic errors.
4: Strong Support - The logic is correct, but perhaps contains extra unverified assumptions.
5: Deductive - The answer follows perfectly and inexorably from the code outputs and reasoning.

Response with JUST the integer grade."""

    try:
        response = judge.completion(prompt).strip()
        # Extract first digit
        import re
        match = re.search(r'\d', response)
        return int(match.group()) if match else 1
    except Exception:
        return 1

def score(prediction: str, ground_truth: str) -> dict:
    """Return a dict with em, f1, precision, recall."""
    em = exact_match(prediction, ground_truth)
    f1, prec, rec = f1_score(prediction, ground_truth)
    return {"em": em, "f1": f1, "precision": prec, "recall": rec}
