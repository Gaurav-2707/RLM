"""
Control-token vocabulary and tokenizer surgery for RLM-0.

The recursive grammar uses special tokens so that decision points
(test verdict, rollback, confidence) become low-entropy, probeable
positions in the sequence. Confidence is discretized into 5 buckets.
"""

from typing import List

# --- structural tokens ---
STRUCT_TOKENS = [
    "<task>", "</task>",
    "<step>", "</step>",
    "<state>", "</state>",
    "<reason>", "</reason>",
    "<action>", "</action>",
    "<test>", "</test>",
    "<rollback>", "</rollback>",
    "<conf>", "</conf>",
    "<final>",
]

# --- verdict tokens (single tokens, world-supplied) ---
VERDICT_TOKENS = ["PASS", "FAIL"]

# --- outcome tokens ---
OUTCOME_TOKENS = ["SUCCESS", "FAILURE"]

# --- confidence buckets: float -> token via CONF_BIN_EDGES ---
CONF_TOKENS = ["<conf_vl>", "<conf_l>", "<conf_m>", "<conf_h>", "<conf_vh>"]
CONF_BIN_EDGES = [0.2, 0.4, 0.6, 0.8]  # 4 edges -> 5 buckets

CONTROL_TOKENS: List[str] = STRUCT_TOKENS + VERDICT_TOKENS + OUTCOME_TOKENS + CONF_TOKENS

# Decision tokens get upweighted in the loss (the recursive control surface).
DECISION_TOKEN_STRS = ["<rollback>", "</rollback>", "<action>"] + CONF_TOKENS


def conf_to_token(confidence: float) -> str:
    """Map a [0,1] confidence float to its bucket token."""
    c = max(0.0, min(1.0, float(confidence)))
    idx = 0
    for edge in CONF_BIN_EDGES:
        if c >= edge:
            idx += 1
    return CONF_TOKENS[idx]


def add_control_tokens(tokenizer, model=None, mean_init: bool = True):
    """
    Add control tokens to the tokenizer and (optionally) resize model
    embeddings, initializing new rows to the mean of existing embeddings
    plus small noise for faster convergence.

    Returns the number of tokens added.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": CONTROL_TOKENS}
    )
    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        if mean_init:
            _mean_init_new_rows(model, num_added)
    return num_added


def _mean_init_new_rows(model, num_added: int):
    import torch

    emb = model.get_input_embeddings().weight.data
    out = model.get_output_embeddings()
    n = emb.size(0)
    old_mean = emb[:-num_added].mean(dim=0, keepdim=True)
    emb[-num_added:] = old_mean + 0.02 * torch.randn_like(emb[-num_added:])
    if out is not None and out.weight.data.size(0) == n:
        out_mean = out.weight.data[:-num_added].mean(dim=0, keepdim=True)
        out.weight.data[-num_added:] = out_mean + 0.02 * torch.randn_like(
            out.weight.data[-num_added:]
        )


def decision_token_ids(tokenizer) -> List[int]:
    """Token ids that should receive the decision-token loss weight."""
    return [tokenizer.convert_tokens_to_ids(t) for t in DECISION_TOKEN_STRS]


def special_id(tokenizer, token: str) -> int:
    return tokenizer.convert_tokens_to_ids(token)
