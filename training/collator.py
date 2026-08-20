"""
Masked, reward/role-weighted collator for RLM-0.

`encode_trajectory` turns a RenderedTrajectory into aligned arrays:
  input_ids, labels (-100 where masked), tok_weights, ul_mask
by tokenizing each span and propagating its role/flags to every token,
so masks never drift from a re-tokenization pass.

`RLMCollator` pads a batch and carries per-example traj_weight / is_success.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from .render import RenderedTrajectory, render_trajectory
from .tokens import DECISION_TOKEN_STRS, conf_to_token

LABEL_IGNORE = -100
DECISION_WEIGHT = 2.0   # lambda_dec
DEFAULT_WEIGHT = 1.0

# roles whose tokens count as decision tokens for tok_weight
DECISION_ROLES = {"action", "rollback", "conf"}


def encode_trajectory(rt: RenderedTrajectory, tokenizer,
                      max_seq_len: int = 16384) -> Dict[str, Any]:
    input_ids: List[int] = []
    labels: List[int] = []
    tok_weights: List[float] = []
    ul_mask: List[int] = []

    for span in rt.spans:
        ids = tokenizer.encode(span.text, add_special_tokens=False)
        if not ids:
            continue
        w = DECISION_WEIGHT if span.role in DECISION_ROLES else DEFAULT_WEIGHT
        for tid in ids:
            input_ids.append(tid)
            labels.append(tid if span.trainable else LABEL_IGNORE)
            tok_weights.append(w if span.trainable else 0.0)
            ul_mask.append(1 if span.is_ul else 0)

    # tail-truncate to budget (keep <final> by preferring the end)
    if len(input_ids) > max_seq_len:
        cut = len(input_ids) - max_seq_len
        input_ids = input_ids[cut:]
        labels = labels[cut:]
        tok_weights = tok_weights[cut:]
        ul_mask = ul_mask[cut:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "tok_weights": tok_weights,
        "ul_mask": ul_mask,
        "traj_weight": rt.traj_weight,
        "is_success": rt.is_success,
        "num_rollbacks": rt.num_rollbacks,
        "task_id": rt.task_id,
    }


@dataclass
class RLMCollator:
    tokenizer: Any
    max_seq_len: int = 16384
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        # accept either raw RenderedTrajectory/encoded dicts
        encoded = []
        for f in features:
            if isinstance(f, RenderedTrajectory):
                encoded.append(encode_trajectory(f, self.tokenizer, self.max_seq_len))
            elif "input_ids" in f:
                encoded.append(f)
            else:
                raise TypeError("collator expects encoded dicts or RenderedTrajectory")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        maxlen = max(len(e["input_ids"]) for e in encoded)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            maxlen = ((maxlen + m - 1) // m) * m

        def pad(seq, val, n):
            return seq + [val] * (n - len(seq))

        batch = {
            "input_ids": [], "labels": [], "attention_mask": [],
            "tok_weights": [], "ul_mask": [],
        }
        traj_w, is_succ, nrb = [], [], []
        for e in encoded:
            L = len(e["input_ids"])
            batch["input_ids"].append(pad(e["input_ids"], pad_id, maxlen))
            batch["labels"].append(pad(e["labels"], LABEL_IGNORE, maxlen))
            batch["attention_mask"].append([1] * L + [0] * (maxlen - L))
            batch["tok_weights"].append(pad(e["tok_weights"], 0.0, maxlen))
            batch["ul_mask"].append(pad(e["ul_mask"], 0, maxlen))
            traj_w.append(e["traj_weight"])
            is_succ.append(1.0 if e["is_success"] else 0.0)
            nrb.append(e["num_rollbacks"])

        out = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "tok_weights": torch.tensor(batch["tok_weights"], dtype=torch.float),
            "ul_mask": torch.tensor(batch["ul_mask"], dtype=torch.float),
            "traj_weight": torch.tensor(traj_w, dtype=torch.float),
            "is_success": torch.tensor(is_succ, dtype=torch.float),
            "num_rollbacks": torch.tensor(nrb, dtype=torch.float),
        }
        return out
