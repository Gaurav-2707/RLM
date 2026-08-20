"""
Three-term RLM-0 objective (spec §1.3-1.5):

  L_total = L_ce  +  alpha * L_ul  +  beta * L_rb

  L_ce  : masked, reward- & role-weighted causal LM loss
  L_ul  : unlikelihood on final-failed-action tokens (push prob away)
  L_rb  : rollback-budget penalty (keeps rollback frequency near data mean)
"""

from dataclasses import dataclass


@dataclass
class LossWeights:
    alpha_ul: float = 0.5
    beta_rb: float = 0.1
    r_hat: float = 1.0          # empirical mean rollbacks among successful trajs
    eps: float = 1e-6


def recursion_loss(logits, batch, rollback_token_id: int, weights: LossWeights):
    """
    logits:  [B, T, V]
    batch:   output of RLMCollator (tensors on same device as logits)
    Returns (total_loss, metrics_dict).
    """
    import torch
    import torch.nn.functional as F

    labels = batch["labels"]
    tok_w = batch["tok_weights"]
    ul_mask = batch["ul_mask"]
    traj_w = batch["traj_weight"]

    # shift for next-token prediction
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    tok_w = tok_w[:, 1:]
    ul_mask = ul_mask[:, 1:]

    B, T, V = logits.shape
    logp = F.log_softmax(logits.float(), dim=-1)

    mask = (labels != -100).float()
    safe_labels = labels.clamp_min(0)
    tgt_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T]

    # --- L_ce: token-role weighted, then trajectory-reward weighted ---
    ce_tok = -tgt_logp * mask * tok_w                      # [B,T]
    ce_per_traj = ce_tok.sum(dim=1) * traj_w               # [B]
    denom = (mask * tok_w).sum().clamp_min(weights.eps)
    l_ce = ce_per_traj.sum() / denom

    # --- L_ul: unlikelihood on final failed action ---
    p_tgt = tgt_logp.exp().clamp(max=1.0 - 1e-4)
    ul_tok = -torch.log1p(-p_tgt) * ul_mask
    ul_denom = ul_mask.sum().clamp_min(1.0)
    l_ul = ul_tok.sum() / ul_denom

    # --- L_rb: rollback-budget penalty ---
    # expected rollback-token mass emitted at trainable positions per trajectory
    p_rb = logp[..., rollback_token_id].exp()              # [B,T]
    emitted = (p_rb * mask).sum(dim=1)                     # [B]
    over = (emitted - weights.r_hat).clamp_min(0.0)
    l_rb = (over ** 2).mean()

    total = l_ce + weights.alpha_ul * l_ul + weights.beta_rb * l_rb

    metrics = {
        "loss": total.detach(),
        "l_ce": l_ce.detach(),
        "l_ul": l_ul.detach(),
        "l_rb": l_rb.detach(),
        "mean_rollback_mass": emitted.mean().detach(),
    }
    return total, metrics
