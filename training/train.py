"""
RLM-0 QLoRA training driver.

Ties the package together:
  tokenizer surgery -> 4-bit base + LoRA -> reward-weighted three-term loss
  -> periodic recursion-proof eval -> checkpoint on best P3 (feedback value)
  -> general-capability tripwire early-stop.

Run:
  python -m training.train --traj_dir ~/.rlm/traces --out runs/rlm0

Heavy deps (torch/transformers/peft/bitsandbytes) are imported inside main()
so the rest of the package stays usable for data prep without a GPU stack.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Callable, List

from .tokens import add_control_tokens, special_id
from .collator import RLMCollator
from .loss import recursion_loss, LossWeights
from .data import load_trajectories, compute_r_hat, TrajectoryDataset, ReplayMixDataset


@dataclass
class TrainConfig:
    base_model: str = "meta-llama/Llama-3.1-8B"
    traj_dir: str = "~/.rlm/traces"
    out_dir: str = "runs/rlm0"
    max_seq_len: int = 16384
    micro_bs: int = 1
    grad_accum: int = 64           # effective batch 64
    lr: float = 1e-4
    epochs: int = 2
    warmup_frac: float = 0.03
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    alpha_ul: float = 0.5
    beta_rb: float = 0.1
    replay_frac: float = 0.04
    eval_every: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    # general-capability tripwire: stop if probe drops > this (absolute)
    forgetting_tol: float = 0.03


# ---- hooks the caller supplies (kept out of core so eval/probe stay flexible) ----
EvalFn = Callable[[object, object], dict]      # (model, tokenizer) -> metrics
ProbeFn = Callable[[object, object], float]    # (model, tokenizer) -> gen-capability score


def build_model_and_tokenizer(cfg: TrainConfig):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    add_control_tokens(tok, model)             # resize + mean-init new rows
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def train(cfg: TrainConfig, eval_fn: Optional[EvalFn] = None,
          probe_fn: Optional[ProbeFn] = None,
          general_items: Optional[List] = None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    try:
        from bitsandbytes.optim import PagedAdamW8bit as Optim
    except Exception:
        Optim = torch.optim.AdamW

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    trajectories = load_trajectories(cfg.traj_dir)
    if not trajectories:
        raise RuntimeError(f"no verified trajectories in {cfg.traj_dir}")
    r_hat = compute_r_hat(trajectories)
    print(f"loaded {len(trajectories)} trajectories | r_hat={r_hat:.3f}")

    model, tok = build_model_and_tokenizer(cfg)
    rb_id = special_id(tok, "<rollback>")
    weights = LossWeights(alpha_ul=cfg.alpha_ul, beta_rb=cfg.beta_rb, r_hat=r_hat)

    ds = TrajectoryDataset(trajectories)
    if general_items:
        ds = ReplayMixDataset(ds, general_items, cfg.replay_frac, cfg.seed)
    collate = RLMCollator(tok, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=cfg.micro_bs, shuffle=True,
                        collate_fn=collate)

    total_steps = (len(loader) // cfg.grad_accum) * cfg.epochs
    opt = Optim([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.warmup_frac * total_steps), total_steps)

    base_probe = probe_fn(model, tok) if probe_fn else None
    best_p3 = float("-inf")
    gstep = 0
    model.train()

    for epoch in range(cfg.epochs):
        for it, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            loss, metrics = recursion_loss(out.logits, batch, rb_id, weights)
            (loss / cfg.grad_accum).backward()

            if (it + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += 1

                if gstep % 10 == 0:
                    print(f"e{epoch} s{gstep} " + " ".join(
                        f"{k}={float(v):.4f}" for k, v in metrics.items()))

                if gstep % cfg.eval_every == 0:
                    _checkpoint_step(cfg, model, tok, eval_fn, probe_fn,
                                     base_probe, gstep, best_p3_ref=[best_p3])
    # final save
    model.save_pretrained(os.path.join(cfg.out_dir, "final"))
    tok.save_pretrained(os.path.join(cfg.out_dir, "final"))
    print("done.")


def _checkpoint_step(cfg, model, tok, eval_fn, probe_fn, base_probe, gstep,
                     best_p3_ref):
    if eval_fn is None:
        return
    model.eval()
    import torch
    with torch.no_grad():
        m = eval_fn(model, tok)
        probe = probe_fn(model, tok) if probe_fn else None
    model.train()

    p3 = m.get("P3_feedback_value", float("nan"))
    print(f"[eval s{gstep}] P3={p3} RR={m.get('recovery_rate')} "
          f"rb_prec={m.get('rollback_precision')} ece={m.get('ece')} "
          f"probe={probe}")
    _log_jsonl(cfg, {"step": gstep, "probe": probe, **m})

    # forgetting tripwire
    if base_probe is not None and probe is not None and \
       (base_probe - probe) > cfg.forgetting_tol:
        print(f"!! forgetting tripwire: probe {base_probe:.3f}->{probe:.3f}; stopping")
        raise SystemExit(0)

    # checkpoint on best P3 (decisive recursion signal)
    if p3 == p3 and p3 > best_p3_ref[0]:
        best_p3_ref[0] = p3
        path = os.path.join(cfg.out_dir, "best_p3")
        model.save_pretrained(path); tok.save_pretrained(path)
        print(f"   saved best P3={p3:.3f} -> {path}")


def _log_jsonl(cfg, row):
    with open(os.path.join(cfg.out_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in row.items()}) + "\n")


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="RLM-0 QLoRA trainer")
    cfg = TrainConfig()
    for f, v in asdict(cfg).items():
        p.add_argument(f"--{f}", type=type(v), default=v)
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    # eval_fn/probe_fn are injected by a wrapper that builds a ModelRunner
    # from training.eval; left None here so the driver runs standalone.
    train(_parse_args())
