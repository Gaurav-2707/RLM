# RLM-0 Training Package

Turns objectively test-verified `RLMTrajectory` objects (see
`utils/trajectory.py`) into a reward-weighted SFT corpus for Llama 3.1 8B
(QLoRA), and provides the **P1–P4 recursion-proof** eval harness.

## Modules

| Module | Responsibility |
|---|---|
| `tokens.py` | Control-token vocab + tokenizer surgery (mean-init new rows). Confidence → 5 buckets. |
| `render.py` | `RLMTrajectory` → grammar spans. Masks world-supplied spans (state, verdict); flags final-failed-action for unlikelihood; computes `traj_weight`. |
| `collator.py` | `encode_trajectory` (span-aligned `input_ids/labels/tok_weights/ul_mask`, no mask drift) + `RLMCollator` (pad + carry per-example weights). |
| `loss.py` | Three-term objective: masked reward/role-weighted CE + unlikelihood + rollback-budget penalty. |
| `eval.py` | P1 verdict-flip, P2 novel-error, P3 feedback-ablation (`RR_with − RR_without`), P4 rollback-necessity; rollback P/R, ECE; pass/fail gate. |

## Loss masking rules

- **Trainable:** `<reason>`, `<action>`, `<rollback>`, `<conf_*>`, `<final>`.
- **Masked (context):** `<task>`, `<state>`, `<test> PASS/FAIL`. The verdict is
  world-supplied — training on it teaches the model to hallucinate `PASS`.
- **Decision tokens** (`action`/`rollback`/`conf`) get `tok_weight = 2.0`.

## Wiring a training step

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from training import add_control_tokens, render_trajectory, RLMCollator
from training.loss import recursion_loss, LossWeights
from training.tokens import special_id

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)
add_control_tokens(tok, model)                 # resize + mean-init
model = get_peft_model(model, LoraConfig(      # see spec §3
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    modules_to_save=["embed_tokens","lm_head"]))

rendered = [render_trajectory(t) for t in trajectories]   # t: RLMTrajectory
collate  = RLMCollator(tok, max_seq_len=16384)
rb_id    = special_id(tok, "<rollback>")
W        = LossWeights(r_hat=mean_rollbacks_among_successes)

batch  = collate(rendered[:B])                  # -> dict of tensors
logits = model(input_ids=batch["input_ids"],
               attention_mask=batch["attention_mask"]).logits
loss, metrics = recursion_loss(logits, batch, rb_id, W)
loss.backward()
```

## Evaluation

Implement a `training.eval.ModelRunner` against your live model+env, then:

```python
from training.eval import recursion_proof
metrics = recursion_proof(runner, held_out_tasks)
assert metrics["passes_recursion_bar"]   # P3≥0.25, rb_prec≥0.7, ECE≤0.1, P1≥0.8
```

**P3 (`feedback_value = RR_with − RR_without`) is the decisive metric** — if
ablating the test verdict doesn't drop recovery rate, the model isn't actually
recursing. Checkpoint on best P3.

> Note: `loss.py` and `collator.py` import `torch` lazily, so `render.py` /
> `tokens.py` are usable for data prep without a GPU stack installed.
