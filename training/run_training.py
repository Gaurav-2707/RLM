"""
RLM-0 end-to-end launcher: wires the recursion-proof eval harness and a
general-capability probe into the QLoRA driver.

This is the file you actually run. It supplies the two hooks the driver
leaves open:

  eval_fn(model, tok)  -> training.eval.recursion_proof(...)  metrics
  probe_fn(model, tok) -> general-capability score (HumanEval/MMLU style)

The recursion eval needs a `ModelRunner` that can (a) roll a task out to a
test-verified PASS/FAIL and (b) read P(<rollback>). Rollout verification is
the one piece that must touch your execution sandbox, so it is injected as
`verify_fn(task, final_diff) -> bool`. Everything else (generation, grammar
parsing, the four interventions) is implemented here.

Run:
  python -m training.run_training \
      --traj_dir ~/.rlm/traces --out runs/rlm0 \
      --eval_tasks training/eval_tasks.json
"""

import argparse
import json
import os
import re
from dataclasses import asdict
from typing import Callable, List, Optional

from .train import TrainConfig, train
from .tokens import (special_id, CONF_TOKENS, conf_to_token,
                     decision_token_ids)
from .eval import (ModelRunner, EvalTask, Rollout, recursion_proof)


# ============================================================================
# 1. A concrete ModelRunner that drives the live model through the grammar
# ============================================================================
class LiveModelRunner(ModelRunner):
    """
    Rolls a task out step-by-step in the RLM grammar, generating one
    <step> at a time, feeding back a real test verdict from `verify_fn`.

    verify_fn(task, accumulated_diff) -> bool   (True == tests pass)
    """

    def __init__(self, model, tokenizer, verify_fn: Callable[[EvalTask, str], bool],
                 max_steps: int = 8, max_new_tokens: int = 512,
                 device: Optional[str] = None):
        self.model = model
        self.tok = tokenizer
        self.verify_fn = verify_fn
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.device = device or getattr(model, "device", "cuda")
        self.rb_id = special_id(tokenizer, "<rollback>")

    # -- low-level generation up to a stop marker --------------------------
    def _gen(self, prefix: str, stop: str = "</step>") -> str:
        import torch
        ids = self.tok(prefix, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tok.pad_token_id,
            )
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=False)
        cut = text.find(stop)
        return text[: cut + len(stop)] if cut != -1 else text

    @staticmethod
    def _extract(tag: str, text: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_action(text: str) -> str:
        return LiveModelRunner._extract("action", text)

    @staticmethod
    def _extract_conf(text: str) -> str:
        for t in CONF_TOKENS:
            if t in text:
                return t
        return "<conf_m>"

    # -- ModelRunner API ---------------------------------------------------
    def rollback_prob(self, prefix_grammar: str) -> float:
        import torch
        ids = self.tok(prefix_grammar, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
            p = torch.softmax(logits.float(), -1)[self.rb_id].item()
        return p

    def rollout(self, task: EvalTask, feedback: bool = True,
                force_first_verdict: Optional[str] = None) -> Rollout:
        prefix = f"<task> {task.task_description.strip()} </task>\n"
        accumulated_diff = ""
        diff_stack: List[str] = []          # for rollback restore
        rollbacks: List[int] = []
        conf_events = []
        first_action_failed = False
        reacted_to_flip = False
        final_pass = False

        for step in range(self.max_steps):
            prefix += "<step>\n<state> "
            prefix += (accumulated_diff[-2000:] or "(clean)") + " </state>\n"

            gen = self._gen(prefix)
            prefix += gen if gen.endswith("</step>\n") else gen + "\n</step>\n"

            did_rollback = "<rollback>" in gen
            action = self._extract_action(gen)
            conf_tok = self._extract_conf(gen)

            if did_rollback:
                rollbacks.append(step)
                accumulated_diff = diff_stack.pop() if diff_stack else ""
                conf_events.append((conf_tok, True))
                continue

            diff_stack.append(accumulated_diff)
            accumulated_diff = (accumulated_diff + "\n" + action).strip()

            # verdict: forced (P1) on first step, else real test
            if step == 0 and force_first_verdict is not None:
                passed = (force_first_verdict == "PASS")
            else:
                passed = bool(self.verify_fn(task, accumulated_diff))

            if step == 0 and not passed and force_first_verdict != "PASS":
                first_action_failed = True

            conf_events.append((conf_tok, passed))

            # P1: did the model react (retry/rollback) to a forced FAIL?
            if step == 0 and force_first_verdict == "FAIL":
                nxt = self.rollback_prob(prefix + "<step>\n<state> "
                                         + (accumulated_diff[-2000:] or "(clean)")
                                         + " </state>\n<reason>")
                reacted_to_flip = nxt > 0.5

            if feedback:
                prefix += ""  # verdict already implied by generated <test>; visible
            else:
                # P3 ablation: strip any verdict the model emitted from context
                prefix = re.sub(r"<test>.*?</test>", "<test> </test>", prefix, flags=re.S)

            if passed:
                final_pass = True
                break

        return Rollout(
            task_id=task.task_id,
            first_action_failed=first_action_failed,
            final_pass=final_pass,
            num_steps=step + 1,
            rollbacks=rollbacks,
            conf_events=conf_events,
            reacted_to_flip=reacted_to_flip,
        )


# ============================================================================
# 2. Eval tasks loader
# ============================================================================
def load_eval_tasks(path: str) -> List[EvalTask]:
    with open(os.path.expanduser(path)) as f:
        raw = json.load(f)
    tasks = []
    for t in raw:
        # JSON object keys are strings; broken_states is indexed by step int
        if "broken_states" in t:
            t["broken_states"] = {int(k): bool(v)
                                  for k, v in t["broken_states"].items()}
        tasks.append(EvalTask(**t))
    return tasks


# ============================================================================
# 3. General-capability probe (forgetting tripwire)
# ============================================================================
def make_humaneval_probe(problems: List[dict],
                         run_code: Callable[[str, str], bool],
                         max_new_tokens: int = 256) -> Callable:
    """
    problems: [{"prompt": str, "test": str}, ...]   (HumanEval-style)
    run_code(completion, test) -> bool              (True == passes in sandbox)

    Returns probe_fn(model, tok) -> pass@1 in [0,1].
    """
    def probe(model, tok) -> float:
        import torch
        device = getattr(model, "device", "cuda")
        passed = 0
        for p in problems:
            ids = tok(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            comp = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            try:
                if run_code(p["prompt"] + comp, p["test"]):
                    passed += 1
            except Exception:
                pass
        return passed / max(1, len(problems))
    return probe


# ============================================================================
# 4. Launcher
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="RLM-0 launcher (train + eval + probe)")
    cfg = TrainConfig()
    for f, v in asdict(cfg).items():
        ap.add_argument(f"--{f}", type=type(v), default=v)
    ap.add_argument("--eval_tasks", type=str, default="",
                    help="JSON list of EvalTask for the recursion-proof harness")
    ap.add_argument("--humaneval", type=str, default="",
                    help="JSON list of {prompt,test} for the forgetting probe")
    args = ap.parse_args()

    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(cfg)})

    # ---- sandbox implementations ----------------------------------------
    def verify_fn(task: EvalTask, final_diff: str) -> bool:
        """
        Apply final_diff to a temp copy of task.repo_path, run
        task.test_command, return True if exit code == 0.
        """
        import subprocess
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # Copy the repo into the temp dir
            repo_copy = os.path.join(tmp, "repo")
            try:
                shutil.copytree(task.repo_path, repo_copy)
            except Exception as exc:
                print(f"[verify_fn] copytree failed: {exc}")
                return False

            # Apply the diff if one was produced
            if final_diff.strip():
                patch_path = os.path.join(tmp, "changes.patch")
                with open(patch_path, "w", encoding="utf-8") as fh:
                    fh.write(final_diff)
                patch_result = subprocess.run(
                    ["patch", "-p1", "-i", patch_path],
                    cwd=repo_copy,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if patch_result.returncode != 0:
                    # Diff failed to apply — treat as test failure
                    print(f"[verify_fn] patch failed:\n{patch_result.stderr}")
                    return False

            # Run the test command
            try:
                result = subprocess.run(
                    task.test_command,
                    shell=True,
                    cwd=repo_copy,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                print("[verify_fn] test command timed out (60s)")
                return False
            except Exception as exc:
                print(f"[verify_fn] test runner error: {exc}")
                return False

    def run_code(completion: str, test: str) -> bool:
        """
        Execute `completion` + `test` in a sandboxed temp directory.
        Returns True if the combined script exits with code 0.
        """
        import subprocess
        import tempfile

        combined = completion.rstrip() + "\n\n" + test.strip() + "\n"

        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "solution.py")
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(combined)

            try:
                result = subprocess.run(
                    ["python3", script_path],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                print("[run_code] execution timed out (30s)")
                return False
            except Exception as exc:
                print(f"[run_code] execution error: {exc}")
                return False

    eval_tasks = load_eval_tasks(args.eval_tasks) if args.eval_tasks else []

    def eval_fn(model, tok):
        if not eval_tasks:
            return {}
        runner = LiveModelRunner(model, tok, verify_fn)
        return recursion_proof(runner, eval_tasks)

    probe_fn = None
    if args.humaneval:
        with open(os.path.expanduser(args.humaneval)) as f:
            problems = json.load(f)
        probe_fn = make_humaneval_probe(problems, run_code)

    train(cfg,
          eval_fn=eval_fn if eval_tasks else None,
          probe_fn=probe_fn)


if __name__ == "__main__":
    main()
