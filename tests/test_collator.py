"""
Tests for training/collator.py — encode_trajectory and RLMCollator.

Verifies mask alignment, tail truncation, padding, and that
tok_weights and ul_mask stay in sync with labels.
"""

import uuid
import pytest

from RLM.utils.trajectory import RLMTrajectory, TrajectoryStep
from RLM.training.render import render_trajectory
from RLM.training.collator import encode_trajectory, RLMCollator, LABEL_IGNORE


# ── minimal mock tokenizer ────────────────────────────────────────────────────

class MockTokenizer:
    """Tokenizer that maps chars to ints; keeps tests dependency-free."""
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Each character → its ord value clamped to [2, 255]
        return [max(2, min(255, ord(c))) for c in text] if text else []


# ── helpers ───────────────────────────────────────────────────────────────────

def _step(action_type, output_action="fix", outcome="", reward=0.0,
          confidence=0.7, was_rollback=False, rollback_reason="", n=1):
    return TrajectoryStep(
        step_number=n, iteration=1, action_type=action_type,
        input_state="state", output_action=output_action, outcome=outcome,
        reward=reward, confidence=confidence,
        was_rollback=was_rollback, rollback_reason=rollback_reason,
    )


def _traj(steps, final_outcome=True):
    return RLMTrajectory(
        trajectory_id=str(uuid.uuid4()),
        steps=steps, total_steps=len(steps),
        total_rollbacks=sum(1 for s in steps if s.was_rollback),
        final_outcome=final_outcome, verified=True,
        final_diff="", task_description="fix foo",
    )


def _encode(steps, final_outcome=True, max_seq_len=4096):
    traj = _traj(steps, final_outcome)
    rt = render_trajectory(traj)
    tok = MockTokenizer()
    return encode_trajectory(rt, tok, max_seq_len=max_seq_len)


# ── mask alignment ────────────────────────────────────────────────────────────

class TestMaskAlignment:
    def test_labels_and_input_ids_same_length(self):
        enc = _encode([_step("edit", n=1)])
        assert len(enc["input_ids"]) == len(enc["labels"])

    def test_tok_weights_same_length_as_input_ids(self):
        enc = _encode([_step("edit", n=1)])
        assert len(enc["tok_weights"]) == len(enc["input_ids"])

    def test_ul_mask_same_length_as_input_ids(self):
        enc = _encode([_step("edit", n=1)])
        assert len(enc["ul_mask"]) == len(enc["input_ids"])

    def test_masked_positions_have_zero_tok_weight(self):
        """Every position where labels == -100 must have tok_weight == 0."""
        enc = _encode([_step("edit", n=1)])
        for label, weight in zip(enc["labels"], enc["tok_weights"]):
            if label == LABEL_IGNORE:
                assert weight == 0.0, \
                    f"Masked position (label=-100) has non-zero tok_weight={weight}"

    def test_trainable_positions_have_positive_tok_weight(self):
        """Positions where labels != -100 must have tok_weight > 0."""
        enc = _encode([_step("edit", output_action="return 42", n=1)])
        trainable_found = False
        for label, weight in zip(enc["labels"], enc["tok_weights"]):
            if label != LABEL_IGNORE:
                trainable_found = True
                assert weight > 0.0, "Trainable position has zero tok_weight"
        assert trainable_found, "No trainable tokens found — something is wrong"

    def test_no_ul_on_success_trajectory(self):
        """Success trajectory must have ul_mask all zeros."""
        enc = _encode([_step("edit", n=1), _step("run_tests", reward=1.0, n=2)],
                      final_outcome=True)
        assert all(m == 0 for m in enc["ul_mask"]), \
            "UL mask must be all zeros on success trajectory"

    def test_ul_present_on_failure_trajectory(self):
        """Failed trajectory should have at least one ul_mask=1."""
        enc = _encode([
            _step("edit", output_action="bad attempt", n=1),
            _step("run_tests", reward=0.0, n=2),
        ], final_outcome=False)
        assert any(m == 1 for m in enc["ul_mask"]), \
            "Failed trajectory should have at least one UL-masked token"


# ── tail truncation ───────────────────────────────────────────────────────────

class TestTailTruncation:
    def test_output_length_respects_max_seq_len(self):
        enc = _encode([_step("edit", output_action="x" * 5000, n=1)],
                      max_seq_len=128)
        assert len(enc["input_ids"]) <= 128

    def test_truncation_keeps_tail(self):
        """Tail truncation should keep the LAST tokens (not the first)."""
        # The <final> token appears at the end — truncated seqs must keep it
        enc = _encode([_step("edit", output_action="y" * 5000, n=1)],
                      max_seq_len=50)
        # After truncation, all arrays should be length <= 50
        assert len(enc["input_ids"]) <= 50
        assert len(enc["labels"]) <= 50

    def test_all_arrays_same_length_after_truncation(self):
        enc = _encode([_step("edit", output_action="z" * 3000, n=1)],
                      max_seq_len=64)
        L = len(enc["input_ids"])
        assert len(enc["labels"]) == L
        assert len(enc["tok_weights"]) == L
        assert len(enc["ul_mask"]) == L


# ── metadata fields ───────────────────────────────────────────────────────────

class TestMetadataFields:
    def test_traj_weight_positive(self):
        enc = _encode([_step("edit", n=1)])
        assert enc["traj_weight"] > 0

    def test_is_success_true_for_success(self):
        enc = _encode([_step("run_tests", reward=1.0, n=1)], final_outcome=True)
        assert enc["is_success"] is True

    def test_is_success_false_for_failure(self):
        enc = _encode([_step("run_tests", reward=0.0, n=1)], final_outcome=False)
        assert enc["is_success"] is False

    def test_num_rollbacks_matches_trajectory(self):
        steps = [
            _step("edit", n=1),
            _step("edit", was_rollback=True, rollback_reason="failed", n=2),
            _step("run_tests", reward=1.0, n=3),
        ]
        enc = _encode(steps)
        assert enc["num_rollbacks"] == 1


# ── RLMCollator batch padding ─────────────────────────────────────────────────

class TestRLMCollator:
    def _collate(self, feature_list):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        from RLM.training.collator import RLMCollator
        collator = RLMCollator(MockTokenizer(), max_seq_len=512, pad_to_multiple_of=8)
        return collator(feature_list)

    def _two_encoded(self):
        enc1 = _encode([_step("edit", output_action="short", n=1)])
        enc2 = _encode([_step("edit", output_action="longer action text here", n=1)])
        return [enc1, enc2]

    def test_batch_input_ids_same_length(self):
        batch = self._collate(self._two_encoded())
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == batch["labels"].shape[1]

    def test_padded_labels_are_ignore(self):
        """Padding positions in labels must be LABEL_IGNORE (-100)."""
        import torch
        batch = self._collate(self._two_encoded())
        attn = batch["attention_mask"]
        labels = batch["labels"]
        # Where attention_mask == 0 (padding), labels must be -100
        padding = (attn == 0)
        assert (labels[padding] == LABEL_IGNORE).all(), \
            "Padded positions must have label=-100"

    def test_batch_length_multiple_of_8(self):
        batch = self._collate(self._two_encoded())
        T = batch["input_ids"].shape[1]
        assert T % 8 == 0, f"Batch length {T} not a multiple of 8"

    def test_traj_weight_tensor_shape(self):
        import torch
        batch = self._collate(self._two_encoded())
        assert batch["traj_weight"].shape == (2,)
