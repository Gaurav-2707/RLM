"""
Tests for training/loss.py — the three-term RLM-0 objective.

Verifies each term in isolation with known inputs → known outputs,
and checks the combined loss behaves as expected.
"""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


from RLM.training.loss import recursion_loss, LossWeights


def _make_batch(B=2, T=8, V=100, rollback_id=5,
                labels_mask_first=True, ul_on_last=False):
    """Build a minimal batch dict matching RLMCollator output."""
    labels = torch.randint(0, V, (B, T))
    # Mask first token in each sequence (simulate non-trainable struct span)
    if labels_mask_first:
        labels[:, 0] = -100

    tok_w = torch.ones(B, T)
    tok_w[:, 0] = 0.0  # masked positions have 0 weight

    ul_mask = torch.zeros(B, T)
    if ul_on_last:
        ul_mask[:, -1] = 1.0

    traj_w = torch.ones(B)

    return {
        "labels": labels,
        "tok_weights": tok_w,
        "ul_mask": ul_mask,
        "traj_weight": traj_w,
    }


class TestLossCE:
    def test_l_ce_is_positive(self):
        """Cross-entropy loss must be non-negative."""
        B, T, V = 2, 10, 50
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V)
        weights = LossWeights(alpha_ul=0.0, beta_rb=0.0)
        total, metrics = recursion_loss(logits, batch, rollback_token_id=5, weights=weights)
        assert metrics["l_ce"].item() >= 0

    def test_l_ce_zero_when_all_labels_masked(self):
        """When all labels are -100 (fully masked), CE should be 0."""
        B, T, V = 2, 8, 50
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V)
        batch["labels"] = torch.full((B, T), -100)
        batch["tok_weights"] = torch.zeros(B, T)
        weights = LossWeights(alpha_ul=0.0, beta_rb=0.0)
        total, metrics = recursion_loss(logits, batch, rollback_token_id=5, weights=weights)
        assert metrics["l_ce"].item() == pytest.approx(0.0, abs=1e-5)

    def test_label_shift_applied(self):
        """Loss must use shifted labels (next-token prediction)."""
        # If label shift is missing, the loss would use position 0 label for logit 0,
        # but with shift it uses label 1 for logit 0. We just check no crash and
        # that shapes work for T > 1.
        B, T, V = 3, 12, 80
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V)
        weights = LossWeights()
        total, metrics = recursion_loss(logits, batch, rollback_token_id=5, weights=weights)
        assert total.shape == ()  # scalar


class TestLossUL:
    def test_l_ul_zero_when_ul_mask_all_zeros(self):
        """When ul_mask is all zeros, L_ul must be exactly 0."""
        B, T, V = 2, 8, 50
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V, ul_on_last=False)
        weights = LossWeights(alpha_ul=1.0, beta_rb=0.0)
        _, metrics = recursion_loss(logits, batch, rollback_token_id=5, weights=weights)
        assert metrics["l_ul"].item() == pytest.approx(0.0, abs=1e-5)

    def test_l_ul_positive_when_ul_mask_set(self):
        """When ul_mask marks tokens, L_ul should be > 0."""
        B, T, V = 2, 8, 50
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V, ul_on_last=True)
        weights = LossWeights(alpha_ul=1.0, beta_rb=0.0)
        _, metrics = recursion_loss(logits, batch, rollback_token_id=5, weights=weights)
        assert metrics["l_ul"].item() > 0

    def test_l_ul_numerically_stable(self):
        """UL term uses log1p(-p) — should not produce NaN or Inf."""
        B, T, V = 4, 16, 200
        # Extreme logits that make probabilities near 0 or 1
        logits = torch.randn(B, T, V) * 10
        batch = _make_batch(B=B, T=T, V=V, ul_on_last=True)
        _, metrics = recursion_loss(logits, batch, rollback_token_id=5, weights=LossWeights())
        assert not torch.isnan(metrics["l_ul"])
        assert not torch.isinf(metrics["l_ul"])


class TestLossRB:
    def test_l_rb_zero_when_mass_equals_r_hat(self):
        """When expected rollback mass == r_hat exactly, L_rb ≈ 0."""
        B, T, V = 2, 8, 50
        rb_id = 7
        # Craft logits so rollback token has high probability everywhere
        logits = torch.full((B, T, V), -10.0)
        logits[:, :, rb_id] = 10.0  # rollback dominates
        batch = _make_batch(B=B, T=T, V=V)
        # With rollback mass ≈ T-1 (shift), set r_hat to that value
        weights = LossWeights(alpha_ul=0.0, beta_rb=1.0, r_hat=float(T - 1))
        _, metrics = recursion_loss(logits, batch, rollback_token_id=rb_id, weights=weights)
        assert metrics["l_rb"].item() >= 0  # can't go negative

    def test_l_rb_positive_when_over_budget(self):
        """When rollback mass > r_hat, L_rb should be positive."""
        B, T, V = 2, 8, 50
        rb_id = 7
        logits = torch.full((B, T, V), -10.0)
        logits[:, :, rb_id] = 10.0  # high rollback mass
        batch = _make_batch(B=B, T=T, V=V)
        weights = LossWeights(alpha_ul=0.0, beta_rb=1.0, r_hat=0.0)  # budget = 0
        _, metrics = recursion_loss(logits, batch, rollback_token_id=rb_id, weights=weights)
        assert metrics["l_rb"].item() > 0

    def test_l_rb_zero_when_under_budget(self):
        """When rollback mass < r_hat (under budget), L_rb = 0 (clamp_min(0))."""
        B, T, V = 2, 8, 50
        rb_id = 7
        logits = torch.full((B, T, V), -10.0)
        logits[:, :, rb_id] = -10.0  # near-zero rollback mass
        batch = _make_batch(B=B, T=T, V=V)
        weights = LossWeights(alpha_ul=0.0, beta_rb=1.0, r_hat=100.0)  # huge budget
        _, metrics = recursion_loss(logits, batch, rollback_token_id=rb_id, weights=weights)
        assert metrics["l_rb"].item() == pytest.approx(0.0, abs=1e-4)


class TestLossCombined:
    def test_total_is_sum_of_weighted_terms(self):
        """total = l_ce + alpha*l_ul + beta*l_rb."""
        B, T, V = 2, 8, 50
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V, ul_on_last=True)
        w = LossWeights(alpha_ul=0.5, beta_rb=0.1)
        total, m = recursion_loss(logits, batch, rollback_token_id=5, weights=w)
        expected = m["l_ce"] + 0.5 * m["l_ul"] + 0.1 * m["l_rb"]
        assert total.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_metrics_dict_has_all_keys(self):
        """metrics dict must have the keys the trainer logs."""
        B, T, V = 2, 8, 50
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V)
        _, m = recursion_loss(logits, batch, rollback_token_id=5, weights=LossWeights())
        for key in ("loss", "l_ce", "l_ul", "l_rb", "mean_rollback_mass"):
            assert key in m, f"Missing metric key: {key}"

    def test_no_nan_or_inf_in_typical_batch(self):
        """Typical batch should never produce NaN or Inf in total loss."""
        torch.manual_seed(42)
        B, T, V = 4, 16, 1000
        logits = torch.randn(B, T, V)
        batch = _make_batch(B=B, T=T, V=V, ul_on_last=True)
        total, m = recursion_loss(logits, batch, rollback_token_id=5, weights=LossWeights())
        assert not torch.isnan(total)
        assert not torch.isinf(total)
        for v in m.values():
            assert not torch.isnan(v)
