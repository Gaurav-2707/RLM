"""
tests/test_acc.py
=================
Unit tests for the Adaptive Compute Controller (ACC).

Run with:
    pytest tests/test_acc.py -v
"""

import pytest
from RLM.acc import AdaptiveComputeController, ComplexityScorer
from RLM.acc.models import DepthRecord, EpisodeReport


# ===========================================================================
# ComplexityScorer
# ===========================================================================

class TestComplexityScorer:

    def setup_method(self):
        self.scorer = ComplexityScorer()

    # -- basic range ----------------------------------------------------------

    def test_score_is_in_unit_interval(self):
        queries = [
            "What is 2 + 2?",
            "Why does quantum entanglement violate classical locality?",
            "Analyse the strategic implications of distributed systems design.",
            "",  # edge: empty string
        ]
        for q in queries:
            s = self.scorer.score(q)
            assert 0.0 <= s <= 1.0, f"Score out of range for query: {q!r}"

    def test_empty_query_returns_zero(self):
        assert self.scorer.score("") == 0.0
        assert self.scorer.score("   ") == 0.0

    # -- ordering: shallow < deep --------------------------------------------

    def test_simple_question_scores_lower_than_complex(self):
        simple  = self.scorer.score("What is the capital of France?")
        complex_ = self.scorer.score(
            "Analyse and evaluate the geopolitical implications of EU "
            "expansion on eastern European economies."
        )
        assert simple < complex_, (
            f"Expected simple ({simple:.4f}) < complex ({complex_:.4f})"
        )

    # -- context signal -------------------------------------------------------

    def test_large_context_raises_score(self):
        q = "Summarise the document."
        score_no_ctx  = self.scorer.score(q)
        score_big_ctx = self.scorer.score(q, context="x" * 900_000)
        assert score_big_ctx > score_no_ctx

    # -- custom weights -------------------------------------------------------

    def test_custom_weights_still_normalised(self):
        scorer = ComplexityScorer(weights=(1, 1, 1))
        s = scorer.score("Why do objects fall due to gravity?")
        assert 0.0 <= s <= 1.0

    def test_invalid_context_weight_raises(self):
        with pytest.raises(ValueError):
            ComplexityScorer(context_weight=1.5)

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError):
            ComplexityScorer(weights=(1, 2))  # wrong length


# ===========================================================================
# AdaptiveComputeController – depth mapping
# ===========================================================================

class TestDepthMapping:

    def setup_method(self):
        self.acc = AdaptiveComputeController()
        self.acc.new_episode()

    def test_shallow_score_gives_depth_1(self):
        assert self.acc.select_depth(0.00) == 1
        assert self.acc.select_depth(0.10) == 1
        assert self.acc.select_depth(0.34) == 1

    def test_boundary_shallow_threshold(self):
        # 0.35 is the first score that enters medium tier
        assert self.acc.select_depth(0.35) == 2

    def test_medium_score_gives_depth_2(self):
        assert self.acc.select_depth(0.35) == 2
        assert self.acc.select_depth(0.50) == 2
        assert self.acc.select_depth(0.70) == 2

    def test_boundary_deep_threshold(self):
        # 0.70 is still medium; just above is deep
        self.acc.new_episode()
        assert self.acc.select_depth(0.70) == 2
        self.acc.new_episode()
        assert self.acc.select_depth(0.71) == 3

    def test_deep_score_gives_depth_3(self):
        self.acc.new_episode()
        assert self.acc.select_depth(0.71) == 3
        self.acc.new_episode()
        assert self.acc.select_depth(1.00) == 3


# ===========================================================================
# AdaptiveComputeController – budget enforcement
# ===========================================================================

class TestBudgetEnforcement:

    def test_no_budget_never_exhausts(self):
        acc = AdaptiveComputeController(max_api_calls=None)
        acc.new_episode()
        for _ in range(100):
            depth = acc.select_depth(0.9)
            assert depth == 3
        assert not acc.is_budget_exhausted

    def test_budget_caps_total_api_calls(self):
        acc = AdaptiveComputeController(max_api_calls=10)
        acc.new_episode()
        depths = []
        for _ in range(20):
            d = acc.select_depth(0.9)  # always wants depth 3 (cost=3)
            depths.append(d)
            if d == 0:
                break
        assert acc.api_calls_used <= 10

    def test_budget_exhaustion_returns_zero(self):
        acc = AdaptiveComputeController(max_api_calls=2)
        acc.new_episode()
        # depth 3 costs 3; depth 2 costs 2; depth 1 costs 1
        acc.select_depth(0.9)   # wants depth 3, but only 2 budget → gets depth 2
        result = acc.select_depth(0.9)  # 0 remaining → exhausted
        assert result == 0
        assert acc.is_budget_exhausted

    def test_depth_clamped_when_budget_tight(self):
        acc = AdaptiveComputeController(max_api_calls=2)
        acc.new_episode()
        # Only 2 calls left → depth 3 (cost=3) is too expensive → falls back to depth 2 (cost=2)
        depth = acc.select_depth(0.9)
        assert depth == 2  # clamped from 3

    def test_remaining_budget_property(self):
        acc = AdaptiveComputeController(max_api_calls=10)
        acc.new_episode()
        assert acc.remaining_budget == 10
        acc.select_depth(0.1)   # depth 1, cost 1
        assert acc.remaining_budget == 9
        acc.select_depth(0.5)   # depth 2, cost 2
        assert acc.remaining_budget == 7

    def test_remaining_budget_is_none_without_cap(self):
        acc = AdaptiveComputeController(max_api_calls=None)
        acc.new_episode()
        assert acc.remaining_budget is None


# ===========================================================================
# AdaptiveComputeController – episode lifecycle & records
# ===========================================================================

class TestEpisodeLifecycle:

    def test_new_episode_resets_state(self):
        acc = AdaptiveComputeController(max_api_calls=20)
        acc.new_episode()
        acc.select_depth(0.9)
        acc.select_depth(0.1)
        # Second episode should start fresh
        acc.new_episode()
        assert acc.api_calls_used == 0
        assert acc.records == []
        assert not acc.is_budget_exhausted

    def test_records_have_correct_step_indices(self):
        acc = AdaptiveComputeController()
        acc.new_episode()
        scores = [0.1, 0.5, 0.9]
        for s in scores:
            acc.select_depth(s)
        records = acc.records
        assert [r.step for r in records] == [0, 1, 2]

    def test_records_store_correct_depth(self):
        acc = AdaptiveComputeController()
        acc.new_episode()
        acc.select_depth(0.2)   # depth 1
        acc.select_depth(0.55)  # depth 2
        acc.select_depth(0.85)  # depth 3
        depths = [r.depth_selected for r in acc.records]
        assert depths == [1, 2, 3]

    def test_records_store_complexity_score(self):
        acc = AdaptiveComputeController()
        acc.new_episode()
        acc.select_depth(0.42)
        assert abs(acc.records[0].complexity_score - 0.42) < 1e-9

    def test_end_episode_returns_report(self):
        acc = AdaptiveComputeController(max_api_calls=20)
        acc.new_episode()
        acc.select_depth(0.1)
        acc.select_depth(0.9)
        report = acc.end_episode()
        assert isinstance(report, EpisodeReport)
        assert len(report.records) == 2
        assert report.total_api_calls == 4   # depth1(1) + depth3(3)
        assert not report.budget_exhausted


# ===========================================================================
# EpisodeReport – analytics
# ===========================================================================

class TestEpisodeReport:

    def _make_report(self, scores: list[float]) -> EpisodeReport:
        acc = AdaptiveComputeController()
        acc.new_episode()
        for s in scores:
            acc.select_depth(s)
        return acc.end_episode()

    def test_depth_distribution(self):
        report = self._make_report([0.1, 0.5, 0.9, 0.2, 0.6])
        dist = report.depth_distribution
        assert dist[1] == 2  # 0.1, 0.2
        assert dist[2] == 2  # 0.5, 0.6
        assert dist[3] == 1  # 0.9

    def test_is_non_uniform_true(self):
        report = self._make_report([0.1, 0.9])
        assert report.is_non_uniform is True

    def test_is_non_uniform_false(self):
        # All shallow → uniform depth 1
        report = self._make_report([0.1, 0.2, 0.3])
        assert report.is_non_uniform is False

    def test_average_complexity(self):
        report = self._make_report([0.2, 0.4, 0.6])
        assert abs(report.average_complexity - 0.4) < 1e-6

    def test_correlation_positive_for_monotone_scores(self):
        # Monotonically increasing scores → perfect positive correlation
        report = self._make_report([0.1, 0.4, 0.8])
        assert report.depth_complexity_correlation > 0.9

    def test_summary_keys(self):
        report = self._make_report([0.1, 0.5, 0.9])
        summary = report.summary()
        required_keys = {
            "total_steps",
            "total_api_calls",
            "budget_exhausted",
            "depth_distribution",
            "is_non_uniform_distribution",
            "average_complexity",
            "depth_complexity_correlation",
        }
        assert required_keys.issubset(summary.keys())
