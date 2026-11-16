import numpy as np
import pytest
from scipy.stats import chi2_contingency

from abtest.core import Variant, bayesian, peeking_false_positive_rate, sample_size, z_test


def test_z_test_matches_chi_square():
    a, b = Variant(10_000, 520), Variant(10_000, 580)
    table = np.array([[520, 10_000 - 520], [580, 10_000 - 580]])
    _, p_chi, _, _ = chi2_contingency(table, correction=False)
    assert z_test(a, b).p_value == pytest.approx(p_chi, rel=1e-9)


def test_ci_contains_observed_difference():
    r = z_test(Variant(5000, 250), Variant(5000, 300))
    assert r.ci_low < r.lift_abs < r.ci_high


def test_identical_variants_are_not_significant():
    r = z_test(Variant(1000, 50), Variant(1000, 50))
    assert r.p_value == pytest.approx(1.0) and not r.significant


def test_bayesian_agrees_in_direction():
    clear = bayesian(Variant(10_000, 500), Variant(10_000, 650))
    assert clear.prob_b_better > 0.99
    assert clear.expected_loss_choosing_b < clear.expected_loss_choosing_a


def test_sample_size_matches_standard_calculators():
    # 5% baseline, +10% relative lift, alpha .05, power .8 -> about 31k per arm
    assert 31_000 <= sample_size(0.05, 0.10) <= 31_500


def test_smaller_effects_need_more_traffic():
    assert sample_size(0.05, 0.05) > 3 * sample_size(0.05, 0.10)


def test_peeking_inflates_false_positives():
    r = peeking_false_positive_rate(sims=600)
    assert r["check_once_at_end"] < 0.08
    assert r["peek_daily_stop_early"] > 2 * r["check_once_at_end"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        Variant(100, 101)
