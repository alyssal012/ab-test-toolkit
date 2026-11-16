"""A/B test analysis: frequentist, Bayesian, sample size, and the cost of peeking."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Variant:
    visitors: int
    conversions: int

    @property
    def rate(self) -> float:
        return self.conversions / self.visitors

    def __post_init__(self):
        if self.visitors <= 0 or not 0 <= self.conversions <= self.visitors:
            raise ValueError("need visitors > 0 and 0 <= conversions <= visitors")


@dataclass
class FrequentistResult:
    lift_abs: float
    lift_rel: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float
    significant: bool


def z_test(a: Variant, b: Variant, alpha: float = 0.05) -> FrequentistResult:
    """Two-sided two-proportion z-test, with a Wald CI on the difference."""
    pooled = (a.conversions + b.conversions) / (a.visitors + b.visitors)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / a.visitors + 1 / b.visitors))
    diff = b.rate - a.rate
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = math.sqrt(a.rate * (1 - a.rate) / a.visitors + b.rate * (1 - b.rate) / b.visitors)
    zc = stats.norm.ppf(1 - alpha / 2)
    return FrequentistResult(diff, diff / a.rate if a.rate else float("nan"),
                             diff - zc * se, diff + zc * se, z, p, p < alpha)


@dataclass
class BayesianResult:
    prob_b_better: float
    expected_loss_choosing_b: float
    expected_loss_choosing_a: float
    lift_rel_ci: tuple[float, float]


def bayesian(a: Variant, b: Variant, prior: tuple[float, float] = (1, 1),
             draws: int = 200_000, seed: int = 0) -> BayesianResult:
    """Beta-Binomial model. Answers the question stakeholders actually ask:
    'how likely is B better, and how much do we lose if we pick wrong?'"""
    rng = np.random.default_rng(seed)
    pa = rng.beta(prior[0] + a.conversions, prior[1] + a.visitors - a.conversions, draws)
    pb = rng.beta(prior[0] + b.conversions, prior[1] + b.visitors - b.conversions, draws)
    rel = pb / pa - 1
    return BayesianResult(float((pb > pa).mean()),
                          float(np.maximum(pa - pb, 0).mean()),
                          float(np.maximum(pb - pa, 0).mean()),
                          (float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))))


def sample_size(baseline: float, mde_rel: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Visitors needed *per variant* to detect a relative lift of mde_rel."""
    p1, p2 = baseline, baseline * (1 + mde_rel)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
         / (p2 - p1) ** 2)
    return math.ceil(n)


def peeking_false_positive_rate(baseline: float = 0.05, visitors_per_day: int = 500,
                                days: int = 28, alpha: float = 0.05,
                                sims: int = 2000, seed: int = 0) -> dict:
    """Simulate A/A tests (no real difference) and compare two habits:
    checking once at the end vs. checking every day and stopping at the first p < alpha."""
    rng = np.random.default_rng(seed)
    end_hits = peek_hits = 0
    for _ in range(sims):
        ca = rng.binomial(visitors_per_day, baseline, days).cumsum()
        cb = rng.binomial(visitors_per_day, baseline, days).cumsum()
        n = visitors_per_day * np.arange(1, days + 1)
        pooled = (ca + cb) / (2 * n)
        se = np.sqrt(pooled * (1 - pooled) * 2 / n)
        z = np.divide(cb / n - ca / n, se, out=np.zeros(days), where=se > 0)
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        end_hits += p[-1] < alpha
        peek_hits += (p < alpha).any()
    return {"check_once_at_end": end_hits / sims, "peek_daily_stop_early": peek_hits / sims,
            "nominal_alpha": alpha}
