"""Count distributions and metrics, including that the true distribution scores best."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from footy.evaluate import (
    CountDistribution,
    estimate_dispersion,
    expected_calibration_error,
    poisson_deviance,
    score,
)


def test_poisson_special_case_matches_scipy() -> None:
    """alpha = 0 must reproduce the Poisson exactly, not merely approximately."""
    mu = np.array([0.3, 1.0, 2.5, 7.0])
    distribution = CountDistribution(mu, np.zeros_like(mu))

    for k in range(6):
        np.testing.assert_allclose(
            distribution.pmf(np.full_like(mu, k)), stats.poisson.pmf(k, mu), rtol=1e-4
        )


def test_dispersion_widens_the_distribution() -> None:
    mu = np.full(4, 2.0)
    tight = CountDistribution(mu, np.zeros_like(mu))
    loose = CountDistribution(mu, np.full_like(mu, 1.5))

    # more dispersion puts more probability on zero and on large counts
    assert loose.pmf(np.zeros(4))[0] > tight.pmf(np.zeros(4))[0]
    assert loose.prob_at_least(6)[0] > tight.prob_at_least(6)[0]


def test_log_score_is_proper() -> None:
    """The distribution that generated the data gets the best log score."""
    rng = np.random.default_rng(0)
    truth = 1.5
    y = rng.poisson(truth, size=20000).astype(float)

    correct = CountDistribution(np.full(len(y), truth), np.zeros(len(y)))
    too_low = CountDistribution(np.full(len(y), 0.6), np.zeros(len(y)))
    too_high = CountDistribution(np.full(len(y), 3.0), np.zeros(len(y)))

    assert correct.log_score(y).mean() < too_low.log_score(y).mean()
    assert correct.log_score(y).mean() < too_high.log_score(y).mean()


def test_crps_is_proper_and_finite_on_surprises() -> None:
    """CRPS prefers the true distribution and stays finite on a near-impossible count."""
    rng = np.random.default_rng(1)
    y = rng.poisson(2.0, size=5000).astype(float)

    correct = CountDistribution(np.full(len(y), 2.0), np.zeros(len(y)))
    wrong = CountDistribution(np.full(len(y), 8.0), np.zeros(len(y)))
    assert correct.crps(y).mean() < wrong.crps(y).mean()

    # a near-impossible count, where the log score would blow up
    shock = CountDistribution(np.array([0.001]), np.array([0.0]))
    assert np.isfinite(shock.crps(np.array([12.0]))).all()


def test_prob_at_least_is_monotone() -> None:
    distribution = CountDistribution(np.array([1.2]), np.array([0.4]))
    probabilities = [distribution.prob_at_least(k)[0] for k in range(6)]
    assert probabilities == sorted(probabilities, reverse=True)
    assert probabilities[0] == pytest.approx(1.0)


def test_poisson_deviance_is_zero_at_perfect_prediction() -> None:
    y = np.array([0.0, 1.0, 4.0])
    # mu is floored at 1e-9, so a perfect zero prediction leaves a tiny residual
    np.testing.assert_allclose(poisson_deviance(y, y), 0.0, atol=1e-8)
    assert (poisson_deviance(y, y + 1.5) > 0).all()


def test_calibration_error_detects_a_biased_model() -> None:
    rng = np.random.default_rng(2)
    y = rng.poisson(1.0, size=8000).astype(float)

    honest = CountDistribution(np.full(len(y), 1.0), np.zeros(len(y)))
    biased = CountDistribution(np.full(len(y), 4.0), np.zeros(len(y)))

    assert expected_calibration_error(y, honest) < expected_calibration_error(y, biased)


def test_dispersion_estimate_recovers_overdispersion() -> None:
    rng = np.random.default_rng(3)
    mu = np.full(50000, 2.0)
    # gamma-Poisson mixture with alpha = 0.5
    rates = rng.gamma(shape=2.0, scale=1.0, size=len(mu))
    y = rng.poisson(rates).astype(float)

    estimated = estimate_dispersion(y, np.full(len(y), y.mean()))
    assert estimated > 0.2, "clear overdispersion should not be estimated as Poisson"


def test_score_reports_every_metric() -> None:
    y = np.array([0.0, 1.0, 2.0, 1.0])
    distribution = CountDistribution(np.full(4, 1.0), np.zeros(4))
    result = score(y, distribution, label="test")

    for key in ("MAE", "RMSE", "PoissonDev", "LogScore", "CRPS", "P>=1_pred", "P>=1_obs"):
        assert key in result and np.isfinite(result[key])
    assert result["model"] == "test"
