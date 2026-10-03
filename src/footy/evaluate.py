"""Predictive distributions and the metrics that judge them.

Point estimates are not enough for this problem. "Chris Rigg will commit 0.8 fouls" is
not a statement anyone can act on; ``P(fouls >= 1) = 0.54`` is. So every model here emits
a distribution over counts, and the headline metrics are proper scoring rules that reward
being well calibrated rather than merely close on average.

MAE and RMSE are reported too, but they are the weakest metrics in the table: for a
target that is zero 56% of the time, predicting zero always scores respectably on both
while being useless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

#: Upper bound for the CRPS / calibration sums. The largest count in the Premier League
#: data is 9 fouls, so 40 is far into the tail for every target we model.
MAX_COUNT = 40


@dataclass
class CountDistribution:
    """A negative binomial predictive distribution, one per row.

    Parameterised by mean ``mu`` and dispersion ``alpha`` such that
    ``Var = mu + alpha * mu^2``. ``alpha = 0`` recovers the Poisson, so a Poisson model
    is just this class with zero dispersion rather than a separate code path.
    """

    mu: np.ndarray
    alpha: np.ndarray

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float).clip(min=1e-9)
        self.alpha = np.broadcast_to(
            np.asarray(self.alpha, dtype=float), self.mu.shape
        ).clip(min=0.0).copy()

    def __len__(self) -> int:
        return len(self.mu)

    def _nbinom_params(self) -> tuple[np.ndarray, np.ndarray]:
        # scipy's nbinom(n, p) has mean n(1-p)/p; solving for our (mu, alpha) gives
        # n = 1/alpha and p = n / (n + mu).
        alpha = np.where(self.alpha <= 1e-12, 1e-12, self.alpha)
        n = 1.0 / alpha
        p = n / (n + self.mu)
        return n, p

    def _align(self, k: np.ndarray | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Broadcast the per-row parameters against ``k``.

        ``k`` is either a scalar/1-D array (one value per row) or a 2-D grid of counts
        evaluated for every row, as CRPS needs. In the latter case the parameters gain a
        trailing axis so scipy broadcasts them across the grid.
        """
        n, p = self._nbinom_params()
        k = np.asarray(k)
        if k.ndim == 2:
            return k, n[:, None], p[:, None]
        return k, n, p

    def _mu_for(self, k: np.ndarray) -> np.ndarray:
        return self.mu[:, None] if k.ndim == 2 else self.mu

    def _evaluate(self, k: np.ndarray | int, nbinom_fn, poisson_fn) -> np.ndarray:
        """Use the exact Poisson wherever dispersion is zero.

        Approximating the Poisson as a negative binomial with a tiny alpha loses several
        digits, so the limiting case is computed directly rather than approached. In
        practice alpha is uniform across rows, so only one of the two branches runs --
        which matters, because these are evaluated over a 40-wide grid for CRPS.
        """
        k, n, p = self._align(k)
        poisson_rows = self.alpha <= 1e-12

        if poisson_rows.all():
            return poisson_fn(k, self._mu_for(k))
        if not poisson_rows.any():
            return nbinom_fn(k, n, p)

        mask = poisson_rows[:, None] if k.ndim == 2 else poisson_rows
        return np.where(mask, poisson_fn(k, self._mu_for(k)), nbinom_fn(k, n, p))

    def pmf(self, k: np.ndarray | int) -> np.ndarray:
        return self._evaluate(k, stats.nbinom.pmf, stats.poisson.pmf)

    def cdf(self, k: np.ndarray | int) -> np.ndarray:
        return self._evaluate(k, stats.nbinom.cdf, stats.poisson.cdf)

    def prob_at_least(self, k: int) -> np.ndarray:
        """P(X >= k) -- the form a betting line or a scouting note actually takes."""
        if k <= 0:
            return np.ones_like(self.mu)
        return 1.0 - self.cdf(k - 1)

    def log_score(self, y: np.ndarray) -> np.ndarray:
        """Negative log-likelihood of the observed counts. Lower is better."""
        y = np.asarray(y)
        n, p = self._nbinom_params()
        return -np.where(
            self.alpha <= 1e-12,
            stats.poisson.logpmf(y, self.mu),
            stats.nbinom.logpmf(y, n, p),
        )

    def crps(self, y: np.ndarray) -> np.ndarray:
        """Ranked probability score for counts.

        ``sum_k (F(k) - 1{y <= k})^2`` -- the discrete analogue of CRPS. Rewards putting
        probability mass near the truth, and unlike log-score it stays finite when a
        model assigns an observed value near-zero probability.
        """
        y = np.asarray(y, dtype=float)
        grid = np.arange(MAX_COUNT + 1)
        cdf = self.cdf(grid[None, :])
        indicator = (y[:, None] <= grid[None, :]).astype(float)
        return ((cdf - indicator) ** 2).sum(axis=1)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Unit Poisson deviance. The natural error measure for counts."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float).clip(min=1e-9)
    # y*log(y/mu) -> 0 as y -> 0, handled explicitly.
    term = np.where(y > 0, y * np.log(np.divide(y, mu, where=y > 0, out=np.ones_like(mu))), 0.0)
    return 2.0 * (term - (y - mu))


def score(
    y: np.ndarray, distribution: CountDistribution, *, label: str = ""
) -> dict[str, float]:
    """Every metric for one target/model pair."""
    y = np.asarray(y, dtype=float)
    mu = distribution.mu

    metrics = {
        "n": float(len(y)),
        "actual_mean": float(y.mean()),
        "pred_mean": float(mu.mean()),
        "MAE": float(np.abs(y - mu).mean()),
        "RMSE": float(np.sqrt(((y - mu) ** 2).mean())),
        "PoissonDev": float(poisson_deviance(y, mu).mean()),
        "LogScore": float(distribution.log_score(y).mean()),
        "CRPS": float(distribution.crps(y).mean()),
    }
    for k in (1, 2, 3):
        predicted = distribution.prob_at_least(k).mean()
        observed = float((y >= k).mean())
        metrics[f"P>={k}_pred"] = float(predicted)
        metrics[f"P>={k}_obs"] = observed
    if label:
        metrics["model"] = label
    return metrics


def calibration_table(
    y: np.ndarray, distribution: CountDistribution, k: int = 1, bins: int = 10
) -> pd.DataFrame:
    """Reliability of ``P(X >= k)``: predicted probability against observed frequency.

    A well-calibrated model has ``observed ~= predicted`` in every bin. A model can have
    excellent MAE and still be badly miscalibrated here, which is exactly why the v1
    project could not tell whether its predictions meant anything.
    """
    predicted = distribution.prob_at_least(k)
    observed = (np.asarray(y) >= k).astype(float)

    edges = np.unique(np.quantile(predicted, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A model that predicts (nearly) the same probability everywhere has no spread to
        # bin. That is still calibrated or not, so report it as one bin rather than NaN.
        index = np.zeros(len(predicted), dtype=int)
    else:
        index = np.clip(np.digitize(predicted, edges[1:-1]), 0, len(edges) - 2)
    frame = pd.DataFrame({"bin": index, "predicted": predicted, "observed": observed})
    out = frame.groupby("bin").agg(
        n=("observed", "size"),
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
    ).reset_index()
    return out


def expected_calibration_error(
    y: np.ndarray, distribution: CountDistribution, k: int = 1, bins: int = 10
) -> float:
    """Weighted mean gap between predicted and observed frequency. Lower is better."""
    table = calibration_table(y, distribution, k=k, bins=bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["predicted"] - table["observed"]).abs()).sum())


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Assemble per-model rows into the table the CLI prints, best log-score first."""
    frame = pd.DataFrame(results)
    ordered = [
        c for c in
        ["target", "model", "n", "actual_mean", "pred_mean", "LogScore", "CRPS",
         "PoissonDev", "MAE", "RMSE", "ECE", "P>=1_pred", "P>=1_obs"]
        if c in frame.columns
    ]
    remaining = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + remaining]
    if {"target", "LogScore"} <= set(frame.columns):
        frame = frame.sort_values(["target", "LogScore"])
    return frame.reset_index(drop=True)


def estimate_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments estimate of the negative binomial ``alpha``.

    Lets the baselines emit a genuine distribution rather than an arbitrary one, so they
    compete fairly on log-score and CRPS instead of being handicapped by construction.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float).clip(min=1e-9)
    excess = ((y - mu) ** 2 - mu).mean()
    scale = (mu**2).mean()
    return float(max(excess / scale, 0.0)) if scale > 0 else 0.0
