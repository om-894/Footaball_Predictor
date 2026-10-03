"""
Count distributions and the metrics used to score them.

Every model predicts a full distribution over counts, so the results can say things like
P(fouls >= 1) = 54%. The main metrics, log score and CRPS, reward a well calibrated
distribution. MAE is reported too, but always predicting zero scores well on it for a
target that is usually zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# highest count summed over for CRPS and calibration, far above anything in the data
MAX_COUNT = 40


@dataclass
class CountDistribution:
    """A negative binomial distribution for each row, with mean `mu` and dispersion `alpha`.

    Var = mu + alpha * mu^2, so alpha = 0 is a Poisson.
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
        # scipy's nbinom(n, p) has mean n(1-p)/p, so n = 1/alpha and p = n / (n + mu)
        alpha = np.where(self.alpha <= 1e-12, 1e-12, self.alpha)
        n = 1.0 / alpha
        p = n / (n + self.mu)
        return n, p

    def _align(self, k: np.ndarray | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shape the per-row parameters to match `k`, which is one value per row or a 2-D grid."""
        n, p = self._nbinom_params()
        k = np.asarray(k)
        if k.ndim == 2:
            return k, n[:, None], p[:, None]
        return k, n, p

    def _mu_for(self, k: np.ndarray) -> np.ndarray:
        return self.mu[:, None] if k.ndim == 2 else self.mu

    def _evaluate(self, k: np.ndarray | int, nbinom_fn, poisson_fn) -> np.ndarray:
        """Apply the negative binomial function, or the exact Poisson one where alpha is zero."""
        k, n, p = self._align(k)
        poisson_rows = self.alpha <= 1e-12

        # usually every row is one kind, so only one function runs
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
        """P(X >= k) for each row."""
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
        """Ranked probability score, sum over k of (F(k) - 1{y <= k})^2. Lower is better."""
        y = np.asarray(y, dtype=float)
        grid = np.arange(MAX_COUNT + 1)
        cdf = self.cdf(grid[None, :])
        indicator = (y[:, None] <= grid[None, :]).astype(float)
        return ((cdf - indicator) ** 2).sum(axis=1)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson deviance for each row."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float).clip(min=1e-9)
    # y * log(y / mu) is taken as 0 when y is 0
    term = np.where(y > 0, y * np.log(np.divide(y, mu, where=y > 0, out=np.ones_like(mu))), 0.0)
    return 2.0 * (term - (y - mu))


def score(
    y: np.ndarray, distribution: CountDistribution, *, label: str = ""
) -> dict[str, float]:
    """Every metric for one target and model."""
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
    """Predicted P(X >= k) against how often it actually happened, in quantile bins."""
    predicted = distribution.prob_at_least(k)
    observed = (np.asarray(y) >= k).astype(float)

    edges = np.unique(np.quantile(predicted, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # a model that predicts nearly the same probability everywhere gets a single bin
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
    """Average gap between predicted and observed frequency, weighted by bin size. Lower is better."""
    table = calibration_table(y, distribution, k=k, bins=bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["predicted"] - table["observed"]).abs()).sum())


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Per-model score rows as one table, best log score first within each target."""
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
    """Method of moments estimate of the negative binomial alpha, so every model gets a real distribution."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float).clip(min=1e-9)
    excess = ((y - mu) ** 2 - mu).mean()
    scale = (mu**2).mean()
    return float(max(excess / scale, 0.0)) if scale > 0 else 0.0
