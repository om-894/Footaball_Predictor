"""Common interface for every count model in the ladder.

All models -- baselines, GLMs, gradient boosting, the neural net -- expose the same three
methods, so ``evaluate`` can score them identically and the comparison table is honest.

The shared contract is a *rate* model. Each predicts an expected count per 90 minutes;
the exposure (minutes) is applied on top. This is what makes the comparison meaningful:
no model gets to smuggle in extra information about how long the player was on the pitch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from footy.config import FULL_MATCH_MINUTES
from footy.evaluate import CountDistribution, estimate_dispersion


class CountModel(ABC):
    """Predicts a distribution over a non-negative count for a player-match."""

    name: str = "base"

    def __init__(self) -> None:
        self.alpha_: float = 0.0
        self.columns_: list[str] = []

    @abstractmethod
    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        """Fit the per-90 rate. ``exposure`` is minutes / 90."""

    @abstractmethod
    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted count per 90 minutes."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, minutes: np.ndarray) -> "CountModel":
        self.columns_ = list(X.columns)
        exposure = np.asarray(minutes, dtype=float) / FULL_MATCH_MINUTES
        y = np.asarray(y, dtype=float)

        self._fit_rate(X, y, exposure)

        # Overdispersion is estimated on the training fit, so baselines and the neural
        # net are scored under comparably shaped distributions.
        fitted = self._predict_rate(X) * exposure
        self.alpha_ = estimate_dispersion(y, fitted)
        return self

    def predict(self, X: pd.DataFrame, minutes: np.ndarray) -> np.ndarray:
        """Expected count for the given minutes."""
        exposure = np.asarray(minutes, dtype=float) / FULL_MATCH_MINUTES
        return self._predict_rate(X) * exposure

    def predict_distribution(
        self, X: pd.DataFrame, minutes: np.ndarray
    ) -> CountDistribution:
        return CountDistribution(self.predict(X, minutes), self.alpha_)


def log_exposure(exposure: np.ndarray) -> np.ndarray:
    """Log of minutes / 90, floored so a zero can never give -inf."""
    return np.log(np.clip(exposure, 1e-6, None))


def clean_matrix(X: pd.DataFrame, columns: list[str], medians: pd.Series | None = None):
    """Align columns and fill gaps.

    Missing values here are structural -- a debutant has no form -- so they are filled
    with the *training* fold's medians. Computing them on the full dataset would be the
    same leak the v1 pipeline had when it fit its scaler before splitting.
    """
    frame = X.reindex(columns=columns)
    if medians is None:
        medians = frame.median(numeric_only=True)
    return frame.fillna(medians).replace([np.inf, -np.inf], 0.0), medians
