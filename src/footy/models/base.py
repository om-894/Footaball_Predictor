"""
The shared interface for every count model.

Each model predicts a per-90 rate and the base class multiplies it by minutes / 90, so
the only way a model sees how long a player was on the pitch is through that offset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from footy.config import FULL_MATCH_MINUTES
from footy.evaluate import CountDistribution, estimate_dispersion


class CountModel(ABC):
    """Predicts a distribution over a non-negative count for each player-match."""

    name: str = "base"

    def __init__(self) -> None:
        self.alpha_: float = 0.0
        self.columns_: list[str] = []

    @abstractmethod
    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        """Fit the per-90 rate. `exposure` is minutes / 90."""

    @abstractmethod
    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted count per 90 minutes."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, minutes: np.ndarray) -> "CountModel":
        """Fit the rate, then estimate the dispersion from the fitted training counts."""
        self.columns_ = list(X.columns)
        exposure = np.asarray(minutes, dtype=float) / FULL_MATCH_MINUTES
        y = np.asarray(y, dtype=float)

        self._fit_rate(X, y, exposure)

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
    """Pick `columns` and fill gaps, e.g. a debutant's missing form, with the training medians."""
    frame = X.reindex(columns=columns)
    if medians is None:
        medians = frame.median(numeric_only=True)
    return frame.fillna(medians).replace([np.inf, -np.inf], 0.0), medians
