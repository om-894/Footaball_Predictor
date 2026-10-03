"""
Simple baselines the other models have to beat.

PlayerEWMA, the player's own recent rate, is the benchmark in the results tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.config import EWMA_HALFLIVES
from footy.models.base import CountModel


def _pooled_rate(y: np.ndarray, exposure: np.ndarray) -> float:
    """Total count over total exposure, or 0 when there is no exposure."""
    total = exposure.sum()
    return float(y.sum() / total) if total > 0 else 0.0


class GlobalMean(CountModel):
    """The same per-90 rate for every player."""

    name = "GlobalMean"

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        self.rate_ = _pooled_rate(y, exposure)

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class PositionMean(CountModel):
    """One per-90 rate for each position group."""

    name = "PositionMean"

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        groups = [c for c in X.columns if c.startswith("posgrp_")]
        self.groups_ = groups
        self.rates_ = {}
        for group in groups:
            mask = X[group].to_numpy() == 1
            self.rates_[group] = _pooled_rate(y[mask], exposure[mask])
        self.default_ = _pooled_rate(y, exposure)

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        rate = np.full(len(X), self.default_)
        for group in self.groups_:
            if group in X.columns:
                mask = X[group].to_numpy() == 1
                rate[mask] = self.rates_[group]
        return rate


class PlayerEWMA(CountModel):
    """The player's recent rate (smoothed counts over smoothed minutes), times one fitted scale.

    This is the benchmark, roughly what an analyst would judge from recent form by eye.
    """

    name = "PlayerEWMA"

    def __init__(self, target: str, halflife: int = 6) -> None:
        super().__init__()
        if halflife not in EWMA_HALFLIVES:
            raise ValueError(f"halflife must be one of {EWMA_HALFLIVES}")
        self.target = target
        self.halflife = halflife
        self.column = f"{target}_rate{halflife}"
        self.fallback_column = f"{target}_career_p90"

    def _rates(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            raise KeyError(f"{self.column} missing; build features before fitting")
        rates = X[self.column]
        # a debutant has no recent form, so fall back to the shrunk career rate
        if self.fallback_column in X.columns:
            rates = rates.fillna(X[self.fallback_column])
        return rates.fillna(self.prior_).clip(lower=0.0).to_numpy()

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        self.prior_ = _pooled_rate(y, exposure)
        raw = self._rates(X)
        predicted = (raw * exposure).sum()
        self.scale_ = float(y.sum() / predicted) if predicted > 0 else 1.0

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return self._rates(X) * self.scale_


class NaivePer90EWMA(PlayerEWMA):
    """PlayerEWMA on smoothed per-90 rates instead, to show what averaging rates costs."""

    name = "NaivePer90EWMA"

    def __init__(self, target: str, halflife: int = 6) -> None:
        super().__init__(target, halflife)
        self.column = f"{target}_p90_ewm{halflife}"


class ShrunkCareerRate(CountModel):
    """The player's career rate pulled towards their position's, times one fitted scale."""

    name = "ShrunkCareerRate"

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        self.column = f"{target}_career_p90"

    def _rates(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].fillna(self.prior_).clip(lower=0.0).to_numpy()

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        self.prior_ = _pooled_rate(y, exposure)
        predicted = (self._rates(X) * exposure).sum()
        self.scale_ = float(y.sum() / predicted) if predicted > 0 else 1.0

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return self._rates(X) * self.scale_
