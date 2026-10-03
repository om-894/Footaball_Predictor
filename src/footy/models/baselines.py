"""The models the neural network has to beat.

These exist to keep the project honest. A deep model that cannot outscore a player's own
exponentially weighted average is not adding anything, and without these rows in the
table there is no way to tell. The v1 project reported only its own training loss, so the
question could not even be asked.
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
    """One rate for everyone. The floor: any model below this is broken."""

    name = "GlobalMean"

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        self.rate_ = _pooled_rate(y, exposure)

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class PositionMean(CountModel):
    """Per-90 rate by position group. Cheap, and surprisingly hard to beat for tackles."""

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
    """The player's own exponentially weighted form rate.

    **This is the benchmark that matters.** It is roughly what an experienced analyst does
    by eye -- "he's been getting two shots a game lately" -- and it uses a single
    precomputed feature. Anything more elaborate has to justify itself against it.

    Uses the exposure-weighted rate (smoothed counts over smoothed minutes) rather than a
    smoothed per-90 rate. See ``NaivePer90EWMA`` for what the difference costs.

    A single scale factor is fitted so the baseline is not penalised for a systematic
    offset it could trivially correct.
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
        # A debutant has no EWMA; fall back to the shrunk career rate, which is defined
        # for everyone because it borrows from the positional prior.
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
    """``PlayerEWMA`` over smoothed per-90 rates instead of exposure-weighted ones.

    Included specifically to price the v1 pipeline's central assumption. It is the same
    model as ``PlayerEWMA`` in every other respect, so the gap between the two rows in the
    results table is exactly the cost of averaging rates instead of weighting by minutes.
    """

    name = "NaivePer90EWMA"

    def __init__(self, target: str, halflife: int = 6) -> None:
        super().__init__(target, halflife)
        self.column = f"{target}_p90_ewm{halflife}"


class ShrunkCareerRate(CountModel):
    """The empirical-Bayes career rate, used directly.

    Complements ``PlayerEWMA``: it weights a whole career rather than recent form, so it
    is steadier for fringe players and slower to react to a change in role.
    """

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
