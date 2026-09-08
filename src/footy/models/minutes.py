"""Stage one: how long will the player be on the pitch?

Every count we forecast scales with minutes, so getting this wrong contaminates
everything downstream. Minutes are awkward to model: bounded to [1, 90], and strongly
bimodal -- the median is 90 and the lower quartile 55, because players tend to either
start and finish or come off the bench.

A plain regression to the conditional mean lands in the empty middle of that
distribution. Instead we predict a *distribution* over minutes with a quantile model, and
let the count models integrate over it.

Known limitation: the source data contains only players who actually appeared, so this
model is conditional on selection. It cannot tell you whether a player will be picked --
only how long they are likely to play if they are. ``footy.sources.fbref_live`` can
supply lineups including unused substitutes, which would lift the restriction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from footy.config import FULL_MATCH_MINUTES

log = logging.getLogger(__name__)

#: Quantiles used to represent the minutes distribution.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


class MinutesModel:
    """Quantile gradient boosting over minutes played, given selection."""

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES, seed: int = 42, **kwargs):
        self.quantiles = quantiles
        self.seed = seed
        self.params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "verbose": -1,
            **kwargs,
        }
        self.models: dict[float, LGBMRegressor] = {}
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MinutesModel":
        self.columns = list(X.columns)
        for quantile in self.quantiles:
            model = LGBMRegressor(
                objective="quantile", alpha=quantile, random_state=self.seed, **self.params
            )
            model.fit(X, y)
            self.models[quantile] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """One column per quantile, each clipped to a legal number of minutes."""
        predictions = {
            q: np.clip(model.predict(X[self.columns]), 1.0, FULL_MATCH_MINUTES)
            for q, model in self.models.items()
        }
        frame = pd.DataFrame(predictions, index=X.index)
        # Quantile models are fitted independently and can cross; sorting each row
        # restores monotonicity without materially changing the fit.
        sorted_values = np.sort(frame.to_numpy(), axis=1)
        return pd.DataFrame(sorted_values, columns=list(self.quantiles), index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Expected minutes, as a trapezoidal average over the predicted quantiles."""
        quantiles = self.predict_quantiles(X)
        levels = np.asarray(self.quantiles)
        values = quantiles.to_numpy()

        # Weight each quantile by the probability mass it represents.
        edges = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [1.0]])
        weights = np.diff(edges)
        return values @ weights


class NaiveMinutesModel:
    """Fallback that predicts each player's recent average minutes.

    Useful as a reference point, and as a stand-in when a fold has too little history to
    fit the gradient-boosted version.
    """

    def __init__(self, column: str = "Min_ewm6") -> None:
        self.column = column
        self.fallback = FULL_MATCH_MINUTES

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NaiveMinutesModel":
        self.fallback = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            return np.full(len(X), self.fallback)
        return X[self.column].fillna(self.fallback).clip(1.0, FULL_MATCH_MINUTES).to_numpy()
