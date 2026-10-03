"""
Stage one: how many minutes a player will play.

Minutes pile up at 90 and at short substitute spells, so a model of the average would
predict the empty middle. Instead several quantiles are predicted and averaged.

The data only holds players who actually played, so this predicts minutes if picked,
not whether a player will be picked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from footy.config import FULL_MATCH_MINUTES

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9) # quantiles predicted to describe the minutes spread


class MinutesModel:
    """Quantile LightGBM models of minutes played, given the player appears."""

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
        """One column per quantile, each clipped to between 1 and 90 minutes."""
        predictions = {
            q: np.clip(model.predict(X[self.columns]), 1.0, FULL_MATCH_MINUTES)
            for q, model in self.models.items()
        }
        frame = pd.DataFrame(predictions, index=X.index)
        # the quantile models are fitted separately and can cross, so sort each row
        sorted_values = np.sort(frame.to_numpy(), axis=1)
        return pd.DataFrame(sorted_values, columns=list(self.quantiles), index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Expected minutes, averaging the quantiles weighted by the probability each covers."""
        quantiles = self.predict_quantiles(X)
        levels = np.asarray(self.quantiles)
        values = quantiles.to_numpy()

        edges = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [1.0]])
        weights = np.diff(edges)
        return values @ weights
