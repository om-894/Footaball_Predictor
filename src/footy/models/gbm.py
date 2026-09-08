"""Gradient boosting with a Poisson objective.

LightGBM handles the missing values that are structural in this data (a debutant has no
form) natively, and captures the interactions a GLM cannot -- a defender against a team
that crosses constantly is a different proposition from the same defender against a side
that plays through the middle.

Exposure enters through ``init_score``, LightGBM's equivalent of a GLM offset.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from footy.models.base import CountModel

log = logging.getLogger(__name__)


class PoissonGBM(CountModel):
    """LightGBM regression with ``objective="poisson"`` and a log-minutes offset."""

    name = "PoissonGBM"

    def __init__(self, seed: int = 42, **kwargs) -> None:
        super().__init__()
        self.seed = seed
        self.params = {
            "objective": "poisson",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "verbose": -1,
            **kwargs,
        }
        self.eval_set_: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None

    def set_validation(
        self, X: pd.DataFrame, y: np.ndarray, minutes: np.ndarray
    ) -> "PoissonGBM":
        """Supply a validation fold for early stopping.

        Deliberately explicit: the validation data must come from the fold, never from a
        random slice of training, or early stopping itself becomes a leak.
        """
        self.eval_set_ = (X, np.asarray(y, dtype=float), np.asarray(minutes, dtype=float))
        return self

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        self.model_ = LGBMRegressor(random_state=self.seed, **self.params)
        offset = np.log(np.clip(exposure, 1e-6, None))

        fit_kwargs = {"init_score": offset}
        if self.eval_set_ is not None:
            X_valid, y_valid, minutes_valid = self.eval_set_
            valid_offset = np.log(np.clip(minutes_valid / 90.0, 1e-6, None))
            fit_kwargs.update(
                eval_set=[(X_valid.reindex(columns=X.columns), y_valid)],
                eval_init_score=[valid_offset],
                eval_metric="poisson",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )

        self.model_.fit(X, y, **fit_kwargs)

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        # raw_score gives the linear predictor; exponentiating without the offset yields
        # the per-90 rate, matching every other model's contract.
        linear = self.model_.predict(
            X.reindex(columns=self.columns_), raw_score=True
        )
        return np.exp(np.clip(linear, -20, 20))

    def feature_importance(self, top: int = 25) -> pd.Series:
        """Gain-based importances, for the write-up rather than the model."""
        importances = pd.Series(
            self.model_.booster_.feature_importance(importance_type="gain"),
            index=self.columns_,
        )
        return importances.sort_values(ascending=False).head(top)
