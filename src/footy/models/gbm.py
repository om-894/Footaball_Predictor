"""
LightGBM with a Poisson objective and a log-minutes offset.

The offset goes in through `init_score`, which plays the same part as a GLM offset.
LightGBM also copes with missing values itself, e.g. a debutant's missing form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from footy.config import FULL_MATCH_MINUTES
from footy.models.base import CountModel, log_exposure


class PoissonGBM(CountModel):
    """LightGBM regression with objective="poisson" and a log-minutes offset."""

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
        """Use the fold's validation season for early stopping."""
        self.eval_set_ = (X, np.asarray(y, dtype=float), np.asarray(minutes, dtype=float))
        return self

    def _fit_rate(self, X: pd.DataFrame, y: np.ndarray, exposure: np.ndarray) -> None:
        self.model_ = LGBMRegressor(random_state=self.seed, **self.params)
        offset = log_exposure(exposure)

        fit_kwargs = {"init_score": offset}
        if self.eval_set_ is not None:
            X_valid, y_valid, minutes_valid = self.eval_set_
            valid_offset = log_exposure(minutes_valid / FULL_MATCH_MINUTES)
            fit_kwargs.update(
                eval_set=[(X_valid.reindex(columns=X.columns), y_valid)],
                eval_init_score=[valid_offset],
                eval_metric="poisson",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )

        self.model_.fit(X, y, **fit_kwargs)

    def _predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        # raw_score is the linear predictor without the offset, so exp of it is the per-90 rate
        linear = self.model_.predict(
            X.reindex(columns=self.columns_), raw_score=True
        )
        return np.exp(np.clip(linear, -20, 20))
